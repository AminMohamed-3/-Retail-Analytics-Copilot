"""LangGraph workflow for hybrid RAG + SQL agent."""
from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END
import json
import re
import os
from datetime import datetime
from pathlib import Path

from config import setup_dspy
from agent.dspy_signatures import Router, NLToSQL, Synthesizer
from agent.rag.retrieval import DocumentRetriever, Chunk
from agent.tools.sqlite_tool import SQLiteTool


class AgentState(TypedDict):
    """State for the agent workflow."""
    question: str
    question_id: str
    format_hint: str
    query_type: Optional[str]
    retrieved_chunks: List[Chunk]
    context: str
    sql_query: str
    sql_results: dict
    final_answer: Any
    explanation: str
    citations: List[str]
    confidence: float
    repair_count: int
    repair_type: Optional[str]
    error: Optional[str]
    db_tables_used: List[str]
    trace: List[str]


class HybridAgent:
    """Hybrid RAG + SQL agent using LangGraph."""
    
    def __init__(self, trace_log_dir: str = "traces", use_optimized: bool = False):
        setup_dspy()
        
        self.router = Router()
        if os.path.exists("optimized_router_bootstrap.json"):
            try:
                self.router.load("optimized_router_bootstrap.json")
            except:
                pass
        
        self.nl_to_sql = NLToSQL()
        if use_optimized and os.path.exists("optimized_nl_to_sql.json"):
            try:
                self.nl_to_sql.load("optimized_nl_to_sql.json")
            except:
                pass
        
        self.synthesizer = Synthesizer()
        
        self.retriever = DocumentRetriever()
        self.retriever.load_documents()
        self.sql_tool = SQLiteTool()
        self.schema_info = self.sql_tool.get_full_schema_summary()
        
        self.trace_log_dir = Path(trace_log_dir)
        self.trace_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("router", self._router_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("nl_sql", self._nl_sql_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("repair", self._repair_node)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {"rag": "retriever", "sql": "nl_sql", "hybrid": "retriever"}
        )
        
        workflow.add_edge("retriever", "planner")
        
        workflow.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {"skip_sql": "synthesizer", "need_sql": "nl_sql"}
        )
        
        workflow.add_edge("nl_sql", "executor")
        
        workflow.add_conditional_edges(
            "executor",
            self._route_after_executor,
            {"synthesize": "synthesizer", "repair": "repair"}
        )
        
        workflow.add_conditional_edges(
            "repair",
            self._route_after_repair,
            {"nl_sql": "nl_sql", "synthesizer": "synthesizer", "end": END}
        )
        
        workflow.add_conditional_edges(
            "synthesizer",
            self._route_after_synthesizer,
            {"end": END, "repair": "repair"}
        )
        
        return workflow.compile()
    
    def _log_trace(self, state: AgentState, message: str, console: bool = True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        state["trace"].append(log_entry)
        if console:
            print(log_entry)
    
    def _router_node(self, state: AgentState) -> AgentState:
        self._log_trace(state, "Router: Classifying question type")
        query_type = self.router.forward(state["question"])
        state["query_type"] = query_type
        self._log_trace(state, f"Router: Determined type = {query_type}")
        return state
    
    def _route_after_router(self, state: AgentState) -> str:
        return state["query_type"]
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        self._log_trace(state, "Retriever: Fetching relevant documents")
        chunks = self.retriever.retrieve(state["question"], top_k=5)
        state["retrieved_chunks"] = chunks
        chunk_details = ", ".join([f"{c.chunk_id}({c.score:.3f})" for c in chunks[:3]])
        self._log_trace(state, f"Retriever: Found {len(chunks)} chunks. Top: {chunk_details}")
        return state
    
    def _planner_node(self, state: AgentState) -> AgentState:
        self._log_trace(state, "Planner: Extracting constraints from documents")
        
        context_parts = []
        for chunk in state["retrieved_chunks"]:
            context_parts.append(f"[{chunk.chunk_id}]: {chunk.content}")
        
        context = "\n\n".join(context_parts)
        
        extracted_info = []
        
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, context)
        if dates:
            extracted_info.append(f"Dates found: {', '.join(set(dates))}")
        
        if "AOV" in context or "Average Order Value" in context:
            extracted_info.append("KPI: AOV = SUM(UnitPrice*Qty*(1-Discount))/COUNT(DISTINCT OrderID)")
        if "Gross Margin" in context or "GM" in context:
            extracted_info.append("KPI: Margin = Revenue - (0.7 * Revenue) when cost unknown")
        
        category_pattern = r'(Beverages|Condiments|Confections|Dairy Products|Grains/Cereals|Meat/Poultry|Produce|Seafood)'
        categories = re.findall(category_pattern, context)
        if categories:
            extracted_info.append(f"Categories: {', '.join(set(categories))}")
        
        if extracted_info:
            context = "\n".join(extracted_info) + "\n\n" + context
        
        state["context"] = context
        self._log_trace(state, f"Planner: Extracted constraints - Dates: {len(dates)}, KPIs: {len([x for x in extracted_info if 'KPI' in x])}, Categories: {len(categories)}")
        return state
    
    def _route_after_planner(self, state: AgentState) -> str:
        if state["query_type"] == "rag":
            return "skip_sql"
        return "need_sql"
    
    def _validate_sql_preflight(self, sql: str) -> tuple:
        """Pre-flight validation to catch common hallucinations before execution."""
        sql_upper = sql.upper()
        
        # Check for text-as-date hallucination: BETWEEN 'Some Text' without dates
        bad_between = re.search(r"BETWEEN\s+'[A-Z][A-Za-z\s]+'(?!\s*AND\s*'\d)", sql, re.IGNORECASE)
        if bad_between:
            return False, "Invalid date: use format 'YYYY-MM-DD' not text strings"
        
        # Check for undefined alias 'c.' when Categories is meant
        if re.search(r"\bc\.CategoryName\b", sql) or re.search(r"\bc\.CategoryID\b", sql):
            if "Categories c" not in sql and "categories c" not in sql:
                return False, "Alias 'c' used for CategoryName but Categories not aliased as 'c'. Use 'cat' for Categories"
        
        # Check for hallucinated keywords
        hallucinations = ["BETWELOGOF", "BETWEDIR", "strftDirty", "BETWEWS"]
        for h in hallucinations:
            if h in sql_upper:
                return False, f"Invalid keyword '{h}' - use standard SQL"
        
        # Check for SUM(OrderDate BETWEEN...) pattern - common hallucination
        if re.search(r"SUM\s*\([^)]*BETWEEN", sql, re.IGNORECASE):
            return False, "Invalid: Cannot SUM a BETWEEN expression. Use BETWEEN in WHERE clause only"
        
        return True, ""
    
    def _nl_sql_node(self, state: AgentState) -> AgentState:
        self._log_trace(state, "NL-SQL: Generating SQL query")
        
        context = state.get("context", "")
        sql_query = self.nl_to_sql.forward(
            question=state["question"],
            schema_info=self.schema_info,
            context=context
        )
        
        # Pre-flight validation
        valid, error_msg = self._validate_sql_preflight(sql_query)
        if not valid:
            self._log_trace(state, f"NL-SQL: Pre-flight validation failed - {error_msg}")
            state["sql_query"] = sql_query
            state["error"] = error_msg
            state["sql_results"] = {"success": False, "error": error_msg, "rows": [], "row_count": 0, "columns": []}
        else:
            state["sql_query"] = sql_query
        
        self._log_trace(state, f"NL-SQL: Generated query ({len(sql_query)} chars): {sql_query[:80]}...")
        return state
    
    def _executor_node(self, state: AgentState) -> AgentState:
        # Skip execution if pre-flight failed
        if state.get("error"):
            self._log_trace(state, f"Executor: Skipped (pre-flight error)")
            return state
        
        self._log_trace(state, "Executor: Executing SQL query")
        
        sql_results = self.sql_tool.execute_query(state["sql_query"])
        state["sql_results"] = sql_results
        
        tables_used = self._extract_table_names(state["sql_query"])
        state["db_tables_used"] = list(set(state.get("db_tables_used", []) + tables_used))
        
        if sql_results["success"]:
            self._log_trace(state, f"Executor: Query succeeded - {sql_results['row_count']} rows, columns: {sql_results['columns']}")
        else:
            state["error"] = sql_results["error"]
            self._log_trace(state, f"Executor: Query failed - {sql_results['error']}")
        
        return state
    
    def _extract_table_names(self, sql: str) -> List[str]:
        tables = []
        patterns = [r'FROM\s+["\']?(\w+)["\']?', r'JOIN\s+["\']?(\w+)["\']?']
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.extend(matches)
        
        mapping = {
            "orders": "Orders",
            "order_items": "Order Details",
            "products": "Products",
            "customers": "Customers"
        }
        
        normalized = []
        for table in tables:
            normalized.append(mapping.get(table.lower(), table))
        
        return normalized
    
    def _route_after_executor(self, state: AgentState) -> str:
        repair_count = state.get("repair_count", 0)
        
        if state.get("error") or not state["sql_results"].get("success", False):
            if repair_count < 2:
                self._log_trace(state, f"Routing to repair (attempt {repair_count + 1}/2)")
                return "repair"
            else:
                self._log_trace(state, "Max repairs reached, synthesizing with error")
                return "synthesize"
        
        if state["sql_results"]["row_count"] == 0 and repair_count < 2:
            self._log_trace(state, f"Empty results, routing to repair (attempt {repair_count + 1}/2)")
            return "repair"
        
        return "synthesize"
    
    def _route_after_synthesizer(self, state: AgentState) -> str:
        repair_count = state.get("repair_count", 0)
        
        format_valid = self._validate_answer_format(state["final_answer"], state["format_hint"])
        if not format_valid and repair_count < 2:
            self._log_trace(state, f"Invalid answer format, routing to repair (attempt {repair_count + 1}/2)")
            return "repair"
        
        return "end"
    
    def _validate_answer_format(self, answer: any, format_hint: str) -> bool:
        if answer is None:
            return False
        
        if format_hint == "int":
            return isinstance(answer, int)
        elif format_hint == "float":
            return isinstance(answer, (int, float))
        elif format_hint.startswith("list["):
            return isinstance(answer, list)
        elif format_hint.startswith("{"):
            return isinstance(answer, dict) and len(answer) > 0
        
        return True
    
    def _get_dynamic_repair_hints(self, error_msg: str, previous_query: str) -> str:
        """Generate targeted repair hints based on error type."""
        error_lower = error_msg.lower() if error_msg else ""
        hints = []
        
        if "no such column" in error_lower:
            # Parse which column is missing
            col_match = re.search(r"no such column:\s*(\w+\.?\w*)", error_lower)
            if col_match:
                bad_col = col_match.group(1)
                hints.append(f"Column '{bad_col}' not found.")
                
                if "c.categoryname" in bad_col.lower():
                    hints.append("FIX: Use 'cat.CategoryName' - alias 'cat' for Categories table.")
                    hints.append("JOIN Categories cat ON p.CategoryID = cat.CategoryID")
                elif "c." in bad_col.lower() and "category" in bad_col.lower():
                    hints.append("FIX: Use 'cat' alias for Categories, 'c' for customers.")
                elif "cats." in bad_col.lower():
                    hints.append("FIX: Use 'cat' not 'cats' as Categories alias.")
        
        elif "syntax error" in error_lower:
            hints.append("SQL syntax error detected.")
            if "BETWEEN" in previous_query.upper():
                hints.append("Check BETWEEN syntax: column BETWEEN 'date1' AND 'date2'")
            hints.append("Ensure all parentheses are balanced.")
            hints.append("Use standard SQLite syntax only.")
        
        elif "0 rows" in error_lower or "empty" in error_lower:
            hints.append("Query returned no results.")
            hints.append("Check date format: use '2017-MM-DD' (dates shifted +20 years)")
            hints.append("Check WHERE conditions are not too restrictive.")
        
        # Add universal hints if specific hints found
        if hints:
            hints.append("\nALIAS REMINDER: o=orders, od=order_items, p=products, cat=Categories, c=customers")
        else:
            # Generic fallback
            hints.append("Previous query failed. Check table names and aliases.")
            hints.append("ALIASES: o=orders, od=order_items, p=products, cat=Categories, c=customers")
        
        return "\n".join(hints)
    
    def _repair_node(self, state: AgentState) -> AgentState:
        repair_count = state.get("repair_count", 0) + 1
        state["repair_count"] = repair_count
        self._log_trace(state, f"Repair: Attempt {repair_count}/2")
        
        has_sql_error = state.get("error") or not state.get("sql_results", {}).get("success", False)
        has_empty_results = state.get("sql_results", {}).get("success", False) and state.get("sql_results", {}).get("row_count", 0) == 0
        has_output_issue = state.get("final_answer") is not None and not self._validate_answer_format(state.get("final_answer"), state.get("format_hint", ""))
        
        if has_sql_error or has_empty_results:
            if has_empty_results:
                self._log_trace(state, "Repair: Fixing SQL query (empty results)")
            else:
                self._log_trace(state, "Repair: Fixing SQL query")
            
            error_msg = state.get("error", "Query returned 0 rows")
            previous_query = state.get("sql_query", "")
            
            # Generate targeted hints instead of dumping full schema
            dynamic_hints = self._get_dynamic_repair_hints(error_msg, previous_query)
            
            repair_context = f"REPAIR NEEDED:\n"
            repair_context += f"Error: {error_msg}\n"
            repair_context += f"Failed query: {previous_query[:200]}...\n\n"
            repair_context += f"HINTS:\n{dynamic_hints}"
            
            # Keep original context but add repair info
            original_context = state.get("context", "")
            # Limit context to prevent overflow
            if len(original_context) > 500:
                original_context = original_context[:500] + "..."
            
            state["context"] = original_context + "\n\n" + repair_context
            
            state["error"] = None
            state["final_answer"] = None
            state["citations"] = []
            state["repair_type"] = "sql"
            self._log_trace(state, "Repair: Added targeted hints for SQL regeneration")
        
        elif has_output_issue:
            self._log_trace(state, "Repair: Fixing output format")
            repair_context = f"Previous answer had format issues. Format required: {state['format_hint']}"
            state["context"] = state.get("context", "") + "\n\n" + repair_context
            
            doc_chunks = state.get("retrieved_chunks", [])
            
            result = self.synthesizer.forward(
                question=state["question"],
                sql_results=state.get("sql_results", {}),
                format_hint=state["format_hint"],
                context=state.get("context", ""),
                db_tables_used=state.get("db_tables_used", []),
                doc_chunks_used=doc_chunks
            )
            
            final_answer = self._format_answer(
                result["final_answer"],
                state["format_hint"],
                state["sql_results"]
            )
            
            state["final_answer"] = final_answer
            state["explanation"] = result["explanation"]
            state["citations"] = result["citations"]
            state["repair_type"] = "output"
            self._log_trace(state, f"Repair: Regenerated answer - Type: {type(final_answer).__name__}")
        else:
            state["repair_type"] = "end"
        
        return state
    
    def _route_after_repair(self, state: AgentState) -> str:
        repair_count = state.get("repair_count", 0)
        repair_type = state.get("repair_type", "sql")
        
        if repair_count >= 2:
            self._log_trace(state, "Max repairs reached, ending")
            return "end"
        
        if repair_type == "sql":
            return "nl_sql"
        elif repair_type == "output":
            return "synthesizer"
        return "end"
    
    def _synthesizer_node(self, state: AgentState) -> AgentState:
        self._log_trace(state, "Synthesizer: Formatting final answer")
        
        if state["query_type"] == "rag" and not state.get("sql_results"):
            state["sql_results"] = {"success": True, "rows": [], "row_count": 0, "columns": []}
        
        doc_chunks = state.get("retrieved_chunks", [])
        
        try:
            result = self.synthesizer.forward(
                question=state["question"],
                sql_results=state.get("sql_results", {}),
                format_hint=state["format_hint"],
                context=state.get("context", ""),
                db_tables_used=state.get("db_tables_used", []),
                doc_chunks_used=doc_chunks
            )
            
            final_answer = self._format_answer(
                result.get("final_answer"),
                state["format_hint"],
                state["sql_results"]
            )
            
            state["final_answer"] = final_answer
            state["explanation"] = result.get("explanation", "Answer from SQL results.")
            state["citations"] = result.get("citations", [])
        except Exception as e:
            self._log_trace(state, f"Synthesizer failed: {e}, extracting from SQL directly")
            
            final_answer = self._format_answer(
                None,
                state["format_hint"],
                state["sql_results"]
            )
            
            state["final_answer"] = final_answer
            state["explanation"] = f"Direct extraction from SQL results."
            
            citations = []
            if state.get("db_tables_used"):
                citations.extend(state["db_tables_used"])
            if doc_chunks:
                citations.extend([chunk.chunk_id for chunk in doc_chunks])
            state["citations"] = citations
        
        state["confidence"] = self._calculate_confidence(state)
        
        self._log_trace(state, f"Synthesizer: Answer formatted - Type: {type(state['final_answer']).__name__}, Citations: {len(state['citations'])}")
        return state
    
    def _validate_citations(self, state: AgentState) -> bool:
        if state.get("sql_query") and state.get("db_tables_used"):
            db_citations = [c for c in state.get("citations", []) if c in state["db_tables_used"]]
            if not db_citations:
                return False
        
        if state.get("retrieved_chunks") and state["query_type"] in ["rag", "hybrid"]:
            doc_citations = [c for c in state.get("citations", []) if "::chunk" in c]
            if not doc_citations:
                return False
        
        return True
    
    def _extract_from_sql_results(self, sql_results: dict, format_hint: str) -> Any:
        if not sql_results.get("success") or not sql_results.get("rows"):
            return None
        
        rows = sql_results["rows"]
        
        if format_hint == "int":
            if rows:
                for val in rows[0].values():
                    try:
                        return int(float(val))
                    except:
                        continue
            return 0
        
        elif format_hint == "float":
            if rows:
                for val in rows[0].values():
                    try:
                        return round(float(val), 2)
                    except:
                        continue
            return 0.0
        
        elif format_hint.startswith("list["):
            inner_match = re.search(r'\{([^}]+)\}', format_hint)
            if inner_match:
                fields = {}
                for field in inner_match.group(1).split(','):
                    field = field.strip()
                    if ':' in field:
                        key, _ = field.split(':', 1)
                        fields[key.strip()] = True
                
                formatted_rows = []
                for row in rows[:10]:
                    formatted_row = {}
                    for col_name, col_value in row.items():
                        col_lower = col_name.lower()
                        for field_name in fields.keys():
                            if field_name.lower() in col_lower or col_lower in field_name.lower():
                                formatted_row[field_name] = col_value
                                break
                        if not formatted_row:
                            formatted_row = dict(row)
                            break
                    formatted_rows.append(formatted_row)
                return formatted_rows[:3] if "top 3" in format_hint.lower() else formatted_rows
            return rows[:10]
        
        elif format_hint.startswith("{"):
            if rows:
                row = rows[0]
                inner_match = re.search(r'\{([^}]+)\}', format_hint)
                if inner_match:
                    fields = {}
                    for field in inner_match.group(1).split(','):
                        field = field.strip()
                        if ':' in field:
                            key, _ = field.split(':', 1)
                            fields[key.strip()] = True
                    
                    formatted_row = {}
                    for col_name, col_value in row.items():
                        col_lower = col_name.lower()
                        for field_name in fields.keys():
                            if field_name.lower() in col_lower or col_lower in field_name.lower():
                                formatted_row[field_name] = col_value
                                break
                    if formatted_row:
                        return formatted_row
                return row
            return {}
        
        return None
    
    def _format_answer(self, answer_str: str, format_hint: str, sql_results: dict) -> Any:
        if answer_str is None or (isinstance(answer_str, str) and not answer_str.strip()):
            if sql_results.get("success") and sql_results.get("rows"):
                return self._extract_from_sql_results(sql_results, format_hint)
            return None
        
        if isinstance(answer_str, str):
            answer_str = answer_str.strip()
        else:
            return answer_str
        
        if format_hint == "int":
            match = re.search(r'\d+', answer_str)
            if match:
                return int(match.group())
            try:
                return int(float(answer_str))
            except:
                if sql_results.get("success") and sql_results.get("rows"):
                    for val in sql_results["rows"][0].values():
                        try:
                            return int(float(val))
                        except:
                            continue
                return 0
        
        elif format_hint == "float":
            match = re.search(r'\d+\.?\d*', answer_str)
            if match:
                return round(float(match.group()), 2)
            try:
                return round(float(answer_str), 2)
            except:
                if sql_results.get("success") and sql_results.get("rows"):
                    for val in sql_results["rows"][0].values():
                        try:
                            return round(float(val), 2)
                        except:
                            continue
                return 0.0
        
        elif format_hint.startswith("list["):
            json_match = re.search(r'\[.*?\]', answer_str, re.DOTALL)
            if json_match:
                json_str = json_match.group().replace("'", '"')
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
            
            if sql_results.get("success") and sql_results.get("rows"):
                return self._extract_from_sql_results(sql_results, format_hint)
            return []
        
        elif format_hint.startswith("{"):
            json_match = re.search(r'\{.*?\}', answer_str, re.DOTALL)
            if json_match:
                json_str = json_match.group().replace("'", '"')
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        return parsed
                except:
                    pass
            
            if sql_results.get("success") and sql_results.get("rows"):
                return self._extract_from_sql_results(sql_results, format_hint)
            return {}
        
        return answer_str
    
    def _calculate_confidence(self, state: AgentState) -> float:
        confidence = 0.5
        
        if state["sql_results"].get("success"):
            confidence += 0.2
        
        if state["sql_results"].get("row_count", 0) > 0:
            confidence += 0.1
        
        if state.get("retrieved_chunks"):
            avg_score = sum(c.score for c in state["retrieved_chunks"]) / len(state["retrieved_chunks"])
            confidence += min(avg_score * 0.2, 0.2)
        
        repair_count = state.get("repair_count", 0)
        confidence -= repair_count * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def process_question(self, question: str, question_id: str, format_hint: str) -> dict:
        initial_state: AgentState = {
            "question": question,
            "question_id": question_id,
            "format_hint": format_hint,
            "query_type": None,
            "retrieved_chunks": [],
            "context": "",
            "sql_query": "",
            "sql_results": {},
            "final_answer": None,
            "explanation": "",
            "citations": [],
            "confidence": 0.0,
            "repair_count": 0,
            "repair_type": None,
            "error": None,
            "db_tables_used": [],
            "trace": []
        }
        
        self._log_trace(initial_state, f"=== Processing question: {question_id} ===")
        self._log_trace(initial_state, f"Question: {question}")
        self._log_trace(initial_state, f"Format hint: {format_hint}")
        
        final_state = self.graph.invoke(initial_state)
        
        self._log_trace(final_state, f"=== Completed question: {question_id} ===")
        self._log_trace(final_state, f"Final answer type: {type(final_state['final_answer']).__name__}")
        self._log_trace(final_state, f"Confidence: {final_state['confidence']:.2f}")
        
        trace_file = self.trace_log_dir / f"{question_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(trace_file, "w") as f:
            f.write("\n".join(final_state["trace"]))
        
        return {
            "id": question_id,
            "final_answer": final_state["final_answer"],
            "sql": final_state["sql_query"],
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"]
        }
