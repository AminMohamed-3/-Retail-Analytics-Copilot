"""DSPy signatures and modules for Router, NL-SQL, and Synthesizer."""
import dspy
import re
import json


class RouterSignature(dspy.Signature):
    """Classify what data sources a question needs.
    
    RULES:
    - rag: Questions about policies, definitions, or document content only
    - sql: Questions needing database queries (counts, totals, lists, rankings)
    - hybrid: Questions referencing BOTH docs (campaigns, KPIs) AND database data
    
    Examples:
    - "What is the return policy?" → rag
    - "How many orders in 1997?" → sql  
    - "Revenue during Summer Beverages 1997" → hybrid (needs campaign dates + SQL)
    """
    question = dspy.InputField(desc="The user's question")
    query_type = dspy.OutputField(desc="ONLY output one word: rag, sql, or hybrid")


class Router(dspy.Module):
    """Router module to classify question type."""
    
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(RouterSignature)  # Simple Predict, not ChainOfThought
    
    def forward(self, question: str) -> str:
        q_lower = question.lower()
        
        # Try LLM classification first
        try:
            result = self.classifier(question=question)
            query_type = result.query_type.lower().strip()
            
            # Clean up common outputs
            query_type = query_type.replace('"', '').replace("'", '').strip()
            
            if 'rag' in query_type:
                return 'rag'
            elif 'sql' in query_type:
                return 'sql'
            elif 'hybrid' in query_type:
                return 'hybrid'
        except Exception:
            pass  # Fall through to heuristics
        
        # Fallback heuristics based on keywords (more robust)
        # RAG indicators: policy questions, definitions
        rag_keywords = ['policy', 'return window', 'definition', 'what is the', 'according to']
        if any(kw in q_lower for kw in rag_keywords) and 'revenue' not in q_lower:
            return 'rag'
        
        # Hybrid indicators: campaign names + data queries
        hybrid_keywords = ['summer beverages', 'winter classics', 'campaign', 'marketing calendar', 
                          'kpi definition', 'during', 'aov definition', 'gross margin']
        if any(kw in q_lower for kw in hybrid_keywords):
            return 'hybrid'
        
        # Default to SQL for data queries
        return 'sql'


class NLToSQLSignature(dspy.Signature):
    """Generate SQLite SQL query from natural language question.
    
    CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:
    1. TABLE ALIASES: 
       - `c` is for `customers` table ONLY.
       - `cat` is for `Categories` table ONLY.
       - `p` for `products`, `o` for `orders`, `od` for `order_items`.
    2. DATE LOGIC:
       - NEVER use `strftime` in the WHERE clause.
       - CORRECT: `o.OrderDate BETWEEN '1997-01-01' AND '1997-12-31'`
       - WRONG: `strftime(...) BETWEEN ...`
    3. SCHEMA:
       - `Categories` table has `CategoryName`, NOT `CompanyName`.
       - `Customers` table has `CompanyName`.
       - If asked for "Customer Margin", you MUST `JOIN customers c ON o.CustomerID = c.CustomerID`.
    4. NO HALLUCINATIONS:
       - Do not use `BETWEDIR` or `strftDirty`. Use standard `BETWEEN`.
    5. STRING MATCHING: 
       - Categories are Capitalized (e.g., 'Beverages', not 'beverages').
    """
    question = dspy.InputField(desc="The user's question")
    schema_info = dspy.InputField(desc="Database schema information")
    context = dspy.InputField(desc="Additional context from documents (dates, KPIs, etc.)")
    sql_query = dspy.OutputField(desc="ONLY a valid SQLite SQL query.")

class NLToSQL(dspy.Module):
    """Natural language to SQL converter."""
    
    def __init__(self):
        super().__init__()
        # Use Predictor instead of ChainOfThought for more reliable output
        # ChainOfThought can be problematic with smaller models
        self.sql_generator = dspy.Predict(NLToSQLSignature)
    
    def __call__(self, question: str, schema_info: str, context: str = "") -> str:
        """Generate SQL query. Supports both call() and forward() syntax."""
        return self.forward(question, schema_info, context)
    
    def forward(self, question: str, schema_info: str, context: str = "") -> str:
        """Generate SQL query."""
        try:
            result = self.sql_generator(
                question=question,
                schema_info=schema_info,
                context=context
            )
            
            # Check if result has sql_query attribute
            if not hasattr(result, 'sql_query'):
                # Try to extract SQL from result string if it's malformed
                result_str = str(result)
                # Look for SQL-like patterns
                import re
                sql_match = re.search(r'(SELECT|SELECT\s+.*?FROM)', result_str, re.IGNORECASE | re.DOTALL)
                if sql_match:
                    sql = sql_match.group(0).strip()
                else:
                    raise ValueError(f"Model returned invalid response: {result_str[:200]}")
            else:
                sql = result.sql_query.strip()
            
            # Clean up common issues
            if sql.startswith("```sql"):
                sql = sql.replace("```sql", "").strip()
            if sql.startswith("```"):
                sql = sql.replace("```", "").strip()
            if sql.endswith("```"):
                sql = sql[:-3].strip()
            
            # Fix common SQL typos that the model makes
            sql = self._fix_sql_typos(sql)
            
            # Validate SQL is not empty or garbage
            if not sql or len(sql) < 5:
                raise ValueError("Generated SQL is too short or empty")
            
            return sql
        except Exception as e:
            # Re-raise exception instead of silently returning empty string
            # This allows evaluation to properly count failures
            raise RuntimeError(f"SQL generation failed: {e}") from e
    
    def _fix_sql_typos(self, sql: str) -> str:
            """Fix common SQL typos, aliases, logic errors, and date shifts."""
            import re

            # 1. THE TIME TRAVEL FIX
            sql = sql.replace("'1996", "'2016")
            sql = sql.replace("'1997", "'2017")
            sql = sql.replace("'1998", "'2018")

            # 2. Fix Hallucinations & Syntax
            sql = sql.replace("BETWEDIR", "BETWEEN")
            sql = sql.replace("BETWEWEN", "BETWEEN")
            sql = sql.replace("BETWEWITH", "BETWEEN")
            sql = sql.replace("BETWEWS WITH", "BETWEEN")
            sql = sql.replace("BETWE", "BETWEEN")
            sql = sql.replace("BETWEENEN", "BETWEEN")
            sql = sql.replace("strftForms", "strftime")
            
            # 3. THE COST ESTIMATOR (New!)
            # The model loves to invent 'CostOfGoods' columns.
            # We replace them with the assignment's approximation rule: Cost = 0.7 * UnitPrice
            # Pattern: Any variations of CostOfGoods, Cost, etc.
            sql = re.sub(r"\b\w*\.?CostOfGoods\w*\b", "(od.UnitPrice * 0.7)", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\b\w*\.?StandardCost\w*\b", "(od.UnitPrice * 0.7)", sql, flags=re.IGNORECASE)
            
            # 4. Fix Case Sensitivity
            categories = ["Beverages", "Condiments", "Confections", "Dairy Products", 
                        "Grains/Cereals", "Meat/Poultry", "Produce", "Seafood"]
            for cat in categories:
                sql = re.sub(f"'{cat.lower()}'", f"'{cat}'", sql, flags=re.IGNORECASE)

            # 5. FORCE Standard Date Logic
            date_pattern = r"strftime\s*\(.*?(?:o\.|)OrderDate.*?\)\s*BETWEEN"
            if re.search(date_pattern, sql, re.IGNORECASE):
                sql = re.sub(date_pattern, "o.OrderDate BETWEEN", sql, flags=re.IGNORECASE)

            # 6. Fix Table Aliases & Missing Joins
            
            # INJECTOR 1: Categories (The fix for your Q2 regression)
            # If query uses CategoryName but forgets to join Categories table...
            if "CategoryName" in sql and "Categories" not in sql:
                # Inject the join after products
                if "JOIN products p" in sql:
                    sql = sql.replace("JOIN products p ON od.ProductID = p.ProductID", 
                                    "JOIN products p ON od.ProductID = p.ProductID JOIN Categories cat ON p.CategoryID = cat.CategoryID")
                # If it tried to use 'c' for category, switch it to 'cat'
                sql = sql.replace("c.CategoryName", "cat.CategoryName")

            # INJECTOR 2: Customers
            if "CompanyName" in sql and "customers" not in sql.lower():
                if "FROM orders o" in sql:
                    sql = sql.replace("FROM orders o", "FROM orders o JOIN customers c ON o.CustomerID = c.CustomerID")

            # 7. Alias Cleanup
            if "Categories" in sql:
                # Force "JOIN Categories cat"
                sql = re.sub(r"JOIN\s+Categories\s+c\b", "JOIN Categories cat", sql, flags=re.IGNORECASE)
                sql = re.sub(r"\bc\.CategoryName\b", "cat.CategoryName", sql, flags=re.IGNORECASE)
                sql = re.sub(r"\bc\.CategoryID\b", "cat.CategoryID", sql, flags=re.IGNORECASE)

            # 8. General Cleanup
            sql = sql.replace("`", "") 
            if "FROM orders o" in sql:
                sql = sql.replace("orders.", "o.")
            if "JOIN order_items od" in sql:
                sql = sql.replace("order_items.", "od.")
            if "FROM orders JOIN" in sql and "FROM orders o JOIN" not in sql:
                sql = sql.replace("FROM orders JOIN", "FROM orders o JOIN")

            return sql.strip()


class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from SQL results and context.
    
    FORMAT RULES:
    - int: Return integer (e.g., 14)
    - float: Return decimal (e.g., 123.45)
    - list[{...}]: Return JSON array
    - {...}: Return JSON object
    """
    question = dspy.InputField(desc="Original question")
    sql_results = dspy.InputField(desc="SQL results (columns and rows)")
    format_hint = dspy.InputField(desc="Expected format (int, float, dict, list)")
    context = dspy.InputField(desc="Document context")
    final_answer = dspy.OutputField(desc="Answer matching format_hint")
    explanation = dspy.OutputField(desc="Brief explanation (1-2 sentences)")
    citations = dspy.OutputField(desc="Comma-separated citations")


class Synthesizer(dspy.Module):
    """Synthesize typed answer from results."""
    
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.Predict(SynthesizerSignature)
    
    def forward(
        self,
        question: str,
        sql_results: dict,
        format_hint: str,
        context: str = "",
        db_tables_used: list = None,
        doc_chunks_used: list = None
    ) -> dict:
        sql_str = ""
        if sql_results.get("success") and sql_results.get("rows"):
            sql_str = f"Columns: {sql_results['columns']}\n"
            sql_str += f"Rows ({sql_results['row_count']}):\n"
            for row in sql_results['rows'][:10]:
                sql_str += f"  {row}\n"
        elif sql_results.get("error"):
            sql_str = f"Error: {sql_results['error']}"
        else:
            sql_str = "No results"
        
        context_str = context if context else "No additional context"
        
        result = None
        final_answer_str = None
        explanation = None
        
        try:
            result = self.synthesizer(
                question=question,
                sql_results=sql_str,
                format_hint=format_hint,
                context=context_str
            )
        except Exception as e:
            error_msg = str(e)
            if "JSONAdapter" in error_msg or "parse" in error_msg.lower():
                final_answer_match = re.search(r'"final_answer":\s*([^,}]+)', error_msg)
                explanation_match = re.search(r'"explanation[^"]*":\s*"([^"]+)"', error_msg)
                
                final_answer_str = final_answer_match.group(1).strip('"') if final_answer_match else None
                explanation = explanation_match.group(1) if explanation_match else "Answer from SQL results."
            else:
                raise
        
        # Build citations
        citations = []
        if db_tables_used:
            citations.extend(db_tables_used)
        if doc_chunks_used:
            citations.extend([chunk.chunk_id for chunk in doc_chunks_used])
        
        # Extract from result if available
        if result is not None:
            try:
                if hasattr(result, 'citations') and result.citations:
                    parsed = [c.strip() for c in result.citations.strip().split(",")]
                    citations.extend(parsed)
            except:
                pass
            
            try:
                if hasattr(result, 'final_answer'):
                    final_answer_str = str(result.final_answer).strip()
                    if final_answer_str.startswith("```"):
                        lines = final_answer_str.split("\n")
                        final_answer_str = "\n".join([l for l in lines if not l.strip().startswith("```")])
                        final_answer_str = final_answer_str.strip()
            except:
                pass
            
            for attr_name in ['explanation', 'explan', 'explanation_text']:
                if hasattr(result, attr_name):
                    explanation = getattr(result, attr_name)
                    break
        
        citations = list(set(citations))
        
        if not explanation or (isinstance(explanation, str) and not explanation.strip()):
            explanation = "Answer from SQL results and documents."
        
        return {
            "final_answer": final_answer_str,
            "explanation": explanation,
            "citations": citations
        }
