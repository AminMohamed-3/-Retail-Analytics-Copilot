"""
Benchmark script for comparing agent performance across configurations.

Usage:
    python benchmark.py                    # Run all configurations
    python benchmark.py --config baseline  # Run only baseline
    python benchmark.py --visualize        # Generate charts from saved results
"""
import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import setup_dspy, clear_cache
from agent.graph_hybrid import HybridAgent
from agent.dspy_signatures import NLToSQL
from agent.tools.sqlite_tool import SQLiteTool


def clear_dspy_cache():
    """Clear DSPy cache to ensure fresh LLM calls."""
    clear_cache()  # Use centralized cache clearing


class BenchmarkRunner:
    """Run benchmarks across different configurations."""
    
    def __init__(self, questions_file: str = "benchmark_questions.jsonl", 
                 results_dir: str = "metrics"):
        self.questions_file = Path(questions_file)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.questions = self._load_questions()
        self.sql_tool = SQLiteTool()
    
    def _load_questions(self) -> List[dict]:
        questions = []
        if self.questions_file.exists():
            with open(self.questions_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        questions.append(json.loads(line))
        return questions
    
    def run_baseline(self) -> dict:
        """Run baseline: no optimization, no fixes."""
        print("\n" + "=" * 60)
        print("CONFIGURATION 1: BASELINE (No optimization, no fixes)")
        print("=" * 60)
        
        setup_dspy(use_cache=False)
        nl_to_sql = NLToSQL(apply_fixes=False)
        
        results = {
            "config": "baseline",
            "timestamp": datetime.now().isoformat(),
            "questions": [],
            "metrics": {
                "total": len(self.questions),
                "sql_success": 0,
                "execution_success": 0,
                "total_time": 0.0,
            }
        }
        
        schema_info = self.sql_tool.get_full_schema_summary()
        
        for q in self.questions:
            start = time.time()
            q_result = {
                "id": q["id"],
                "category": q.get("category", "unknown"),
                "sql_generated": False,
                "sql_executed": False,
                "error": None,
                "time": 0.0,
            }
            
            try:
                sql = nl_to_sql.forward(
                    question=q["question"],
                    schema_info=schema_info,
                    context=""
                )
                q_result["sql_generated"] = True
                q_result["sql"] = sql
                results["metrics"]["sql_success"] += 1
                
                exec_result = self.sql_tool.execute_query(sql)
                if exec_result["success"]:
                    q_result["sql_executed"] = True
                    results["metrics"]["execution_success"] += 1
                else:
                    q_result["error"] = exec_result["error"]
            except Exception as e:
                q_result["error"] = str(e)[:100]
            
            q_result["time"] = time.time() - start
            results["questions"].append(q_result)
            
            status = "✓" if q_result["sql_executed"] else "✗"
            print(f"  {status} {q['id']}: {(q_result.get('error') or 'OK')[:50]}")
        
        results["metrics"]["total_time"] = sum(q["time"] for q in results["questions"])
        self._print_summary(results["metrics"])
        
        return results
    
    def run_hardcoded(self) -> dict:
        """Run hardcoded: typo fixes enabled, no DSPy optimization."""
        print("\n" + "=" * 60)
        print("CONFIGURATION 2: HARDCODED (Typo fixes enabled)")
        print("=" * 60)
        
        setup_dspy(use_cache=False)
        nl_to_sql = NLToSQL(apply_fixes=True)
        
        results = {
            "config": "hardcoded",
            "timestamp": datetime.now().isoformat(),
            "questions": [],
            "metrics": {
                "total": len(self.questions),
                "sql_success": 0,
                "execution_success": 0,
                "total_time": 0.0,
            }
        }
        
        schema_info = self.sql_tool.get_full_schema_summary()
        
        for q in self.questions:
            start = time.time()
            q_result = {
                "id": q["id"],
                "category": q.get("category", "unknown"),
                "sql_generated": False,
                "sql_executed": False,
                "error": None,
                "time": 0.0,
            }
            
            try:
                sql = nl_to_sql.forward(
                    question=q["question"],
                    schema_info=schema_info,
                    context=""
                )
                q_result["sql_generated"] = True
                q_result["sql"] = sql
                results["metrics"]["sql_success"] += 1
                
                exec_result = self.sql_tool.execute_query(sql)
                if exec_result["success"]:
                    q_result["sql_executed"] = True
                    results["metrics"]["execution_success"] += 1
                else:
                    q_result["error"] = exec_result["error"]
            except Exception as e:
                q_result["error"] = str(e)[:100]
            
            q_result["time"] = time.time() - start
            results["questions"].append(q_result)
            
            status = "✓" if q_result["sql_executed"] else "✗"
            print(f"  {status} {q['id']}: {(q_result.get('error') or 'OK')[:50]}")
        
        results["metrics"]["total_time"] = sum(q["time"] for q in results["questions"])
        self._print_summary(results["metrics"])
        
        return results
    
    def run_optimized(self) -> dict:
        """Run optimized: DSPy optimization + selective fixes."""
        print("\n" + "=" * 60)
        print("CONFIGURATION 3: OPTIMIZED (DSPy + selective fixes)")
        print("=" * 60)
        
        if not os.path.exists("optimized_nl_to_sql.json"):
            print("⚠️  Optimized model not found. Run optimize_nl_to_sql.py first.")
            print("   Falling back to hardcoded configuration.")
            return self.run_hardcoded()
        
        setup_dspy(use_cache=False)
        
        nl_to_sql = NLToSQL(apply_fixes=True)
        try:
            nl_to_sql.load("optimized_nl_to_sql.json")
            print("✓ Loaded optimized NLToSQL model")
        except Exception as e:
            print(f"⚠️  Failed to load optimized model: {e}")
            print("   Falling back to hardcoded configuration.")
            return self.run_hardcoded()
        
        results = {
            "config": "optimized",
            "timestamp": datetime.now().isoformat(),
            "questions": [],
            "metrics": {
                "total": len(self.questions),
                "sql_success": 0,
                "execution_success": 0,
                "total_time": 0.0,
            }
        }
        
        schema_info = self.sql_tool.get_full_schema_summary()
        
        for q in self.questions:
            start = time.time()
            q_result = {
                "id": q["id"],
                "category": q.get("category", "unknown"),
                "sql_generated": False,
                "sql_executed": False,
                "error": None,
                "time": 0.0,
            }
            
            try:
                sql = nl_to_sql.forward(
                    question=q["question"],
                    schema_info=schema_info,
                    context=""
                )
                q_result["sql_generated"] = True
                q_result["sql"] = sql
                results["metrics"]["sql_success"] += 1
                
                exec_result = self.sql_tool.execute_query(sql)
                if exec_result["success"]:
                    q_result["sql_executed"] = True
                    results["metrics"]["execution_success"] += 1
                else:
                    q_result["error"] = exec_result["error"]
            except Exception as e:
                q_result["error"] = str(e)[:100]
            
            q_result["time"] = time.time() - start
            results["questions"].append(q_result)
            
            status = "✓" if q_result["sql_executed"] else "✗"
            print(f"  {status} {q['id']}: {(q_result.get('error') or 'OK')[:50]}")
        
        results["metrics"]["total_time"] = sum(q["time"] for q in results["questions"])
        self._print_summary(results["metrics"])
        
        return results
    
    def run_optimized_no_fixes(self) -> dict:
        """Run optimized: DSPy optimization WITHOUT hardcoded fixes.
        
        This isolates the value of DSPy optimization alone,
        without the post-processing fixes.
        """
        print("\n" + "=" * 60)
        print("CONFIGURATION 4: OPTIMIZED NO FIXES (DSPy only, no hardcoding)")
        print("=" * 60)
        
        if not os.path.exists("optimized_nl_to_sql.json"):
            print("⚠️  Optimized model not found. Run optimize_nl_to_sql.py first.")
            print("   Falling back to baseline configuration.")
            return self.run_baseline()
        
        setup_dspy(use_cache=False)
        
        # Load optimized model but disable fixes
        nl_to_sql = NLToSQL(apply_fixes=False)
        try:
            nl_to_sql.load("optimized_nl_to_sql.json")
            print("✓ Loaded optimized NLToSQL model (fixes disabled)")
        except Exception as e:
            print(f"⚠️  Failed to load optimized model: {e}")
            print("   Falling back to baseline configuration.")
            return self.run_baseline()
        
        results = {
            "config": "optimized_no_fixes",
            "timestamp": datetime.now().isoformat(),
            "questions": [],
            "metrics": {
                "total": len(self.questions),
                "sql_success": 0,
                "execution_success": 0,
                "total_time": 0.0,
            }
        }
        
        schema_info = self.sql_tool.get_full_schema_summary()
        
        for q in self.questions:
            start = time.time()
            q_result = {
                "id": q["id"],
                "category": q.get("category", "unknown"),
                "sql_generated": False,
                "sql_executed": False,
                "error": None,
                "time": 0.0,
            }
            
            try:
                sql = nl_to_sql.forward(
                    question=q["question"],
                    schema_info=schema_info,
                    context=""
                )
                q_result["sql_generated"] = True
                q_result["sql"] = sql
                results["metrics"]["sql_success"] += 1
                
                exec_result = self.sql_tool.execute_query(sql)
                if exec_result["success"]:
                    q_result["sql_executed"] = True
                    results["metrics"]["execution_success"] += 1
                else:
                    q_result["error"] = exec_result["error"]
            except Exception as e:
                q_result["error"] = str(e)[:100]
            
            q_result["time"] = time.time() - start
            results["questions"].append(q_result)
            
            status = "✓" if q_result["sql_executed"] else "✗"
            print(f"  {status} {q['id']}: {(q_result.get('error') or 'OK')[:50]}")
        
        results["metrics"]["total_time"] = sum(q["time"] for q in results["questions"])
        self._print_summary(results["metrics"])
        
        return results
    
    def run_full_agent(self, config: str = "hardcoded") -> dict:
        """Run full agent with repair loop."""
        print("\n" + "=" * 60)
        print(f"FULL AGENT TEST ({config} configuration)")
        print("=" * 60)
        
        setup_dspy(use_cache=False)
        use_optimized = (config == "optimized")
        agent = HybridAgent(use_optimized=use_optimized)
        
        results = {
            "config": f"full_agent_{config}",
            "timestamp": datetime.now().isoformat(),
            "questions": [],
            "metrics": {
                "total": len(self.questions),
                "answer_success": 0,
                "sql_success": 0,
                "repair_count": 0,
                "avg_confidence": 0.0,
                "total_time": 0.0,
            }
        }
        
        total_confidence = 0.0
        
        for q in self.questions:
            start = time.time()
            
            try:
                result = agent.process_question(
                    question=q["question"],
                    question_id=q["id"],
                    format_hint=q["format_hint"]
                )
                
                q_result = {
                    "id": q["id"],
                    "category": q.get("category", "unknown"),
                    "success": result["final_answer"] is not None,
                    "confidence": result["confidence"],
                    "sql": result.get("sql", ""),
                    "time": time.time() - start,
                }
                
                if result["final_answer"] is not None:
                    results["metrics"]["answer_success"] += 1
                if result.get("sql"):
                    results["metrics"]["sql_success"] += 1
                
                total_confidence += result["confidence"]
                
            except Exception as e:
                q_result = {
                    "id": q["id"],
                    "category": q.get("category", "unknown"),
                    "success": False,
                    "confidence": 0.0,
                    "error": str(e),
                    "time": time.time() - start,
                }
            
            results["questions"].append(q_result)
            status = "✓" if q_result.get("success") else "✗"
            conf = q_result.get("confidence", 0)
            print(f"  {status} {q['id']}: conf={conf:.2f}")
        
        results["metrics"]["total_time"] = sum(q["time"] for q in results["questions"])
        if self.questions:
            results["metrics"]["avg_confidence"] = total_confidence / len(self.questions)
        self._print_summary(results["metrics"])
        
        return results
    
    def _print_summary(self, metrics: dict):
        print("\n--- Summary ---")
        print(f"  SQL Generation: {metrics.get('sql_success', 0)}/{metrics['total']}")
        print(f"  SQL Execution:  {metrics.get('execution_success', 0)}/{metrics['total']}")
        if 'answer_success' in metrics:
            print(f"  Answer Success: {metrics['answer_success']}/{metrics['total']}")
        if metrics.get('avg_confidence'):
            print(f"  Avg Confidence: {metrics['avg_confidence']:.2f}")
        print(f"  Total Time:     {metrics['total_time']:.1f}s")
    
    def save_results(self, results: dict, suffix: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{results['config']}_{timestamp}{suffix}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved: {filepath}")
        return filepath
    
    def run_all(self) -> Dict[str, dict]:
        """Run all configurations."""
        all_results = {}
        
        all_results["baseline"] = self.run_baseline()
        self.save_results(all_results["baseline"])
        
        all_results["hardcoded"] = self.run_hardcoded()
        self.save_results(all_results["hardcoded"])
        
        all_results["optimized_no_fixes"] = self.run_optimized_no_fixes()
        self.save_results(all_results["optimized_no_fixes"])
        
        all_results["optimized"] = self.run_optimized()
        self.save_results(all_results["optimized"])
        
        all_results["full_agent_hardcoded"] = self.run_full_agent("hardcoded")
        self.save_results(all_results["full_agent_hardcoded"])
        
        all_results["full_agent_optimized"] = self.run_full_agent("optimized")
        self.save_results(all_results["full_agent_optimized"])
        
        self._print_comparison(all_results)
        
        return all_results
    
    def _print_comparison(self, all_results: Dict[str, dict]):
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Configuration':<15} {'SQL Gen':<12} {'SQL Exec':<12} {'Time':<10}")
        print("-" * 60)
        
        for name, results in all_results.items():
            m = results["metrics"]
            sql_gen = f"{m.get('sql_success', 0)}/{m['total']}"
            sql_exec = f"{m.get('execution_success', 0)}/{m['total']}"
            time_str = f"{m['total_time']:.1f}s"
            print(f"{name:<15} {sql_gen:<12} {sql_exec:<12} {time_str:<10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark agent configurations")
    parser.add_argument(
        "--config",
        choices=["all", "baseline", "hardcoded", "optimized", "optimized_no_fixes", 
                 "full_hardcoded", "full_optimized"],
        default="all",
        help="Configuration to run"
    )
    parser.add_argument(
        "--questions",
        default="benchmark_questions.jsonl",
        help="Questions file"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization from saved results"
    )
    parser.add_argument(
        "--no-cache-clear",
        action="store_true",
        help="Skip cache clearing"
    )
    args = parser.parse_args()
    
    if args.visualize:
        print("Run visualize_results.py for charts")
        return
    
    if not args.no_cache_clear:
        print("\n=== Clearing DSPy Cache ===")
        clear_dspy_cache()
    
    runner = BenchmarkRunner(questions_file=args.questions)
    
    if args.config == "all":
        runner.run_all()
    elif args.config == "baseline":
        results = runner.run_baseline()
        runner.save_results(results)
    elif args.config == "hardcoded":
        results = runner.run_hardcoded()
        runner.save_results(results)
    elif args.config == "optimized":
        results = runner.run_optimized()
        runner.save_results(results)
    elif args.config == "optimized_no_fixes":
        results = runner.run_optimized_no_fixes()
        runner.save_results(results)
    elif args.config == "full_hardcoded":
        results = runner.run_full_agent("hardcoded")
        runner.save_results(results)
    elif args.config == "full_optimized":
        results = runner.run_full_agent("optimized")
        runner.save_results(results)


if __name__ == "__main__":
    main()
