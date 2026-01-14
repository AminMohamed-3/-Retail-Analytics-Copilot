"""Optimize NL-to-SQL using BootstrapFewShot with Teacher-Student Setup.

Teacher Model: Qwen3 4B - Used for generating better few-shot examples
Student Model: Phi-3.5 - The model being optimized and used at inference

BootstrapFewShot is more reliable than MIPROv2 for small models.
"""
import dspy
import json
from datetime import datetime
from pathlib import Path

from config import setup_dspy, get_teacher_lm, clear_cache, STUDENT_MODEL, TEACHER_MODEL
from agent.dspy_signatures import NLToSQL
from agent.tools.sqlite_tool import SQLiteTool
from data.training_set import get_training_examples, get_validation_examples


def sql_execution_metric(example, pred, trace=None):
    """Metric: SQL executes successfully and returns rows."""
    sql_tool = SQLiteTool()
    
    try:
        sql = pred.sql_query if hasattr(pred, 'sql_query') else str(pred)
        
        # Clean up SQL
        if sql.startswith("```"):
            sql = sql.replace("```sql", "").replace("```", "").strip()
        
        result = sql_tool.execute_query(sql)
        
        if not result["success"]:
            return 0.0
        
        # Base score for successful execution
        score = 0.5
        
        # Bonus for returning rows
        if result["row_count"] > 0:
            score += 0.3
        
        # Bonus for matching expected structure
        expected_sql = example.sql_query.upper()
        actual_sql = sql.upper()
        
        # Check key patterns
        if "GROUP BY" in expected_sql and "GROUP BY" in actual_sql:
            score += 0.1
        if "ORDER BY" in expected_sql and "ORDER BY" in actual_sql:
            score += 0.1
        
        return min(score, 1.0)
    
    except Exception as e:
        print(f"  Metric error: {e}")
        return 0.0


def evaluate_model(model, examples, name="Model"):
    """Evaluate model on examples."""
    sql_tool = SQLiteTool()
    schema_info = sql_tool.get_full_schema_summary()
    
    success = 0
    execution_success = 0
    
    print(f"\nEvaluating {name}...")
    for i, ex in enumerate(examples):
        try:
            sql = model(
                question=ex.question,
                schema_info=schema_info,
                context=ex.context
            )
            
            result = sql_tool.execute_query(sql)
            
            if result["success"]:
                execution_success += 1
                if result["row_count"] > 0:
                    success += 1
                    print(f"  [{i+1}] ✓ {ex.question[:50]}...")
                else:
                    print(f"  [{i+1}] ~ {ex.question[:50]}... (0 rows)")
            else:
                print(f"  [{i+1}] ✗ {ex.question[:50]}... ({result['error'][:30]})")
        except Exception as e:
            print(f"  [{i+1}] ✗ {ex.question[:50]}... (Exception: {str(e)[:30]})")
    
    total = len(examples)
    print(f"\n{name} Results:")
    print(f"  Execution Success: {execution_success}/{total} ({100*execution_success/total:.1f}%)")
    print(f"  Full Success: {success}/{total} ({100*success/total:.1f}%)")
    
    return {
        "execution_rate": execution_success / total,
        "success_rate": success / total,
        "execution_count": execution_success,
        "success_count": success,
        "total": total
    }


def optimize_nl_to_sql():
    """Run Teacher-Student BootstrapFewShot optimization on NL-to-SQL module.
    
    Uses Qwen3 4B as teacher to generate better demonstrations,
    while optimizing Phi-3.5 as the student model.
    """
    print("=" * 70)
    print("NL-to-SQL Teacher-Student Optimization (BootstrapFewShot)")
    print("=" * 70)
    print(f"Teacher Model: {TEACHER_MODEL} (for generating examples)")
    print(f"Student Model: {STUDENT_MODEL} (for inference)")
    print()
    
    # Clear cache for fresh results
    print("Clearing cache...")
    clear_cache()
    
    # Setup student model for inference
    student_lm = setup_dspy(use_cache=False)
    
    # Get teacher model for optimization
    teacher_lm = get_teacher_lm()
    
    sql_tool = SQLiteTool()
    schema_info = sql_tool.get_full_schema_summary()
    
    print("\nLoading training examples...")
    train_examples = get_training_examples(schema_info)
    val_examples = get_validation_examples(schema_info)
    print(f"  Training: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples")
    
    # ============================================
    # BASELINE WITHOUT HARDCODED FIXES
    # ============================================
    print("\n" + "=" * 70)
    print("PHASE 1: Baseline WITHOUT Hardcoded Fixes")
    print("=" * 70)
    print("This measures raw model capability before any post-processing.")
    
    baseline_no_fixes = NLToSQL(apply_fixes=False)
    baseline_no_fixes_results = evaluate_model(baseline_no_fixes, val_examples, "Baseline (no fixes)")
    
    # ============================================
    # BASELINE WITH HARDCODED FIXES
    # ============================================
    print("\n" + "=" * 70)
    print("PHASE 2: Baseline WITH Hardcoded Fixes")
    print("=" * 70)
    print("This measures improvement from post-processing fixes alone.")
    
    baseline_with_fixes = NLToSQL(apply_fixes=True)
    baseline_with_fixes_results = evaluate_model(baseline_with_fixes, val_examples, "Baseline (with fixes)")
    
    # ============================================
    # TEACHER-STUDENT OPTIMIZATION
    # ============================================
    print("\n" + "=" * 70)
    print("PHASE 3: Teacher-Student BootstrapFewShot Optimization")
    print("=" * 70)
    print(f"Using {TEACHER_MODEL} to bootstrap better examples for {STUDENT_MODEL}")
    print("This may take a few minutes...")
    
    optimized_results = None
    optimized_model = None
    
    try:
        # Use teacher model to bootstrap examples
        with dspy.context(lm=teacher_lm):
            optimizer = dspy.BootstrapFewShot(
                metric=sql_execution_metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=6,
            )
        
        # Compile with student model (Phi-3.5)
        dspy.settings.configure(lm=student_lm)
        
        optimized_model = optimizer.compile(
            student=NLToSQL(apply_fixes=False),  # No fixes - measure optimization alone
            trainset=train_examples[:20]
        )
        
        print("\n✅ Optimization completed!")
        
        # Evaluate optimized model
        optimized_results = evaluate_model(optimized_model, val_examples, "Optimized (no fixes)")
        
        # Save optimized model
        save_path = "optimized_nl_to_sql.json"
        optimized_model.save(save_path)
        print(f"\n✓ Saved optimized model to: {save_path}")
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nUsing baseline model as fallback...")
        baseline_no_fixes.save("optimized_nl_to_sql.json")
        optimized_results = baseline_no_fixes_results
    
    # ============================================
    # FINAL COMPARISON
    # ============================================
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "optimizer": "BootstrapFewShot",
        "teacher_model": TEACHER_MODEL,
        "student_model": STUDENT_MODEL,
        "baseline_no_fixes": baseline_no_fixes_results,
        "baseline_with_fixes": baseline_with_fixes_results,
        "optimized_no_fixes": optimized_results,
        "improvements": {
            "hardcoding_value": baseline_with_fixes_results["execution_rate"] - baseline_no_fixes_results["execution_rate"],
            "optimization_value": optimized_results["execution_rate"] - baseline_no_fixes_results["execution_rate"] if optimized_results else 0,
        }
    }
    
    print(f"\n{'Configuration':<30} {'Exec Rate':<15} {'Success Rate':<15}")
    print("-" * 60)
    print(f"{'Baseline (no fixes)':<30} {100*baseline_no_fixes_results['execution_rate']:.1f}%{'':<10} {100*baseline_no_fixes_results['success_rate']:.1f}%")
    print(f"{'Baseline (with fixes)':<30} {100*baseline_with_fixes_results['execution_rate']:.1f}%{'':<10} {100*baseline_with_fixes_results['success_rate']:.1f}%")
    if optimized_results:
        print(f"{'Optimized (no fixes)':<30} {100*optimized_results['execution_rate']:.1f}%{'':<10} {100*optimized_results['success_rate']:.1f}%")
    
    print("\nImprovements:")
    print(f"  Hardcoding alone: {100*results['improvements']['hardcoding_value']:+.1f}%")
    print(f"  Optimization alone: {100*results['improvements']['optimization_value']:+.1f}%")
    
    # Save results
    results_path = Path("metrics") / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {results_path}")
    
    return optimized_model, results


if __name__ == "__main__":
    optimize_nl_to_sql()
