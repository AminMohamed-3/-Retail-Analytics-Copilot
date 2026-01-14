"""Optimize Router using Teacher-Student BootstrapFewShot.

Teacher Model: Qwen3 4B - Used for generating better few-shot examples
Student Model: Phi-3.5 - The model being optimized and used at inference
"""
import dspy
from config import setup_dspy, get_teacher_lm, STUDENT_MODEL, TEACHER_MODEL
from agent.dspy_signatures import Router, RouterSignature
import json


def create_training_examples():
    """Create training examples for Router optimization."""
    examples = [
        # RAG examples (document-only questions)
        dspy.Example(question="According to the product policy, what is the return window for Beverages?", query_type="rag").with_inputs("question"),
        dspy.Example(question="What does the marketing calendar say about Summer Beverages?", query_type="rag").with_inputs("question"),
        dspy.Example(question="What is the KPI definition of AOV?", query_type="rag").with_inputs("question"),
        dspy.Example(question="What categories are mentioned in the catalog?", query_type="rag").with_inputs("question"),
        
        # SQL examples (database-only questions)
        dspy.Example(question="Top 3 products by total revenue all-time", query_type="sql").with_inputs("question"),
        dspy.Example(question="How many orders were placed in 1997?", query_type="sql").with_inputs("question"),
        dspy.Example(question="List all customers from USA", query_type="sql").with_inputs("question"),
        dspy.Example(question="What is the average order value?", query_type="sql").with_inputs("question"),
        
        # Hybrid examples (need both docs and DB)
        dspy.Example(question="During Summer Beverages 1997, which category had highest quantity?", query_type="hybrid").with_inputs("question"),
        dspy.Example(question="Using the AOV definition from KPI docs, what was AOV during Winter Classics 1997?", query_type="hybrid").with_inputs("question"),
        dspy.Example(question="Total revenue from Beverages category during Summer Beverages 1997 dates", query_type="hybrid").with_inputs("question"),
        dspy.Example(question="Per the KPI definition of gross margin, who was the top customer by gross margin in 1997?", query_type="hybrid").with_inputs("question"),
    ]
    return examples


def test_router(router, label="Router"):
    """Test router on example questions."""
    test_questions = [
        ("What is the return window for Beverages?", "rag"),
        ("Top 5 products by revenue", "sql"),
        ("During Summer Beverages 1997, which category had highest sales?", "hybrid"),
        ("How many customers are from Germany?", "sql"),
    ]
    
    correct = 0
    print(f"\nTesting {label}:")
    for q, expected in test_questions:
        try:
            result = router(q)
            is_correct = result.lower() == expected.lower()
            correct += int(is_correct)
            status = "✓" if is_correct else "✗"
            print(f"  {status} '{q[:40]}...' -> {result} (expected: {expected})")
        except Exception as e:
            print(f"  ✗ '{q[:40]}...' -> Error: {str(e)[:30]}")
    
    accuracy = correct / len(test_questions) * 100
    print(f"  Accuracy: {correct}/{len(test_questions)} ({accuracy:.0f}%)")
    return accuracy


def optimize_router():
    """Optimize Router using Teacher-Student BootstrapFewShot."""
    print("=" * 70)
    print("Router Teacher-Student Optimization")
    print("=" * 70)
    print(f"Teacher Model: {TEACHER_MODEL} (for optimization guidance)")
    print(f"Student Model: {STUDENT_MODEL} (for inference)")
    print()
    
    # Setup student model
    student_lm = setup_dspy(use_cache=False)
    
    # Get teacher model
    teacher_lm = get_teacher_lm()
    
    print("Creating training examples...")
    trainset = create_training_examples()
    print(f"  Training examples: {len(trainset)}")
    
    print("\nExample questions:")
    for ex in trainset[:3]:
        print(f"  - {ex.question[:50]}... -> {ex.query_type}")
    
    # ============================================
    # BASELINE ROUTER
    # ============================================
    print("\n" + "=" * 70)
    print("PHASE 1: Baseline Router")
    print("=" * 70)
    
    baseline_router = Router()
    baseline_accuracy = test_router(baseline_router, "Baseline Router")
    
    # ============================================
    # TEACHER-STUDENT OPTIMIZATION
    # ============================================
    print("\n" + "=" * 70)
    print("PHASE 2: Teacher-Student Optimization")
    print("=" * 70)
    print(f"Using {TEACHER_MODEL} to bootstrap better examples for {STUDENT_MODEL}")
    
    optimized_router = None
    optimized_accuracy = baseline_accuracy
    
    try:
        # Metric for router - pred is a string (rag/sql/hybrid)
        def router_metric(example, pred, trace=None):
            expected = example.query_type.lower()
            # pred might be a string or an object with query_type
            if isinstance(pred, str):
                actual = pred.lower()
            elif hasattr(pred, 'query_type'):
                actual = pred.query_type.lower()
            else:
                actual = str(pred).lower()
            return expected == actual
        
        # Use teacher model for bootstrapping examples
        with dspy.context(lm=teacher_lm):
            optimizer = dspy.BootstrapFewShot(
                metric=router_metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=8,
            )
        
        # Compile with student model
        dspy.settings.configure(lm=student_lm)
        
        optimized_router = optimizer.compile(
            student=Router(),
            trainset=trainset
        )
        
        print("\n✅ Optimization completed!")
        
        # Test optimized router
        optimized_accuracy = test_router(optimized_router, "Optimized Router")
        
        # Save the optimized router
        print("\nSaving optimized router...")
        optimized_router.save("optimized_router_bootstrap.json")
        print("✅ Saved to optimized_router_bootstrap.json")
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        print("\nThe baseline Router will be used.")
        import traceback
        traceback.print_exc()
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Baseline Accuracy:  {baseline_accuracy:.0f}%")
    print(f"Optimized Accuracy: {optimized_accuracy:.0f}%")
    print(f"Improvement:        {optimized_accuracy - baseline_accuracy:+.0f}%")
    
    return optimized_router


if __name__ == "__main__":
    optimize_router()
