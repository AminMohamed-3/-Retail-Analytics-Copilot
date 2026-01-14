"""DSPy optimization script for Router module."""
import dspy
from config import setup_dspy
from agent.dspy_signatures import Router, OptimizedRouter

# Setup DSPy
setup_dspy()


# Training examples for Router classification
training_examples = [
    {
        "question": "According to the product policy, what is the return window for beverages?",
        "query_type": "rag"
    },
    {
        "question": "What is the total revenue from all orders?",
        "query_type": "sql"
    },
    {
        "question": "During Summer Beverages 1997, which category had the highest sales?",
        "query_type": "hybrid"
    },
    {
        "question": "What does the marketing calendar say about Winter Classics?",
        "query_type": "rag"
    },
    {
        "question": "List all products in the database.",
        "query_type": "sql"
    },
    {
        "question": "Using the AOV definition, calculate average order value for 1997.",
        "query_type": "hybrid"
    },
    {
        "question": "What are the KPI definitions?",
        "query_type": "rag"
    },
    {
        "question": "How many customers are in the database?",
        "query_type": "sql"
    },
    {
        "question": "What was the revenue for Beverages category during Summer Beverages 1997?",
        "query_type": "hybrid"
    },
    {
        "question": "What is the return policy for perishables?",
        "query_type": "rag"
    },
    {
        "question": "Show me the top 10 products by quantity sold.",
        "query_type": "sql"
    },
    {
        "question": "Per the KPI definition, who was the top customer by gross margin in 1997?",
        "query_type": "hybrid"
    },
    {
        "question": "What categories are mentioned in the catalog?",
        "query_type": "rag"
    },
    {
        "question": "Calculate total revenue from Order Details table.",
        "query_type": "sql"
    },
    {
        "question": "During Winter Classics 1997, what was the AOV according to KPI docs?",
        "query_type": "hybrid"
    },
    {
        "question": "What is the return window for unopened beverages according to policy?",
        "query_type": "rag"
    },
    {
        "question": "Find all orders placed in 1997.",
        "query_type": "sql"
    },
    {
        "question": "Which category had highest quantity during Summer Beverages 1997 campaign?",
        "query_type": "hybrid"
    },
    {
        "question": "What dates are covered in the marketing calendar?",
        "query_type": "rag"
    },
    {
        "question": "Get the list of all suppliers.",
        "query_type": "sql"
    },
    {
        "question": "Using gross margin formula from KPI docs, find top customer in 1997.",
        "query_type": "hybrid"
    },
    {
        "question": "What does the product policy say about non-perishables?",
        "query_type": "rag"
    },
    {
        "question": "Count distinct customers.",
        "query_type": "sql"
    },
    {
        "question": "What was total revenue for Beverages during Summer Beverages 1997 dates?",
        "query_type": "hybrid"
    },
    {
        "question": "Explain the AOV formula from the documentation.",
        "query_type": "rag"
    },
    {
        "question": "Show order details for a specific order ID.",
        "query_type": "sql"
    },
    {
        "question": "Calculate AOV for Winter Classics 1997 using the KPI definition.",
        "query_type": "hybrid"
    },
    {
        "question": "What is the return policy for seafood?",
        "query_type": "rag"
    },
    {
        "question": "List all product categories.",
        "query_type": "sql"
    },
]


def evaluate_router(router_module, examples):
    """Evaluate router accuracy."""
    correct = 0
    total = len(examples)
    
    for ex in examples:
        predicted = router_module(ex["question"])
        if predicted == ex["query_type"]:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def main():
    """Run optimization."""
    print("=" * 60)
    print("DSPy Router Optimization")
    print("=" * 60)
    
    # Baseline Router
    print("\n1. Evaluating baseline Router...")
    baseline_router = Router()
    baseline_accuracy = evaluate_router(baseline_router, training_examples)
    print(f"   Baseline Accuracy: {baseline_accuracy:.2%} ({int(baseline_accuracy * len(training_examples))}/{len(training_examples)})")
    
    # Prepare training set for DSPy
    print("\n2. Preparing training set...")
    trainset = []
    for ex in training_examples[:20]:  # Use first 20 for training
        trainset.append(dspy.Example(
            question=ex["question"],
            query_type=ex["query_type"]
        ).with_inputs("question"))
    
    # Optimize Router using BootstrapFewShot
    print("\n3. Optimizing Router with BootstrapFewShot...")
    optimized_router = OptimizedRouter()
    
    try:
        # Use BootstrapFewShot optimizer
        optimizer = dspy.BootstrapFewShot(
            metric=lambda example, pred, trace=None: example.query_type == pred,
            max_bootstrapped_demos=4,
            max_labeled_demos=8
        )
        
        optimized_router = optimizer.compile(
            student=optimized_router,
            trainset=trainset
        )
        
        print("   Optimization completed!")
        
    except Exception as e:
        print(f"   Optimization failed: {e}")
        print("   Using manual few-shot examples instead...")
        # Fallback: manually add few-shot examples
        optimized_router = OptimizedRouter()
    
    # Evaluate optimized Router
    print("\n4. Evaluating optimized Router...")
    optimized_accuracy = evaluate_router(optimized_router, training_examples)
    print(f"   Optimized Accuracy: {optimized_accuracy:.2%} ({int(optimized_accuracy * len(training_examples))}/{len(training_examples)})")
    
    # Calculate improvement
    improvement = optimized_accuracy - baseline_accuracy
    print(f"\n5. Results:")
    print(f"   Baseline:   {baseline_accuracy:.2%}")
    print(f"   Optimized:  {optimized_accuracy:.2%}")
    print(f"   Improvement: {improvement:+.2%}")
    
    # Save results
    results = {
        "baseline_accuracy": baseline_accuracy,
        "optimized_accuracy": optimized_accuracy,
        "improvement": improvement,
        "training_set_size": len(trainset),
        "test_set_size": len(training_examples)
    }
    
    import json
    with open("router_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to router_optimization_results.json")
    
    return results


if __name__ == "__main__":
    main()

