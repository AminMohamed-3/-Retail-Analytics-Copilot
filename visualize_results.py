"""Visualize benchmark results with comparison charts."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_latest_results(metrics_dir: str = "metrics") -> Dict[str, dict]:
    """Load the latest benchmark results for each configuration."""
    metrics_path = Path(metrics_dir)
    if not metrics_path.exists():
        print(f"No metrics directory found at {metrics_dir}")
        return {}
    
    results = {}
    configs = ["baseline", "hardcoded", "optimized"]
    
    for config in configs:
        pattern = f"benchmark_{config}_*.json"
        files = sorted(metrics_path.glob(pattern), reverse=True)
        if files:
            with open(files[0], 'r') as f:
                results[config] = json.load(f)
                print(f"Loaded {config}: {files[0].name}")
    
    return results


def create_comparison_chart(results: Dict[str, dict], output_dir: str = "charts"):
    """Create comparison bar chart for all configurations."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not results:
        print("No results to visualize")
        return
    
    configs = list(results.keys())
    metrics_keys = ["sql_success", "execution_success"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart: Success rates
    x = np.arange(len(configs))
    width = 0.35
    
    sql_success = []
    exec_success = []
    
    for config in configs:
        m = results[config]["metrics"]
        total = m["total"]
        sql_success.append(m.get("sql_success", 0) / total * 100 if total > 0 else 0)
        exec_success.append(m.get("execution_success", 0) / total * 100 if total > 0 else 0)
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, sql_success, width, label='SQL Generation', color='#3498db')
    bars2 = ax1.bar(x + width/2, exec_success, width, label='SQL Execution', color='#2ecc71')
    
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('SQL Success Rates by Configuration')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in configs])
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Time comparison
    ax2 = axes[1]
    times = [results[c]["metrics"]["total_time"] for c in configs]
    bars = ax2.bar(configs, times, color=['#e74c3c', '#f39c12', '#27ae60'])
    
    ax2.set_ylabel('Total Time (seconds)')
    ax2.set_title('Processing Time by Configuration')
    ax2.set_xticklabels([c.capitalize() for c in configs])
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = output_path / f"benchmark_comparison_{timestamp}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Saved chart: {chart_path}")
    
    plt.close()
    
    return chart_path


def create_improvement_summary(results: Dict[str, dict], output_dir: str = "charts"):
    """Create improvement summary chart."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if "baseline" not in results or "hardcoded" not in results:
        print("Need both baseline and hardcoded results for improvement chart")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = ["baseline", "hardcoded"]
    if "optimized" in results:
        configs.append("optimized")
    
    metrics_labels = {
        "sql_success": "SQL Generation",
        "execution_success": "SQL Execution"
    }
    
    x = np.arange(len(configs))
    width = 0.25
    multiplier = 0
    
    colors = ['#3498db', '#2ecc71']
    
    for i, (metric, label) in enumerate(metrics_labels.items()):
        values = []
        for config in configs:
            m = results[config]["metrics"]
            total = m["total"]
            val = m.get(metric, 0) / total * 100 if total > 0 else 0
            values.append(val)
        
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=label, color=colors[i])
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        multiplier += 1
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Performance Improvement: Baseline → Hardcoded → Optimized')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([c.capitalize() for c in configs])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 110)
    
    # Add improvement arrows
    if len(configs) >= 2:
        baseline_exec = results["baseline"]["metrics"].get("execution_success", 0)
        hardcoded_exec = results["hardcoded"]["metrics"].get("execution_success", 0)
        total = results["baseline"]["metrics"]["total"]
        
        if total > 0:
            improvement = (hardcoded_exec - baseline_exec) / total * 100
            ax.annotate(f'+{improvement:.0f}%',
                       xy=(0.5, 50), fontsize=14, color='green', weight='bold',
                       xycoords='axes fraction')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = output_path / f"improvement_summary_{timestamp}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Saved chart: {chart_path}")
    
    plt.close()
    
    return chart_path


def generate_report(results: Dict[str, dict], output_dir: str = "charts"):
    """Generate a text report of the results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    report_lines = [
        "# Benchmark Results Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration Comparison",
        "",
        "| Configuration | SQL Gen | SQL Exec | Time (s) |",
        "|---------------|---------|----------|----------|",
    ]
    
    for config, data in results.items():
        m = data["metrics"]
        total = m["total"]
        sql_gen = f"{m.get('sql_success', 0)}/{total}"
        sql_exec = f"{m.get('execution_success', 0)}/{total}"
        time_val = f"{m['total_time']:.1f}"
        report_lines.append(f"| {config.capitalize():<13} | {sql_gen:<7} | {sql_exec:<8} | {time_val:<8} |")
    
    report_lines.extend([
        "",
        "## Improvement Analysis",
        "",
    ])
    
    if "baseline" in results and "hardcoded" in results:
        baseline = results["baseline"]["metrics"]
        hardcoded = results["hardcoded"]["metrics"]
        total = baseline["total"]
        
        if total > 0:
            exec_improvement = (hardcoded.get("execution_success", 0) - baseline.get("execution_success", 0)) / total * 100
            report_lines.append(f"- Hardcoded vs Baseline: +{exec_improvement:.1f}% execution success")
    
    if "optimized" in results and "hardcoded" in results:
        optimized = results["optimized"]["metrics"]
        hardcoded = results["hardcoded"]["metrics"]
        total = optimized["total"]
        
        if total > 0:
            exec_improvement = (optimized.get("execution_success", 0) - hardcoded.get("execution_success", 0)) / total * 100
            report_lines.append(f"- Optimized vs Hardcoded: +{exec_improvement:.1f}% execution success")
    
    report_content = "\n".join(report_lines)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"benchmark_report_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"Saved report: {report_path}")
    print("\n" + report_content)
    
    return report_path


def main():
    """Generate all visualizations and reports."""
    print("=" * 60)
    print("Benchmark Results Visualization")
    print("=" * 60)
    
    results = load_latest_results()
    
    if not results:
        print("\nNo benchmark results found. Run benchmark.py first.")
        return
    
    print("\nGenerating charts...")
    create_comparison_chart(results)
    create_improvement_summary(results)
    
    print("\nGenerating report...")
    generate_report(results)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

