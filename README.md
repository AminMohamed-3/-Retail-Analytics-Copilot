# Retail Analytics Copilot

A local AI agent that answers retail analytics questions using RAG over documentation and SQL queries over the Northwind SQLite database. Built with DSPy and LangGraph.

## Performance Results

### Benchmark Results (12 questions)

| Configuration | Execution Success | Improvement |
|---------------|-------------------|-------------|
| Baseline (no fixes) | 7/12 (58.3%) | - |
| With Hardcoded Fixes | 9/12 (75.0%) | **+16.7%** |
| Optimized + Hardcoded | 10/12 (83.3%) | **+25.0%** |

### NL-to-SQL Optimization Results (5 validation queries)

| Configuration | Execution Success | Improvement |
|---------------|-------------------|-------------|
| Baseline (no fixes) | 3/5 (60.0%) | - |
| With Hardcoded Fixes | 5/5 (100.0%) | **+40.0%** |
| Optimized (no fixes) | 4/5 (80.0%) | **+20.0%** |

### Router Optimization Results

| Configuration | Accuracy | Change |
|---------------|----------|--------|
| Baseline Router | 3/4 (75%) | - |
| Optimized Router | 2/4 (50%) | **-25%** |

**Note**: Router optimization degraded performance. The baseline router is used in production.

## Key Findings

### Why Hardcoded Fixes Work

The Phi-3.5 model generates SQL with consistent, predictable errors:

1. **Date Function Errors**: Uses `YEAR()` (MySQL syntax) instead of SQLite's `BETWEEN`
2. **Alias Typos**: Writes `cats.CategoryName` instead of `cat.CategoryName`
3. **Keyword Hallucinations**: Generates `BETWEDIR`, `BETWEWITHIN` instead of `BETWEEN`
4. **Date Shift**: Forgets to shift 1997 → 2017 for the Northwind database

Our `_fix_sql_typos()` function catches and corrects these patterns, providing +16.7% to +40% improvement.

### Why Router Optimization Didn't Help

BootstrapFewShot optimization on small models (Phi-3.5) for the router task:
- **Overfits** on training examples
- **Loses generalization** ability
- **Baseline outperforms** optimized version by 25%

**Recommendation**: Use the baseline router for production.

### Why Teacher-Student Optimization

Small models cannot effectively optimize themselves. We use:
- **Teacher Model**: Qwen3 4B - Generates better demonstrations
- **Student Model**: Phi-3.5-mini - The model being optimized

This improved NL-to-SQL from 60% → 80% (+20%) without hardcoded fixes.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ROUTER                                  │
│              Classifies: rag | sql | hybrid                     │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │   RAG    │       │   SQL    │       │  HYBRID  │
    │ Retriever│       │ NL→SQL   │       │   Both   │
    └──────────┘       └──────────┘       └──────────┘
          │                   │                   │
          │                   ▼                   │
          │           ┌──────────┐                │
          │           │ Executor │◄───────────────┘
          │           └──────────┘
          │                   │
          │            ┌──────┴──────┐
          │            │             │
          │         Success       Error
          │            │             │
          │            │             ▼
          │            │      ┌──────────┐
          │            │      │  REPAIR  │
          │            │      │   LOOP   │──┐
          │            │      └──────────┘  │
          │            │             ▲      │
          │            │             └──────┘
          │            │             (max 2x)
          │            │
          ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SYNTHESIZER                               │
│           Combines results + formats final answer               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Final Answer                              │
│    { final_answer, sql, confidence, explanation, citations }    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt

# Ollama models
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M  # Student model (inference)
ollama pull qwen3:4b                           # Teacher model (optimization)
```

### Database Setup

```bash
mkdir -p data
curl -L -o data/northwind.sqlite \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

sqlite3 data/northwind.sqlite <<'SQL'
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;
CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details";
CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;
CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;
SQL
```

### Verify Setup

```bash
python config.py
```

## Usage Guide

### 1. Run the Agent (Main Usage)

```bash
# Answer questions from a file
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs.jsonl

# Answer a single question
python run_agent_hybrid.py --question "How many orders in 1997?"
```

### 2. Run Benchmarks

```bash
# Run benchmark on all questions
python benchmark.py --config baseline      # No fixes
python benchmark.py --config hardcoded     # With hardcoded fixes

# Generate visualization
python visualize_results.py
```

### 3. Run Optimization (Optional)

```bash
# Optimize NL-to-SQL module
python optimize_nl_to_sql.py

# Optimize Router (not recommended - degrades performance)
python optimize_router_bootstrap.py
```

### 4. Test Components

```bash
python test_agent.py
```

## Project Structure

```
Retail_Analytics_Copilot/
├── agent/
│   ├── graph_hybrid.py         # LangGraph workflow (7 nodes)
│   ├── dspy_signatures.py      # DSPy modules (Router, NL-SQL, Synthesizer)
│   ├── rag/retrieval.py        # TF-IDF document retriever
│   └── tools/sqlite_tool.py    # SQLite database access
├── data/
│   ├── northwind.sqlite        # Northwind database
│   └── training_set.py         # 36 training examples
├── docs/                        # RAG document corpus
├── config.py                    # DSPy & DB configuration
├── run_agent_hybrid.py          # CLI entrypoint
├── optimize_nl_to_sql.py        # BootstrapFewShot optimization
├── benchmark.py                 # Benchmark runner
├── visualize_results.py         # Results visualization
└── benchmark_questions.jsonl    # 12 benchmark questions
```

## Output Format

```json
{
  "id": "question_id",
  "final_answer": "<matches format_hint>",
  "sql": "<last executed SQL>",
  "confidence": 0.0-1.0,
  "explanation": "Brief explanation",
  "citations": ["Orders", "kpi_definitions::chunk2"]
}
```

## Technical Notes

### Date Shift (Important!)
The Northwind database dates are **shifted +20 years** from the original:
- Campaign "Summer Beverages **1997**" → Database dates: **2017**-06-01 to **2017**-06-30
- Campaign "Winter Classics **1997**" → Database dates: **2017**-12-01 to **2017**-12-31

The hardcoded fixes automatically convert 1997→2017 in SQL queries.

### CostOfGoods Approximation

The Northwind database lacks a `CostOfGoods` field. For Gross Margin calculations, we assume `CostOfGoods ≈ 0.7 × UnitPrice` (flat rate). 

**Trade-off**: Simple and consistent, but less accurate than category-level averages. Implemented via DSPy prompt instructions rather than hardcoded SQL.

### Other Notes
- All processing is local; no external API calls at inference
- Ollama `keep_alive=0` prevents context caching issues
- Cache clearing is automatic in benchmark mode
- Repair loop with dynamic hints improves SQL error recovery
- Trace logs saved to `traces/` directory for debugging
