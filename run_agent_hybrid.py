"""Main CLI entrypoint for the hybrid agent."""
import json
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

from agent.graph_hybrid import HybridAgent

console = Console()


@click.command()
@click.option('--batch', required=True, help='Input JSONL file with questions')
@click.option('--out', required=True, help='Output JSONL file for results')
def main(batch: str, out: str):
    """Run the hybrid agent on a batch of questions."""
    # Check input file exists
    batch_path = Path(batch)
    if not batch_path.exists():
        console.print(f"[red]Error: Input file not found: {batch}[/red]")
        return
    
    # Load questions
    questions = []
    with open(batch_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    
    console.print(f"[green]Loaded {len(questions)} questions[/green]")
    
    # Initialize agent
    console.print("[yellow]Initializing agent...[/yellow]")
    try:
        agent = HybridAgent()
        console.print("[green]Agent initialized successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        return
    
    # Process questions
    results = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing questions...", total=len(questions))
        
        for q in questions:
            question_id = q.get("id", "unknown")
            question = q.get("question", "")
            format_hint = q.get("format_hint", "str")
            
            console.print(f"\n[bold]Processing: {question_id}[/bold]")
            console.print(f"Question: {question[:80]}...")
            
            try:
                result = agent.process_question(question, question_id, format_hint)
                results.append(result)
                console.print(f"[green]✓ Completed {question_id}[/green]")
            except Exception as e:
                console.print(f"[red]✗ Error processing {question_id}: {e}[/red]")
                # Add error result
                results.append({
                    "id": question_id,
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "citations": []
                })
            
            progress.update(task, advance=1)
    
    # Write results
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    console.print(f"\n[green]Results written to: {out}[/green]")
    console.print(f"[green]Processed {len(results)} questions[/green]")


if __name__ == "__main__":
    main()

