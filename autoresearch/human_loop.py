"""
Human interaction loop — review and control of the agent.

Available commands:
  [r] review   — view latest iterations
  [i] inject   — inject params manually
  [p] pause    — pause the loop
  [h] history  — view full history
  [t] tag      — add tag to an iteration
  [a] analysis — tag analysis + trend
  [q] quit     — exit
  [c] continue — continue (Enter key)
"""
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm

from autoresearch.tracker import Tracker, VALID_TAGS

console = Console()


def show_iteration_result(iteration: int, result, previous_score: float):
    """Displays the result of an iteration in a formatted way."""
    if not result.sucesso:
        console.print(Panel(
            f"[red]ERROR:[/red] {result.erro}",
            title=f"Iteration {iteration} — FAILED",
            border_style="red",
        ))
        return

    metrics = result.metricas
    score = metrics.get('score_composto', 0)
    delta = score - previous_score
    delta_str = f"[green]+{delta:.4f}[/green]" if delta >= 0 else f"[red]{delta:.4f}[/red]"

    cache_str = "[yellow](labels cached)[/yellow]" if result.labels_reutilizados else "[blue](new labels)[/blue]"

    # OOS Equity
    equity_final = metrics.get('equity_500_final')
    equity_by_year = metrics.get('equity_500_por_ano', {})
    if equity_final:
        profit = equity_final - 500
        equity_str = f"€{equity_final:.0f} ({profit:+.0f}€)"
        if equity_by_year:
            equity_str += "  [" + "  ".join(f"{yr}→€{eq:.0f}" for yr, eq in sorted(equity_by_year.items())) + "]"
    else:
        equity_str = "n/a"

    content = (
        f"Score: {score:.4f} ({delta_str} vs previous)\n"
        f"Sharpe: {metrics.get('sharpe_raw', 0):.2f} | "
        f"Return: {metrics.get('retorno_anual_pct', 0):+.1f}% | "
        f"DD: {abs(metrics.get('max_drawdown_pct', 0)):.1f}%\n"
        f"Trades: {metrics.get('n_trades', 0)} | Win%: {metrics.get('win_rate_pct', 0):.1f}%\n"
        f"Equity OOS (500€): {equity_str}\n"
        f"Time: {result.duracao_total_segundos:.0f}s {cache_str}"
    )
    color = "green" if delta >= 0 else "yellow"
    console.print(Panel(content, title=f"Iteration {iteration}", border_style=color))


def show_help():
    console.print(Panel(
        "[cyan]r[/cyan] review     — view latest iterations\n"
        "[cyan]i[/cyan] inject     — inject research_params.py manually\n"
        "[cyan]p[/cyan] pause      — pause the loop (confirm to continue)\n"
        "[cyan]h[/cyan] history    — view full history\n"
        "[cyan]t[/cyan] tag        — add tag to an iteration\n"
        "[cyan]a[/cyan] analysis   — tag analysis + trend\n"
        "[cyan]q[/cyan] quit       — exit the agent\n"
        "[cyan]Enter[/cyan]        — continue to next iteration",
        title="Available commands",
        border_style="blue",
    ))


def request_human_review(tracker: Tracker, params_path: Path,
                         iteration: int, config: dict) -> dict:
    """
    Pauses for human review after an iteration.

    Returns:
        dict with action: {'action': 'continue'|'pause'|'exit'|'inject', 'params_path': ...}
    """
    console.print(f"\n[bold blue]═══ Human Review — Iteration {iteration} ═══[/bold blue]")
    show_help()

    while True:
        try:
            command = Prompt.ask(
                "\nCommand",
                choices=['r', 'i', 'p', 'h', 't', 'a', 'q', ''],
                default='',
                show_choices=False,
            ).lower().strip()
        except (KeyboardInterrupt, EOFError):
            return {'action': 'exit'}

        if command == '' or command == 'c':
            return {'action': 'continue'}

        elif command == 'r':
            history = tracker.list_history(10)
            if not history:
                console.print("[yellow]No iterations yet.[/yellow]")
            else:
                for h in history:
                    metrics = h.get('metricas', {})
                    console.print(
                        f"  Iter {h['iteracao']:4d} | {h['status']:10s} | "
                        f"Score={metrics.get('score_composto', 0):.4f} | "
                        f"Sharpe={metrics.get('sharpe_raw', 0):.2f} | "
                        f"Tags={h.get('tags', [])}"
                    )

        elif command == 'h':
            tracker.generate_analysis_report()

        elif command == 'a':
            tracker.generate_analysis_report()

        elif command == 't':
            try:
                iter_num = int(Prompt.ask("Iteration number"))
                tag = Prompt.ask(f"Tag", choices=list(VALID_TAGS))
                note = Prompt.ask("Note (optional)", default="")
                tracker.add_tag(iter_num, tag, note)
            except (ValueError, KeyboardInterrupt):
                console.print("[red]Cancelled[/red]")

        elif command == 'p':
            console.print("[yellow]Loop paused. Press Enter to continue or Ctrl+C to exit.[/yellow]")
            try:
                input()
                return {'action': 'continue'}
            except KeyboardInterrupt:
                return {'action': 'exit'}

        elif command == 'i':
            console.print("[cyan]Manual injection mode[/cyan]")
            console.print(f"Edit the file: {params_path}")
            console.print("Press Enter when ready (the agent will use these params).")
            try:
                input()
                # Show current content
                if params_path.exists():
                    syntax = Syntax(params_path.read_text(), "python", theme="monokai", line_numbers=True)
                    console.print(syntax)
                return {'action': 'inject', 'params_path': params_path}
            except KeyboardInterrupt:
                console.print("[red]Cancelled[/red]")

        elif command == 'q':
            if Confirm.ask("Exit the agent?"):
                return {'action': 'exit'}


def show_proposed_params(previous_code: str, new_code: str):
    """Visual diff between previous and new params."""
    console.print("\n[bold]Parameters proposed by the agent:[/bold]")
    syntax = Syntax(new_code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)
