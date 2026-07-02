"""
Loop de interação humana — revisão e controlo do agente.

Comandos disponíveis:
  [r] review   — ver últimas iterações
  [i] inject   — injetar params manualmente
  [p] pause    — pausar o loop
  [h] history  — ver histórico completo
  [t] tag      — adicionar tag a uma iteração
  [a] analysis — análise por tag + trend
  [q] quit     — terminar
  [c] continue — continuar (tecla Enter)
"""
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm

from autoresearch.tracker import Tracker, TAGS_VALIDAS

console = Console()


def mostrar_resultado_iteracao(iteracao: int, resultado, score_anterior: float):
    """Mostra o resultado de uma iteração de forma formatada."""
    if not resultado.sucesso:
        console.print(Panel(
            f"[red]ERRO:[/red] {resultado.erro}",
            title=f"Iteração {iteracao} — FALHOU",
            border_style="red",
        ))
        return

    m = resultado.metricas
    score = m.get('score_composto', 0)
    delta = score - score_anterior
    delta_str = f"[green]+{delta:.4f}[/green]" if delta >= 0 else f"[red]{delta:.4f}[/red]"

    cache_str = "[yellow](labels cached)[/yellow]" if resultado.labels_reutilizados else "[blue](labels novos)[/blue]"

    # Equity OOS
    equity_final = m.get('equity_500_final')
    equity_por_ano = m.get('equity_500_por_ano', {})
    if equity_final:
        lucro = equity_final - 500
        equity_str = f"€{equity_final:.0f} ({lucro:+.0f}€)"
        if equity_por_ano:
            equity_str += "  [" + "  ".join(f"{yr}→€{eq:.0f}" for yr, eq in sorted(equity_por_ano.items())) + "]"
    else:
        equity_str = "n/a"

    conteudo = (
        f"Score: {score:.4f} ({delta_str} vs anterior)\n"
        f"Sharpe: {m.get('sharpe_raw', 0):.2f} | "
        f"Return: {m.get('retorno_anual_pct', 0):+.1f}% | "
        f"DD: {abs(m.get('max_drawdown_pct', 0)):.1f}%\n"
        f"Trades: {m.get('n_trades', 0)} | Win%: {m.get('win_rate_pct', 0):.1f}%\n"
        f"Equity OOS (500€): {equity_str}\n"
        f"Tempo: {resultado.duracao_total_segundos:.0f}s {cache_str}"
    )
    cor = "green" if delta >= 0 else "yellow"
    console.print(Panel(conteudo, title=f"Iteração {iteracao}", border_style=cor))


def mostrar_ajuda():
    console.print(Panel(
        "[cyan]r[/cyan] review     — ver últimas iterações\n"
        "[cyan]i[/cyan] inject     — injetar research_params.py manualmente\n"
        "[cyan]p[/cyan] pause      — pausar o loop (confirmar para continuar)\n"
        "[cyan]h[/cyan] history    — ver histórico completo\n"
        "[cyan]t[/cyan] tag        — adicionar tag a uma iteração\n"
        "[cyan]a[/cyan] analysis   — análise por tag + trend\n"
        "[cyan]q[/cyan] quit       — terminar o agente\n"
        "[cyan]Enter[/cyan]        — continuar para próxima iteração",
        title="Comandos disponíveis",
        border_style="blue",
    ))


def solicitar_revisao_humana(tracker: Tracker, params_path: Path,
                              iteracao: int, config: dict) -> dict:
    """
    Pausa para revisão humana após uma iteração.

    Returns:
        dict com ação: {'acao': 'continuar'|'pausar'|'sair'|'injetar', 'params_path': ...}
    """
    console.print(f"\n[bold blue]═══ Revisão Humana — Iteração {iteracao} ═══[/bold blue]")
    mostrar_ajuda()

    while True:
        try:
            cmd = Prompt.ask(
                "\nComando",
                choices=['r', 'i', 'p', 'h', 't', 'a', 'q', ''],
                default='',
                show_choices=False,
            ).lower().strip()
        except (KeyboardInterrupt, EOFError):
            return {'acao': 'sair'}

        if cmd == '' or cmd == 'c':
            return {'acao': 'continuar'}

        elif cmd == 'r':
            historico = tracker.listar_historico(10)
            if not historico:
                console.print("[yellow]Nenhuma iteração ainda.[/yellow]")
            else:
                for h in historico:
                    m = h.get('metricas', {})
                    console.print(
                        f"  Iter {h['iteracao']:4d} | {h['status']:10s} | "
                        f"Score={m.get('score_composto', 0):.4f} | "
                        f"Sharpe={m.get('sharpe_raw', 0):.2f} | "
                        f"Tags={h.get('tags', [])}"
                    )

        elif cmd == 'h':
            tracker.gerar_relatorio_analise()

        elif cmd == 'a':
            tracker.gerar_relatorio_analise()

        elif cmd == 't':
            try:
                n = int(Prompt.ask("Número da iteração"))
                tag = Prompt.ask(f"Tag", choices=list(TAGS_VALIDAS))
                nota = Prompt.ask("Nota (opcional)", default="")
                tracker.adicionar_tag(n, tag, nota)
            except (ValueError, KeyboardInterrupt):
                console.print("[red]Cancelado[/red]")

        elif cmd == 'p':
            console.print("[yellow]Loop pausado. Prima Enter para continuar ou Ctrl+C para sair.[/yellow]")
            try:
                input()
                return {'acao': 'continuar'}
            except KeyboardInterrupt:
                return {'acao': 'sair'}

        elif cmd == 'i':
            console.print("[cyan]Modo injeção manual[/cyan]")
            console.print(f"Editar o ficheiro: {params_path}")
            console.print("Prima Enter quando pronto (o agente usará estes params).")
            try:
                input()
                # Mostrar o conteúdo atual
                if params_path.exists():
                    syntax = Syntax(params_path.read_text(), "python", theme="monokai", line_numbers=True)
                    console.print(syntax)
                return {'acao': 'injetar', 'params_path': params_path}
            except KeyboardInterrupt:
                console.print("[red]Cancelado[/red]")

        elif cmd == 'q':
            if Confirm.ask("Terminar o agente?"):
                return {'acao': 'sair'}


def mostrar_params_propostos(codigo_anterior: str, codigo_novo: str):
    """Mostra diff visual entre params anterior e novo."""
    console.print("\n[bold]Parâmetros propostos pelo agente:[/bold]")
    syntax = Syntax(codigo_novo, "python", theme="monokai", line_numbers=True)
    console.print(syntax)
