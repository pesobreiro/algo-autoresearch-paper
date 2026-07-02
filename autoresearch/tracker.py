"""
Tracker de experiências — persistência JSON com tags e notas humanas.

Schema de uma experiência:
{
    "iteracao": 12,
    "timestamp_iso": "2026-03-14T03:22:11",
    "git_commit": "a3f9c2d",
    "status": "aceite",
    "tags": ["promising"],
    "nota_humana": "",
    "alteracoes_vs_anterior": "FEATURES: added volume_norm_15m; STOCH_THRESHOLD: 20→15",
    "metricas": {
        "sharpe_raw": 2.14,
        "retorno_anual_pct": 127.0,
        "max_drawdown_pct": 22.1,
        "win_rate_pct": 39.5,
        "n_trades": 163,
        "score_composto": 0.681
    },
    "params_hash": "d4e5f6a7",
    "labels_reutilizados": true,
    "duracao_total_segundos": 87.3
}
"""
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

TAGS_VALIDAS = {'promising', 'baseline', 'explorado', 'rejeitado', 'interessante', 'bug'}


@dataclass
class RegistoExperiencia:
    """Registo completo de uma iteração da pesquisa."""
    iteracao: int
    timestamp_iso: str
    git_commit: str
    status: str  # "aceite" | "rejeitado" | "erro"
    metricas: dict
    params_hash: str
    labels_reutilizados: bool
    duracao_total_segundos: float
    tags: list = field(default_factory=list)
    nota_humana: str = ""
    alteracoes_vs_anterior: str = ""
    params_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'RegistoExperiencia':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _git_commit_hash() -> str:
    """Retorna o hash curto do último commit git."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


class Tracker:
    """Gestão de experiências: guardar, carregar, tagging, análise."""

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._cache: list[RegistoExperiencia] = []
        self._carregar_cache()

    def _ficheiro_iteracao(self, iteracao: int) -> Path:
        return self.experiments_dir / f'iter_{iteracao:04d}.json'

    def _carregar_cache(self):
        """Carrega todos os registos em memória."""
        self._cache = []
        for f in sorted(self.experiments_dir.glob('iter_*.json')):
            try:
                d = json.loads(f.read_text())
                self._cache.append(RegistoExperiencia.from_dict(d))
            except Exception:
                pass

    def proximo_numero_iteracao(self) -> int:
        if not self._cache:
            return 1
        return max(r.iteracao for r in self._cache) + 1

    def guardar_experiencia(self, registo: RegistoExperiencia) -> Path:
        """Guarda uma experiência em ficheiro JSON."""
        caminho = self._ficheiro_iteracao(registo.iteracao)
        caminho.write_text(json.dumps(registo.to_dict(), indent=2, ensure_ascii=False))

        # Atualizar cache
        self._cache = [r for r in self._cache if r.iteracao != registo.iteracao]
        self._cache.append(registo)
        self._cache.sort(key=lambda r: r.iteracao)

        return caminho

    def criar_registo(self, iteracao: int, status: str, metricas: dict,
                      params_hash: str, labels_reutilizados: bool,
                      duracao: float, alteracoes: str = "",
                      params_snapshot: dict = None) -> RegistoExperiencia:
        """Cria um novo RegistoExperiencia."""
        return RegistoExperiencia(
            iteracao=iteracao,
            timestamp_iso=datetime.now().isoformat(timespec='seconds'),
            git_commit=_git_commit_hash(),
            status=status,
            metricas=metricas,
            params_hash=params_hash,
            labels_reutilizados=labels_reutilizados,
            duracao_total_segundos=duracao,
            alteracoes_vs_anterior=alteracoes,
            params_snapshot=params_snapshot or {},
        )

    def adicionar_tag(self, iteracao: int, tag: str, nota: str = "") -> bool:
        """Adiciona uma tag e nota a uma experiência existente."""
        if tag not in TAGS_VALIDAS:
            console.print(f"[red]Tag inválida: '{tag}'. Válidas: {TAGS_VALIDAS}[/red]")
            return False

        caminho = self._ficheiro_iteracao(iteracao)
        if not caminho.exists():
            console.print(f"[red]Iteração {iteracao} não encontrada[/red]")
            return False

        d = json.loads(caminho.read_text())
        if tag not in d.get('tags', []):
            d.setdefault('tags', []).append(tag)
        if nota:
            d['nota_humana'] = nota
        caminho.write_text(json.dumps(d, indent=2, ensure_ascii=False))

        # Atualizar cache
        for r in self._cache:
            if r.iteracao == iteracao:
                if tag not in r.tags:
                    r.tags.append(tag)
                if nota:
                    r.nota_humana = nota
                break

        console.print(f"[green]Tag '{tag}' adicionada à iteração {iteracao}[/green]")
        return True

    def carregar_experiencias_por_tag(self, tag: str) -> list[RegistoExperiencia]:
        """Retorna experiências com a tag especificada."""
        return [r for r in self._cache if tag in r.tags]

    def resultado_ja_encontrado(self, metricas: dict, tolerancia: float = 0.001) -> bool:
        """Verifica se este resultado Optuna já foi encontrado antes (mesmo ótimo local)."""
        sl  = metricas.get('sl_pct')
        tp  = metricas.get('tp_pct')
        thr = metricas.get('threshold')
        if sl is None or tp is None or thr is None:
            return False
        for r in self._cache:
            if abs(r.metricas.get('score_composto', 0)) < 0.001:
                continue  # ignorar rejeitados de validação/duplicados
            m = r.metricas
            if (abs(m.get('sl_pct', -1) - sl) < tolerancia and
                abs(m.get('tp_pct', -1) - tp) < tolerancia and
                abs(m.get('threshold', -1) - thr) < tolerancia):
                return True
        return False

    def _sort_key(self, r: 'RegistoExperiencia') -> float:
        """Chave de ordenação: sharpe_validation se disponível, senão score_composto."""
        sv = r.metricas.get('sharpe_validation')
        return float(sv) if sv is not None else r.metricas.get('score_composto', -999.0)

    def top_n_scores(self, n: int = 5) -> list[dict]:
        """Retorna os N melhores resultados com pipeline executado (score != 0)."""
        validos = [r for r in self._cache
                   if abs(r.metricas.get('score_composto', 0)) > 0.001
                   or r.metricas.get('sharpe_validation') is not None]
        return [r.to_dict() for r in sorted(
            validos, key=self._sort_key, reverse=True
        )[:n]]

    def melhor_score(self) -> Optional[RegistoExperiencia]:
        """Retorna a experiência com o melhor score (sharpe_validation ou score_composto)."""
        aceites = [r for r in self._cache if r.status == 'aceite']
        if not aceites:
            return None
        return max(aceites, key=self._sort_key)

    def ultimo_aceite(self) -> Optional[RegistoExperiencia]:
        """Retorna a última experiência aceite."""
        aceites = [r for r in self._cache if r.status == 'aceite']
        if not aceites:
            return None
        return max(aceites, key=lambda r: r.iteracao)

    def gerar_relatorio_analise(self) -> str:
        """Gera tabela rich com análise por tag + trend de score."""
        if not self._cache:
            return "Nenhuma experiência registada ainda."

        linhas = []

        # Tabela principal — todas as iterações
        table = Table(
            title=f"Experiências — {len(self._cache)} iterações",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Iter", justify="right", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Sharpe", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("DD%", justify="right", style="red")
        table.add_column("€500 OOS", justify="right", style="magenta")
        table.add_column("Trades", justify="right")
        table.add_column("Tags", style="yellow")
        table.add_column("Nota")

        for r in self._cache:
            m = r.metricas
            status_style = {
                'aceite': '[green]aceite[/green]',
                'rejeitado': '[red]rejeitado[/red]',
                'erro': '[orange1]erro[/orange1]',
            }.get(r.status, r.status)

            equity = m.get('equity_500_final')
            if equity:
                lucro = equity - 500
                equity_str = f"€{equity:.0f} ({lucro:+.0f})"
            else:
                # retroactivo: calcular a partir do retorno total se disponível
                ret_total = m.get('retorno_total_oos_pct') or m.get('retorno_anual_pct', 0)
                if ret_total:
                    eq = 500 * (1 + ret_total / 100)
                    equity_str = f"€{eq:.0f} ({eq-500:+.0f})"
                else:
                    equity_str = '—'

            table.add_row(
                str(r.iteracao),
                status_style,
                f"{m.get('score_composto', 0):.4f}",
                f"{m.get('sharpe_raw', 0):.2f}",
                f"{m.get('retorno_total_oos_pct') or m.get('retorno_anual_pct', 0):+.1f}%",
                f"{abs(m.get('max_drawdown_pct', 0)):.1f}%",
                equity_str,
                str(m.get('n_trades', 0)),
                ', '.join(r.tags) or '—',
                r.nota_humana[:30] if r.nota_humana else '—',
            )

        console.print(table)

        # Melhor resultado
        melhor = self.melhor_score()
        if melhor:
            console.print(f"\n[bold green]Melhor score:[/bold green] "
                          f"Iter {melhor.iteracao} — "
                          f"Score {melhor.metricas.get('score_composto', 0):.4f} | "
                          f"Sharpe {melhor.metricas.get('sharpe_raw', 0):.2f}")

        # Por tag
        for tag in TAGS_VALIDAS:
            tagged = self.carregar_experiencias_por_tag(tag)
            if tagged:
                console.print(f"\n[yellow]{tag}[/yellow] ({len(tagged)} iterações): "
                              f"{', '.join(str(r.iteracao) for r in tagged)}")

        # Trend do score (últimas 10)
        recentes = self._cache[-10:]
        if len(recentes) > 1:
            scores = [r.metricas.get('score_composto', 0) for r in recentes]
            trend = "↑" if scores[-1] > scores[0] else "↓"
            console.print(f"\nTrend (últimas {len(recentes)}): {trend} "
                          f"{scores[0]:.4f} → {scores[-1]:.4f}")

        return ""

    def hash_ja_explorado(self, params_hash: str) -> bool:
        """Verifica se esta configuração exacta já foi testada (aceite ou rejeitada com pipeline)."""
        return any(
            r.params_hash == params_hash and r.status in ('aceite', 'rejeitado', 'erro')
            and r.metricas.get('score_composto', 0) != 0.0  # exclui rejeições de validação (score=0)
            for r in self._cache
        )

    def listar_historico(self, n: int = 10) -> list[dict]:
        """Retorna os últimos N registos como lista de dicts."""
        recentes = self._cache[-n:] if len(self._cache) > n else self._cache
        return [r.to_dict() for r in recentes]

    def calcular_alteracoes(self, params_anterior: dict, params_atual: dict) -> str:
        """Compara dois dicts de params e retorna descrição das alterações."""
        alteracoes = []
        for chave in params_atual:
            if chave not in params_anterior:
                alteracoes.append(f"{chave}: NOVO={params_atual[chave]}")
            elif params_anterior[chave] != params_atual[chave]:
                alteracoes.append(f"{chave}: {params_anterior[chave]}→{params_atual[chave]}")
        return "; ".join(alteracoes) if alteracoes else "sem alterações detectadas"
