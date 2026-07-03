"""
Experience tracker — JSON persistence with tags and human notes.

Schema of an experience:
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

VALID_TAGS = {'promising', 'baseline', 'explorado', 'rejeitado', 'interessante', 'bug'}


@dataclass
class ExperimentRecord:
    """Full record of a research iteration."""
    iteracao: int
    timestamp_iso: str
    git_commit: str
    status: str  # "accepted" | "rejected" | "error"
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
    def from_dict(cls, d: dict) -> 'ExperimentRecord':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _git_commit_hash() -> str:
    """Returns the short hash of the last git commit."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


class Tracker:
    """Experience management: save, load, tagging, analysis."""

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._cache: list[ExperimentRecord] = []
        self._load_cache()

    def _iteration_file(self, iteration: int) -> Path:
        return self.experiments_dir / f'iter_{iteration:04d}.json'

    def _load_cache(self):
        """Loads all records into memory."""
        self._cache = []
        for f in sorted(self.experiments_dir.glob('iter_*.json')):
            try:
                d = json.loads(f.read_text())
                self._cache.append(ExperimentRecord.from_dict(d))
            except Exception:
                pass

    def next_iteration_number(self) -> int:
        if not self._cache:
            return 1
        return max(r.iteracao for r in self._cache) + 1

    def save_experience(self, record: ExperimentRecord) -> Path:
        """Saves an experience to a JSON file."""
        path = self._iteration_file(record.iteracao)
        path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))

        # Update cache
        self._cache = [r for r in self._cache if r.iteracao != record.iteracao]
        self._cache.append(record)
        self._cache.sort(key=lambda r: r.iteracao)

        return path

    def create_record(self, iteration: int, status: str, metricas: dict,
                      params_hash: str, labels_reutilizados: bool,
                      duracao: float, alteracoes: str = "",
                      params_snapshot: dict = None) -> ExperimentRecord:
        """Creates a new ExperimentRecord."""
        return ExperimentRecord(
            iteracao=iteration,
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

    def add_tag(self, iteration: int, tag: str, note: str = "") -> bool:
        """Adds a tag and note to an existing experience."""
        if tag not in VALID_TAGS:
            console.print(f"[red]Invalid tag: '{tag}'. Valid: {VALID_TAGS}[/red]")
            return False

        path = self._iteration_file(iteration)
        if not path.exists():
            console.print(f"[red]Iteration {iteration} not found[/red]")
            return False

        d = json.loads(path.read_text())
        if tag not in d.get('tags', []):
            d.setdefault('tags', []).append(tag)
        if note:
            d['nota_humana'] = note
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False))

        # Update cache
        for r in self._cache:
            if r.iteracao == iteration:
                if tag not in r.tags:
                    r.tags.append(tag)
                if note:
                    r.nota_humana = note
                break

        console.print(f"[green]Tag '{tag}' added to iteration {iteration}[/green]")
        return True

    def load_experiences_by_tag(self, tag: str) -> list[ExperimentRecord]:
        """Returns experiences with the specified tag."""
        return [r for r in self._cache if tag in r.tags]

    def result_already_found(self, metricas: dict, tolerance: float = 0.001) -> bool:
        """Checks whether this Optuna result was already found before (same local optimum)."""
        sl  = metricas.get('sl_pct')
        tp  = metricas.get('tp_pct')
        thr = metricas.get('threshold')
        if sl is None or tp is None or thr is None:
            return False
        for r in self._cache:
            if abs(r.metricas.get('score_composto', 0)) < 0.001:
                continue  # ignore validation/duplicate rejections
            m = r.metricas
            if (abs(m.get('sl_pct', -1) - sl) < tolerance and
                abs(m.get('tp_pct', -1) - tp) < tolerance and
                abs(m.get('threshold', -1) - thr) < tolerance):
                return True
        return False

    def _sort_key(self, r: 'ExperimentRecord') -> float:
        """Sort key: sharpe_validation if available, otherwise score_composto."""
        sv = r.metricas.get('sharpe_validation')
        return float(sv) if sv is not None else r.metricas.get('score_composto', -999.0)

    def top_n_scores(self, n: int = 5) -> list[dict]:
        """Returns the top N results with executed pipeline (score != 0)."""
        valid = [r for r in self._cache
                   if abs(r.metricas.get('score_composto', 0)) > 0.001
                   or r.metricas.get('sharpe_validation') is not None]
        return [r.to_dict() for r in sorted(
            valid, key=self._sort_key, reverse=True
        )[:n]]

    def best_score(self) -> Optional[ExperimentRecord]:
        """Returns the experience with the best score (sharpe_validation or score_composto)."""
        accepted = [r for r in self._cache if r.status == 'aceite']
        if not accepted:
            return None
        return max(accepted, key=self._sort_key)

    def last_accepted(self) -> Optional[ExperimentRecord]:
        """Returns the last accepted experience."""
        accepted = [r for r in self._cache if r.status == 'aceite']
        if not accepted:
            return None
        return max(accepted, key=lambda r: r.iteracao)

    def generate_analysis_report(self) -> str:
        """Generates a rich table with tag analysis + score trend."""
        if not self._cache:
            return "No experiences recorded yet."

        # Main table — all iterations
        table = Table(
            title=f"Experiences — {len(self._cache)} iterations",
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
        table.add_column("Note")

        for r in self._cache:
            m = r.metricas
            status_style = {
                'aceite': '[green]aceite[/green]',
                'rejeitado': '[red]rejeitado[/red]',
                'erro': '[orange1]erro[/orange1]',
            }.get(r.status, r.status)

            equity = m.get('equity_500_final')
            if equity:
                profit = equity - 500
                equity_str = f"€{equity:.0f} ({profit:+.0f})"
            else:
                # retroactive: calculate from total return if available
                total_return = m.get('retorno_total_oos_pct') or m.get('retorno_anual_pct', 0)
                if total_return:
                    eq = 500 * (1 + total_return / 100)
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

        # Best result
        best = self.best_score()
        if best:
            console.print(f"\n[bold green]Best score:[/bold green] "
                          f"Iter {best.iteracao} — "
                          f"Score {best.metricas.get('score_composto', 0):.4f} | "
                          f"Sharpe {best.metricas.get('sharpe_raw', 0):.2f}")

        # By tag
        for tag in VALID_TAGS:
            tagged = self.load_experiences_by_tag(tag)
            if tagged:
                console.print(f"\n[yellow]{tag}[/yellow] ({len(tagged)} iterations): "
                              f"{', '.join(str(r.iteracao) for r in tagged)}")

        # Score trend (last 10)
        recent = self._cache[-10:]
        if len(recent) > 1:
            scores = [r.metricas.get('score_composto', 0) for r in recent]
            trend = "↑" if scores[-1] > scores[0] else "↓"
            console.print(f"\nTrend (last {len(recent)}): {trend} "
                          f"{scores[0]:.4f} → {scores[-1]:.4f}")

        return ""

    def hash_already_explored(self, params_hash: str) -> bool:
        """Checks whether this exact config was already tested (accepted or rejected with pipeline)."""
        return any(
            r.params_hash == params_hash and r.status in ('aceite', 'rejeitado', 'erro')
            and r.metricas.get('score_composto', 0) != 0.0  # exclude validation rejections (score=0)
            for r in self._cache
        )

    def list_history(self, n: int = 10) -> list[dict]:
        """Returns the last N records as a list of dicts."""
        recent = self._cache[-n:] if len(self._cache) > n else self._cache
        return [r.to_dict() for r in recent]

    def compute_changes(self, previous_params: dict, current_params: dict) -> str:
        """Compares two param dicts and returns a description of changes."""
        changes = []
        for key in current_params:
            if key not in previous_params:
                changes.append(f"{key}: NEW={current_params[key]}")
            elif previous_params[key] != current_params[key]:
                changes.append(f"{key}: {previous_params[key]}→{current_params[key]}")
        return "; ".join(changes) if changes else "no changes detected"
