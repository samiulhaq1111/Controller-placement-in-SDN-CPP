"""
Strategy comparison module for SDN Controller Placement framework.

Evaluates every baseline placement strategy, collects metrics, generates
a formatted console report, exports results to CSV, creates comparison
bar-chart plots, and writes a statistical summary text file.

Design Notes
------------
The module follows the **Strategy Pattern**: any new placement method
simply needs to provide a ``dict[str, list[int]]`` mapping
``{method_name: controller_nodes}`` — the comparison pipeline handles
the rest automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from objectives import ControllerPlacementObjectives  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────

@dataclass
class StrategyResult:
    """Container for a single strategy's evaluation output.

    Attributes
    ----------
    method : str
        Human-readable strategy name.
    controllers : list[int]
        Selected controller node IDs.
    average_latency : float
        Mean switch-to-nearest-controller hop distance.
    maximum_latency : float
        Worst-case hop distance.
    load_variance : float
        Population variance of controller loads.
    importance_score : float
        Sum of betweenness centrality for chosen controllers.
    controller_spread : float
        Average pairwise hop distance between selected controllers.
    controller_loads : dict[int, int]
        ``{controller: managed_switch_count}``.
    """

    method: str
    controllers: list[int]
    average_latency: float = 0.0
    maximum_latency: float = 0.0
    load_variance: float = 0.0
    importance_score: float = 0.0
    controller_spread: float = 0.0
    controller_loads: dict[int, int] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Main comparison class
# ──────────────────────────────────────────────

class StrategyComparison:
    """Evaluate and compare multiple placement strategies.

    Parameters
    ----------
    graph : nx.Graph
        Network topology graph.
    distance_matrix : np.ndarray
        All-pairs shortest-path distance matrix (rows/cols = ``sorted(graph.nodes())``).
    betweenness_scores : dict[int, float]
        Pre-computed betweenness centrality for every node.
    """

    # Metric keys and their display labels / optimisation direction
    _METRICS: list[dict[str, str]] = [
        {"key": "average_latency",  "label": "Average Latency",    "direction": "min"},
        {"key": "maximum_latency",  "label": "Maximum Latency",    "direction": "min"},
        {"key": "load_variance",    "label": "Load Variance",      "direction": "min"},
        {"key": "importance_score", "label": "Importance Score",   "direction": "max"},
        {"key": "controller_spread","label": "Controller Spread",  "direction": "max"},
    ]

    def __init__(
        self,
        graph: Any,  # nx.Graph
        distance_matrix: np.ndarray,
        betweenness_scores: dict[int, float],
    ) -> None:
        self.graph = graph
        self.distance_matrix = distance_matrix
        self.betweenness_scores = betweenness_scores
        self.nodes: list[int] = sorted(graph.nodes())

        self.results: list[StrategyResult] = []

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def evaluate_strategies(
        self,
        placements: dict[str, list[int]],
    ) -> list[StrategyResult]:
        """Evaluate every strategy in *placements* and store results.

        Parameters
        ----------
        placements : dict[str, list[int]]
            ``{method_name: [controller_nodes]}``.

        Returns
        -------
        list[StrategyResult]
            One result per strategy, in the same order.
        """
        self.results.clear()

        for method, controllers in placements.items():
            logger.info("Evaluating strategy: %s  controllers=%s", method, controllers)

            obj = ControllerPlacementObjectives(
                graph=self.graph,
                distance_matrix=self.distance_matrix,
                controller_nodes=controllers,
            )

            result = StrategyResult(
                method=method,
                controllers=controllers,
                average_latency=obj.average_controller_latency(),
                maximum_latency=obj.maximum_controller_latency(),
                load_variance=obj.load_variance(),
                importance_score=obj.controller_importance_score(self.betweenness_scores),
                controller_spread=self._compute_controller_spread(controllers),
                controller_loads=obj.controller_load_distribution(),
            )
            self.results.append(result)

        logger.info("Evaluated %d strategies.", len(self.results))
        return self.results

    # ──────────────────────────────────────────
    # Console report
    # ──────────────────────────────────────────

    def print_report(self) -> None:
        """Print a formatted comparison report to stdout."""
        self._require_results()
        sep = "=" * 50

        print()
        print(sep)
        print("BASELINE STRATEGY COMPARISON")
        print(sep)

        for r in self.results:
            print()
            print(f"  Method             : {r.method}")
            print(f"  Controllers        : {r.controllers}")
            print(f"  Average Latency    : {r.average_latency:.4f}")
            print(f"  Maximum Latency    : {r.maximum_latency:.0f}")
            print(f"  Load Variance      : {r.load_variance:.4f}")
            print(f"  Importance Score   : {r.importance_score:.6f}")
            loads_str = ", ".join(
                f"{c}: {n}" for c, n in sorted(r.controller_loads.items())
            )
            print(f"  Controller Loads   : {{{loads_str}}}")
            print(f"  Controller Spread  : {r.controller_spread:.4f}")
            print("-" * 50)

        # Best-strategy summary
        print()
        print(sep)
        print("BEST STRATEGY SUMMARY")
        print(sep)
        for metric in self._METRICS:
            best = self._best_strategy(metric["key"], metric["direction"])
            label = metric["label"]
            value = getattr(best, metric["key"])
            print(f"  {label:24s}: {best.method} ({value:.4f})")
        print()
        print(sep)

    # ──────────────────────────────────────────
    # CSV export
    # ──────────────────────────────────────────

    def export_csv(self, filepath: Path | str) -> None:
        """Save all strategy results to a CSV file.

        Columns: ``Method``, ``Controllers``, ``AverageLatency``,
        ``MaximumLatency``, ``LoadVariance``, ``ImportanceScore``,
        ``ControllerLoads``.
        """
        self._require_results()
        rows = []
        for r in self.results:
            rows.append({
                "Method": r.method,
                "Controllers": str(r.controllers),
                "AverageLatency": round(r.average_latency, 4),
                "MaximumLatency": r.maximum_latency,
                "LoadVariance": round(r.load_variance, 4),
                "ImportanceScore": round(r.importance_score, 6),
                "ControllerSpread": round(r.controller_spread, 4),
                "ControllerLoads": str(r.controller_loads),
            })
        df = pd.DataFrame(rows)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Baseline comparison CSV saved → %s", path)

    # ──────────────────────────────────────────
    # Statistical summary text file
    # ──────────────────────────────────────────

    def export_summary(self, filepath: Path | str) -> None:
        """Write a human-readable statistical summary to a text file.

        Includes best strategy per metric and full strategy rankings.
        """
        self._require_results()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("BASELINE STRATEGY COMPARISON — STATISTICAL SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        # Best strategy per metric
        lines.append("BEST STRATEGY PER METRIC")
        lines.append("-" * 60)
        for metric in self._METRICS:
            best = self._best_strategy(metric["key"], metric["direction"])
            value = getattr(best, metric["key"])
            lines.append(
                f"  {metric['label']:24s} → {best.method:14s} (value: {value:.4f})"
            )
        lines.append("")

        # Rankings
        lines.append("STRATEGY RANKINGS (best → worst)")
        lines.append("-" * 60)
        for metric in self._METRICS:
            key = metric["key"]
            reverse = metric["direction"] == "min"  # ascending = best first
            ranked = sorted(
                self.results,
                key=lambda r: getattr(r, key),
                reverse=not reverse,
            )
            lines.append(f"  {metric['label']}:")
            for rank, r in enumerate(ranked, 1):
                value = getattr(r, key)
                lines.append(f"    {rank}. {r.method:14s}  {value:.4f}")
            lines.append("")

        # Detailed results table
        lines.append("DETAILED RESULTS")
        lines.append("-" * 60)
        header = (
            f"  {'Method':<14s} {'AvgLat':>8s} {'MaxLat':>8s} "
            f"{'LoadVar':>10s} {'ImpScore':>10s} {'Spread':>8s}  Controllers"
        )
        lines.append(header)
        lines.append("  " + "-" * 64)
        for r in self.results:
            lines.append(
                f"  {r.method:<14s} {r.average_latency:>8.4f} "
                f"{r.maximum_latency:>8.0f} {r.load_variance:>10.4f} "
                f"{r.importance_score:>10.6f} {r.controller_spread:>8.4f}  {r.controllers}"
            )
        lines.append("")
        lines.append("=" * 60)

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Strategy summary saved → %s", path)

    # ──────────────────────────────────────────
    # Visualisations
    # ──────────────────────────────────────────

    def generate_plots(self, output_dir: Path | str) -> None:
        """Create one bar-chart PNG per metric and save to *output_dir*.

        Files
        -----
        - ``average_latency_comparison.png``
        - ``maximum_latency_comparison.png``
        - ``load_variance_comparison.png``
        - ``importance_score_comparison.png``
        """
        self._require_results()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        methods = [r.method for r in self.results]
        n = len(methods)
        x = np.arange(n)
        bar_width = 0.55

        for metric in self._METRICS:
            key = metric["key"]
            label = metric["label"]
            values = [getattr(r, key) for r in self.results]

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(x, values, width=bar_width, color="#4C72B0", edgecolor="black")

            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.4f}" if isinstance(val, float) and val < 1 else f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_xlabel("Strategy", fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f"{label} Comparison", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()

            filename = f"{key}_comparison.png"
            fig.savefig(out / filename, dpi=150)
            plt.close(fig)
            logger.info("Plot saved → %s", out / filename)

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _require_results(self) -> None:
        """Raise ``RuntimeError`` if no results are available."""
        if not self.results:
            raise RuntimeError(
                "No strategy results available. "
                "Call evaluate_strategies() first."
            )

    def _best_strategy(self, metric_key: str, direction: str) -> StrategyResult:
        """Return the best strategy for a given metric.

        Parameters
        ----------
        metric_key : str
            Attribute name on :class:`StrategyResult`.
        direction : str
            ``"min"`` (lower is better) or ``"max"`` (higher is better).

        Returns
        -------
        StrategyResult
        """
        if direction == "min":
            return min(self.results, key=lambda r: getattr(r, metric_key))
        return max(self.results, key=lambda r: getattr(r, metric_key))

    def _compute_controller_spread(self, controllers: list[int]) -> float:
        """Compute the average pairwise hop distance between controllers.

        Parameters
        ----------
        controllers : list[int]
            Selected controller node IDs.

        Returns
        -------
        float
            Mean pairwise shortest-path distance.
        """
        if len(controllers) < 2:
            return 0.0

        node_index: dict[int, int] = {n: i for i, n in enumerate(self.nodes)}
        total = 0.0
        count = 0
        for i, c1 in enumerate(controllers):
            for c2 in controllers[i + 1:]:
                total += self.distance_matrix[node_index[c1]][node_index[c2]]
                count += 1
        return total / count if count > 0 else 0.0
