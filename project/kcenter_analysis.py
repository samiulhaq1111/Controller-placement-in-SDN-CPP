"""
K-Center analysis module for SDN Controller Placement framework.

Provides detailed analysis, coverage statistics, visualisation, and a
research-oriented comparison report for the Greedy K-Center placement
strategy versus centrality-based baselines.

Outputs
-------
- ``kcenter_analysis.txt``          : selection rationale and coverage stats
- ``kcenter_coverage.csv``          : per-switch nearest-controller assignment
- ``kcenter_coverage.png``          : topology coloured by controller region
- ``kcenter_vs_centrality_analysis.txt`` : research comparison report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────

@dataclass
class SelectionStep:
    """One step of the greedy k-center selection process.

    Attributes
    ----------
    order : int
        1-based selection order.
    node : int
        Node ID selected.
    reason : str
        Human-readable explanation of why this node was chosen.
    min_distance : float
        Minimum distance from this node to any previously selected controller.
    """

    order: int
    node: int
    reason: str
    min_distance: float = 0.0


@dataclass
class CoverageRecord:
    """Per-switch coverage information.

    Attributes
    ----------
    switch : int
        Node ID of the switch.
    assigned_controller : int
        Nearest controller node ID.
    distance : float
        Hop distance to the assigned controller.
    """

    switch: int
    assigned_controller: int
    distance: float


# ──────────────────────────────────────────────
# Main analysis class
# ──────────────────────────────────────────────

class KCenterAnalysis:
    """Generate K-Center placement analysis artefacts.

    Parameters
    ----------
    graph : nx.Graph
        Network topology graph.
    distance_matrix : np.ndarray
        All-pairs shortest-path hop distance matrix.
    controllers : list[int]
        Controller nodes returned by ``k_center_placement()``.
    nodes : list[int] | None
        Sorted list of all node IDs.  Inferred from *graph* if ``None``.
    """

    def __init__(
        self,
        graph: nx.Graph,  # type: ignore[type-arg]
        distance_matrix: np.ndarray,
        controllers: list[int],
        nodes: list[int] | None = None,
    ) -> None:
        self.graph = graph
        self.distance_matrix = distance_matrix
        self.controllers: list[int] = list(controllers)
        self.nodes: list[int] = nodes if nodes is not None else sorted(graph.nodes())
        self._node_index: dict[int, int] = {n: i for i, n in enumerate(self.nodes)}

        # Lazy caches
        self._selection_order: list[SelectionStep] | None = None
        self._coverage: list[CoverageRecord] | None = None

    # ──────────────────────────────────────────
    # Selection order reconstruction
    # ──────────────────────────────────────────

    def get_selection_order(self) -> list[SelectionStep]:
        """Re-run the greedy selection to capture order and reasoning.

        Returns
        -------
        list[SelectionStep]
            Ordered list of controller selection steps.
        """
        if self._selection_order is not None:
            return self._selection_order

        cc: dict[int, float] = nx.closeness_centrality(self.graph)
        path_lengths: dict[int, dict[int, int]] = dict(
            nx.all_pairs_shortest_path_length(self.graph)
        )

        steps: list[SelectionStep] = []

        # Step 1 – highest closeness centrality seed
        first = max(self.nodes, key=lambda n: (cc.get(n, 0.0), -n))
        steps.append(SelectionStep(
            order=1,
            node=first,
            reason=(
                f"Selected as the initial controller because it has the "
                f"highest closeness centrality ({cc.get(first, 0.0):.6f}), "
                f"providing the best central starting point."
            ),
            min_distance=0.0,
        ))
        selected: list[int] = [first]

        # Steps 2..k – maximise minimum distance
        k = len(self.controllers)
        while len(selected) < k:
            best_node: int | None = None
            best_min_dist: float = -1.0
            for candidate in self.nodes:
                if candidate in selected:
                    continue
                min_dist = min(
                    path_lengths[candidate].get(s, float("inf"))
                    for s in selected
                )
                if (min_dist > best_min_dist) or (
                    min_dist == best_min_dist
                    and (best_node is None or candidate < best_node)
                ):
                    best_min_dist = min_dist
                    best_node = candidate

            if best_node is None:
                break

            nearest_ctrl = min(
                selected,
                key=lambda s: path_lengths[best_node].get(s, float("inf")),
            )
            steps.append(SelectionStep(
                order=len(selected) + 1,
                node=best_node,
                reason=(
                    f"Selected because it was furthest from existing "
                    f"controller(s).  Nearest existing controller is Node "
                    f"{nearest_ctrl} at {int(best_min_dist)} hop(s).  "
                    f"This maximises network coverage."
                ),
                min_distance=float(best_min_dist),
            ))
            selected.append(best_node)

        self._selection_order = steps
        return steps

    # ──────────────────────────────────────────
    # Coverage computation
    # ──────────────────────────────────────────

    def get_coverage(self) -> list[CoverageRecord]:
        """Assign every switch to its nearest controller.

        Returns
        -------
        list[CoverageRecord]
            One record per switch.
        """
        if self._coverage is not None:
            return self._coverage

        records: list[CoverageRecord] = []
        ctrl_indices = [self._node_index[c] for c in self.controllers]

        for node in self.nodes:
            i = self._node_index[node]
            best_ctrl: int | None = None
            best_dist: float = float("inf")
            for c, ci in zip(self.controllers, ctrl_indices):
                d = self.distance_matrix[i][ci]
                if d < best_dist:
                    best_dist = d
                    best_ctrl = c
            records.append(CoverageRecord(
                switch=node,
                assigned_controller=best_ctrl,  # type: ignore[arg-type]
                distance=best_dist,
            ))

        self._coverage = records
        return records

    def get_coverage_regions(self) -> dict[int, list[int]]:
        """Group switches by their assigned controller.

        Returns
        -------
        dict[int, list[int]]
            ``{controller: [switch_ids]}``
        """
        regions: dict[int, list[int]] = {c: [] for c in self.controllers}
        for rec in self.get_coverage():
            regions[rec.assigned_controller].append(rec.switch)
        return regions

    # ──────────────────────────────────────────
    # Export: kcenter_analysis.txt
    # ──────────────────────────────────────────

    def export_analysis(self, filepath: Path | str) -> None:
        """Write the K-Center selection analysis to a text file.

        Includes selection order, inter-controller distances, and
        coverage statistics.
        """
        steps = self.get_selection_order()
        coverage = self.get_coverage()
        regions = self.get_coverage_regions()
        distances = [rec.distance for rec in coverage]

        max_dist = max(distances) if distances else 0.0
        avg_dist = float(np.mean(distances)) if distances else 0.0

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("K-CENTER PLACEMENT — RESEARCH ANALYSIS")
        lines.append("=" * 60)
        lines.append("")

        # Selection order
        lines.append("CONTROLLER SELECTION ORDER")
        lines.append("-" * 60)
        for step in steps:
            lines.append(f"  Controller {step.order}:")
            lines.append(f"    Node {step.node}")
            lines.append(f"    Reason: {step.reason}")
            if step.min_distance > 0:
                lines.append(f"    Min Distance to Existing Controllers: {int(step.min_distance)} hop(s)")
            lines.append("")

        # Distances between controllers
        lines.append("DISTANCES BETWEEN CONTROLLERS")
        lines.append("-" * 60)
        for i, c1 in enumerate(self.controllers):
            for c2 in self.controllers[i + 1:]:
                d = self.distance_matrix[self._node_index[c1]][self._node_index[c2]]
                lines.append(f"  Node {c1} ↔ Node {c2}: {int(d)} hop(s)")
        lines.append("")

        # Coverage statistics
        lines.append("COVERAGE STATISTICS")
        lines.append("-" * 60)
        for ctrl in self.controllers:
            count = len(regions[ctrl])
            lines.append(f"  Controller {ctrl}: {count} switch(es)")
        lines.append("")
        lines.append(f"  Maximum Distance to Nearest Controller: {int(max_dist)} hop(s)")
        lines.append(f"  Average Distance to Nearest Controller: {avg_dist:.4f} hop(s)")
        lines.append("")

        # Research interpretation
        lines.append("RESEARCH INTERPRETATION")
        lines.append("-" * 60)
        lines.append(
            "  The Greedy K-Center strategy directly optimises for worst-case"
        )
        lines.append(
            "  latency by iteratively placing controllers at the node furthest"
        )
        lines.append(
            "  from all existing controllers.  This distance-aware approach is"
        )
        lines.append(
            "  expected to yield lower maximum latency compared to pure"
        )
        lines.append(
            "  centrality-based methods (Degree, Betweenness, Closeness), which"
        )
        lines.append(
            "  may cluster controllers around high-centrality hubs and leave"
        )
        lines.append(
            "  peripheral regions poorly covered."
        )
        lines.append("")
        lines.append(
            "  K-Center serves as a strong distance-oriented baseline for"
        )
        lines.append(
            "  benchmarking multi-objective meta-heuristics such as NSGA-II,"
        )
        lines.append(
            "  which must simultaneously optimise latency, load balance, and"
        )
        lines.append(
            "  controller importance."
        )
        lines.append("")
        lines.append("=" * 60)

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("K-Center analysis saved → %s", path)

    # ──────────────────────────────────────────
    # Export: kcenter_coverage.csv
    # ──────────────────────────────────────────

    def export_coverage_csv(self, filepath: Path | str) -> None:
        """Export per-switch coverage assignment to CSV.

        Columns: ``Switch``, ``AssignedController``, ``Distance``.
        """
        records = self.get_coverage()
        rows = [
            {
                "Switch": rec.switch,
                "AssignedController": rec.assigned_controller,
                "Distance": rec.distance,
            }
            for rec in records
        ]
        df = pd.DataFrame(rows)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("K-Center coverage CSV saved → %s", path)

    # ──────────────────────────────────────────
    # Export: kcenter_coverage.png
    # ──────────────────────────────────────────

    def generate_coverage_plot(self, filepath: Path | str) -> None:
        """Generate a topology plot coloured by controller coverage region.

        Controllers are highlighted with a distinct marker.  Switches are
        coloured according to their assigned controller.  Uses spring layout
        if no geographic coordinates are available in the graph.
        """
        regions = self.get_coverage_regions()
        controller_set = set(self.controllers)

        # Determine layout
        pos: dict[int, tuple[float, float]] | None = None
        if all("x" in self.graph.nodes[n] and "y" in self.graph.nodes[n]
               for n in self.graph.nodes()):
            pos = {
                n: (float(self.graph.nodes[n]["x"]),
                    float(self.graph.nodes[n]["y"]))
                for n in self.graph.nodes()
            }
        if pos is None:
            pos = nx.spring_layout(self.graph, seed=42)

        # Assign colours per controller
        cmap = plt.cm.get_cmap("tab10", len(self.controllers))  # type: ignore[attr-defined]
        ctrl_color_map: dict[int, Any] = {}
        for idx, ctrl in enumerate(self.controllers):
            ctrl_color_map[ctrl] = cmap(idx)

        # Build node colour list
        node_colors: list[Any] = []
        node_sizes: list[int] = []
        node_labels: dict[int, str] = {}
        for node in self.graph.nodes():
            if node in controller_set:
                node_colors.append(ctrl_color_map[node])
                node_sizes.append(350)
                node_labels[node] = f"C{node}"
            else:
                # Find assigned controller
                assigned: int | None = None
                for ctrl, switches in regions.items():
                    if node in switches:
                        assigned = ctrl
                        break
                node_colors.append(
                    ctrl_color_map[assigned] if assigned else "gray"
                )
                node_sizes.append(80)

        fig, ax = plt.subplots(figsize=(12, 9))
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax, alpha=0.15, width=0.8
        )
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="black",
            linewidths=0.5,
        )

        # Draw controller labels prominently
        ctrl_pos = {n: pos[n] for n in self.controllers if n in pos}
        ctrl_labels = {n: f"C{n}" for n in self.controllers}
        nx.draw_networkx_labels(
            self.graph, ctrl_pos, ctrl_labels,
            font_size=10, font_weight="bold",
            font_color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="black",
                edgecolor="black",
                alpha=0.8,
            ),
            ax=ax,
        )

        # Legend
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(
                facecolor=ctrl_color_map[ctrl],
                edgecolor="black",
                label=f"Controller {ctrl} ({len(regions[ctrl])} switches)",
            )
            for ctrl in self.controllers
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            fontsize=9,
            title="Controller Regions",
        )

        ax.set_title(
            "K-Center Controller Coverage Regions",
            fontsize=14, fontweight="bold",
        )
        ax.axis("off")
        fig.tight_layout()

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("K-Center coverage plot saved → %s", path)

    # ──────────────────────────────────────────
    # Export: kcenter_vs_centrality_analysis.txt
    # ──────────────────────────────────────────

    def export_vs_centrality_analysis(
        self,
        filepath: Path | str,
        comparison_results: list[Any],
    ) -> None:
        """Write a research comparison report: KCenter vs centrality baselines.

        Parameters
        ----------
        filepath : Path | str
            Destination text file.
        comparison_results : list[StrategyResult]
            Results from ``StrategyComparison.results``.
        """
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("K-CENTER vs CENTRALITY-BASED PLACEMENT — RESEARCH COMPARISON")
        lines.append("=" * 70)
        lines.append("")

        # Header
        lines.append("STRATEGY COMPARISON TABLE")
        lines.append("-" * 70)
        header = (
            f"  {'Strategy':<14s} {'AvgLat':>8s} {'MaxLat':>8s} "
            f"{'LoadVar':>10s} {'Spread':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 50)

        for r in comparison_results:
            lines.append(
                f"  {r.method:<14s} {r.average_latency:>8.4f} "
                f"{r.maximum_latency:>8.0f} {r.load_variance:>10.4f} "
                f"{r.controller_spread:>8.4f}"
            )
        lines.append("")

        # Per-metric analysis
        metrics_info = [
            ("Average Latency", "average_latency", "min",
             "Lower average latency indicates that switches are, on average, "
             "closer to their nearest controller, reducing overall network "
             "response time."),
            ("Maximum Latency", "maximum_latency", "min",
             "Lower maximum latency means the worst-case switch-to-controller "
             "distance is minimised, ensuring no switch is left far from a "
             "controller.  This is the primary objective of K-Center."),
            ("Load Variance", "load_variance", "min",
             "Lower load variance indicates a more balanced distribution of "
             "switches across controllers, preventing controller overload."),
            ("Controller Spread", "controller_spread", "max",
             "Higher spread indicates controllers are geographically or "
             "topologically distributed more broadly, which typically "
             "correlates with better coverage and fault tolerance."),
        ]

        for label, key, direction, interpretation in metrics_info:
            lines.append(f"ANALYSIS: {label.upper()}")
            lines.append("-" * 70)

            if direction == "min":
                ranked = sorted(comparison_results, key=lambda r: getattr(r, key))
                best = ranked[0]
            else:
                ranked = sorted(
                    comparison_results, key=lambda r: getattr(r, key), reverse=True
                )
                best = ranked[0]

            best_val = getattr(best, key)
            lines.append(f"  Best strategy : {best.method} ({best_val:.4f})")
            lines.append("")
            lines.append(f"  Rankings:")
            for rank, r in enumerate(ranked, 1):
                val = getattr(r, key)
                marker = " ← best" if rank == 1 else ""
                lines.append(f"    {rank}. {r.method:<14s} {val:.4f}{marker}")
            lines.append("")
            lines.append(f"  Interpretation:")
            lines.append(f"    {interpretation}")
            lines.append("")

        # Overall summary
        lines.append("OVERALL RESEARCH SUMMARY")
        lines.append("-" * 70)

        # Identify best per key metric
        best_avg = min(comparison_results, key=lambda r: r.average_latency)
        best_max = min(comparison_results, key=lambda r: r.maximum_latency)
        best_var = min(comparison_results, key=lambda r: r.load_variance)
        best_spr = max(comparison_results, key=lambda r: r.controller_spread)

        lines.append(f"  Minimises Average Latency   : {best_avg.method} ({best_avg.average_latency:.4f})")
        lines.append(f"  Minimises Maximum Latency  : {best_max.method} ({best_max.maximum_latency:.0f})")
        lines.append(f"  Minimises Load Variance    : {best_var.method} ({best_var.load_variance:.4f})")
        lines.append(f"  Maximises Controller Spread: {best_spr.method} ({best_spr.controller_spread:.4f})")
        lines.append("")

        lines.append("  DISCUSSION:")
        lines.append(
            "  The K-Center strategy is specifically designed to minimise the "
            "maximum"
        )
        lines.append(
            "  switch-to-controller distance, making it the theoretically "
            "optimal greedy"
        )
        lines.append(
            "  approach for worst-case latency reduction.  Centrality-based "
            "methods"
        )
        lines.append(
            "  (Degree, Betweenness, Closeness) prioritise placing controllers "
            "at topologically"
        )
        lines.append(
            "  important nodes, which may yield lower average latency but often "
            "cluster"
        )
        lines.append(
            "  controllers in the network core, leaving peripheral switches "
            "underserved."
        )
        lines.append("")
        lines.append(
            "  The Hybrid strategy balances multiple centrality metrics but "
            "does not"
        )
        lines.append(
            "  explicitly optimise for distance.  K-Center complements these "
            "approaches"
        )
        lines.append(
            "  by providing a distance-aware baseline that directly targets "
            "coverage."
        )
        lines.append("")
        lines.append(
            "  For future NSGA-II implementation, K-Center provides the "
            "distance-minimisation"
        )
        lines.append(
            "  extreme of the Pareto front.  An ideal multi-objective "
            "optimiser should"
        )
        lines.append(
            "  find solutions that simultaneously approach K-Center's maximum "
            "latency while"
        )
        lines.append(
            "  maintaining the load balance and importance benefits of "
            "centrality-based methods."
        )
        lines.append("")
        lines.append("=" * 70)

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("K-Center vs centrality analysis saved → %s", path)
