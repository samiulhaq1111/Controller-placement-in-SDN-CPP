"""
Baseline controller placement strategies for SDN analysis.

Provides deterministic and stochastic heuristics that serve as benchmarks
for meta-heuristic optimisers (NSGA-II, GA, RL, etc.).

Strategies
----------
- **Random**      : uniformly random node selection
- **Degree**      : top-k nodes by degree
- **Betweenness** : top-k nodes by betweenness centrality
- **Closeness**   : top-k nodes by closeness centrality
- **Hybrid**      : weighted combination of normalised graph metrics

New strategies can be added by implementing any callable that accepts
``(graph, k, **kwargs)`` and returns ``list[int]``.
"""

from __future__ import annotations

import logging
import random as _random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    HYBRID_BETWEENNESS_WEIGHT,
    HYBRID_CLOSENESS_WEIGHT,
    HYBRID_DEGREE_WEIGHT,
    validate_hybrid_weights,
)

logger = logging.getLogger(__name__)


class PlacementStrategies:
    """Graph-theoretic heuristics for controller placement.

    Parameters
    ----------
    graph : nx.Graph
        The network topology graph.  All node identifiers must be integers.

    Notes
    -----
    Every strategy returns a **sorted** list of unique node IDs so that
    downstream evaluation is deterministic regardless of selection order.
    """

    def __init__(self, graph: nx.Graph) -> None:  # type: ignore[type-arg]
        self.graph = graph
        self.nodes: list[int] = sorted(graph.nodes())

    # ──────────────────────────────────────────
    # Strategy 1 – Random Placement
    # ──────────────────────────────────────────

    def random_placement(
        self, k: int, seed: int | None = None
    ) -> list[int]:
        """Select *k* unique nodes uniformly at random.

        Parameters
        ----------
        k : int
            Number of controllers to place.
        seed : int | None
            Optional random seed for reproducibility.

        Returns
        -------
        list[int]
            Sorted list of selected node IDs.

        Raises
        ------
        ValueError
            If *k* exceeds the number of available nodes.
        """
        self._validate_k(k)
        rng = _random.Random(seed)
        selected = sorted(rng.sample(self.nodes, k))
        logger.info(
            "Random placement (seed=%s, k=%d): %s", seed, k, selected
        )
        return selected

    # ──────────────────────────────────────────
    # Strategy 2 – Degree-Based Placement
    # ──────────────────────────────────────────

    def degree_based_placement(self, k: int) -> list[int]:
        """Select the top-*k* nodes by degree (descending).

        Parameters
        ----------
        k : int
            Number of controllers.

        Returns
        -------
        list[int]
            Sorted list of highest-degree node IDs.
        """
        self._validate_k(k)
        ranked = self._rank_by_metric(dict(self.graph.degree()), k)
        logger.info("Degree-based placement (k=%d): %s", k, ranked)
        return ranked

    # ──────────────────────────────────────────
    # Strategy 3 – Betweenness-Based Placement
    # ──────────────────────────────────────────

    def betweenness_based_placement(self, k: int) -> list[int]:
        """Select the top-*k* nodes by betweenness centrality (descending).

        Parameters
        ----------
        k : int
            Number of controllers.

        Returns
        -------
        list[int]
            Sorted list of highest-betweenness node IDs.
        """
        self._validate_k(k)
        bc: dict[int, float] = nx.betweenness_centrality(self.graph)
        ranked = self._rank_by_metric(bc, k)
        logger.info(
            "Betweenness-based placement (k=%d): %s", k, ranked
        )
        return ranked

    # ──────────────────────────────────────────
    # Strategy 4 – Closeness-Based Placement
    # ──────────────────────────────────────────

    def closeness_based_placement(self, k: int) -> list[int]:
        """Select the top-*k* nodes by closeness centrality (descending).

        Parameters
        ----------
        k : int
            Number of controllers.

        Returns
        -------
        list[int]
            Sorted list of highest-closeness node IDs.
        """
        self._validate_k(k)
        cc: dict[int, float] = nx.closeness_centrality(self.graph)
        ranked = self._rank_by_metric(cc, k)
        logger.info("Closeness-based placement (k=%d): %s", k, ranked)
        return ranked

    # ──────────────────────────────────────────
    # Strategy 5 – Hybrid Placement
    # ──────────────────────────────────────────

    def hybrid_placement(self, k: int) -> list[int]:
        """Select top-*k* nodes by a weighted hybrid score.

        The hybrid score combines min-max normalised degree, betweenness
        centrality, and closeness centrality using configurable weights
        defined in ``config.py``.

        Parameters
        ----------
        k : int
            Number of controllers.

        Returns
        -------
        list[int]
            Sorted list of top hybrid-score node IDs.
        """
        self._validate_k(k)
        validate_hybrid_weights()

        scores = self._compute_hybrid_scores()
        ranked = self._rank_by_metric(scores, k)
        logger.info("Hybrid placement (k=%d): %s", k, ranked)
        return ranked

    # ──────────────────────────────────────────
    # Hybrid score accessor (for HybridScorer)
    # ──────────────────────────────────────────

    def get_hybrid_score_table(self) -> pd.DataFrame:
        """Compute and return the full hybrid score table for all nodes.

        Returns
        -------
        pd.DataFrame
            Sorted by HybridScore descending.  Columns: Node, Degree,
            DegreeNorm, Betweenness, BetweennessNorm, Closeness,
            ClosenessNorm, HybridScore.
        """
        validate_hybrid_weights()
        return self._build_score_dataframe()

    # ──────────────────────────────────────────
    # Convenience: run all strategies at once
    # ──────────────────────────────────────────

    def all_strategies(
        self,
        k: int,
        seed: int | None = None,
    ) -> dict[str, list[int]]:
        """Return placements from every built-in strategy.

        Parameters
        ----------
        k : int
            Number of controllers.
        seed : int | None
            Random seed forwarded to :meth:`random_placement`.

        Returns
        -------
        dict[str, list[int]]
            ``{strategy_name: [node_ids]}``
        """
        return {
            "Random": self.random_placement(k, seed=seed),
            "Degree": self.degree_based_placement(k),
            "Betweenness": self.betweenness_based_placement(k),
            "Closeness": self.closeness_based_placement(k),
            "Hybrid": self.hybrid_placement(k),
        }

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _validate_k(self, k: int) -> None:
        """Raise ``ValueError`` if *k* is invalid."""
        n = len(self.nodes)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        if k > n:
            raise ValueError(
                f"k ({k}) exceeds the number of nodes in the graph ({n})."
            )

    @staticmethod
    def _rank_by_metric(
        scores: dict[int, Any], k: int
    ) -> list[int]:
        """Return the top-*k* node IDs sorted by *scores* descending,
        then return them as a **sorted** list for deterministic output.

        Parameters
        ----------
        scores : dict[int, Any]
            ``{node_id: metric_value}`` mapping.
        k : int
            Number of top nodes to select.

        Returns
        -------
        list[int]
            Sorted list of the *k* best node IDs.
        """
        # Sort by metric value descending, break ties by node ID ascending
        ranked = sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )
        return sorted(node for node, _ in ranked[:k])

    # ──────────────────────────────────────────
    # Hybrid score internals
    # ──────────────────────────────────────────

    def _compute_hybrid_scores(self) -> dict[int, float]:
        """Compute the weighted hybrid score for every node.

        Returns
        -------
        dict[int, float]
            ``{node: hybrid_score}``.
        """
        degree_raw: dict[int, int] = dict(self.graph.degree())
        bc_raw: dict[int, float] = nx.betweenness_centrality(self.graph)
        cc_raw: dict[int, float] = nx.closeness_centrality(self.graph)

        deg_norm = self._min_max_normalize(degree_raw)
        bc_norm = self._min_max_normalize(bc_raw)
        cc_norm = self._min_max_normalize(cc_raw)

        scores: dict[int, float] = {}
        for node in self.nodes:
            scores[node] = (
                HYBRID_DEGREE_WEIGHT * deg_norm.get(node, 0.0)
                + HYBRID_BETWEENNESS_WEIGHT * bc_norm.get(node, 0.0)
                + HYBRID_CLOSENESS_WEIGHT * cc_norm.get(node, 0.0)
            )
        return scores

    def _build_score_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame with raw + normalised metrics and hybrid score.

        Returns
        -------
        pd.DataFrame
            Sorted by HybridScore descending.
        """
        degree_raw: dict[int, int] = dict(self.graph.degree())
        bc_raw: dict[int, float] = nx.betweenness_centrality(self.graph)
        cc_raw: dict[int, float] = nx.closeness_centrality(self.graph)

        deg_norm = self._min_max_normalize(degree_raw)
        bc_norm = self._min_max_normalize(bc_raw)
        cc_norm = self._min_max_normalize(cc_raw)

        rows: list[dict[str, Any]] = []
        for node in self.nodes:
            dn = deg_norm.get(node, 0.0)
            bn = bc_norm.get(node, 0.0)
            cn = cc_norm.get(node, 0.0)
            hybrid = (
                HYBRID_DEGREE_WEIGHT * dn
                + HYBRID_BETWEENNESS_WEIGHT * bn
                + HYBRID_CLOSENESS_WEIGHT * cn
            )
            rows.append({
                "Node": node,
                "Degree": degree_raw[node],
                "DegreeNorm": round(dn, 6),
                "Betweenness": round(bc_raw.get(node, 0.0), 6),
                "BetweennessNorm": round(bn, 6),
                "Closeness": round(cc_raw.get(node, 0.0), 6),
                "ClosenessNorm": round(cn, 6),
                "HybridScore": round(hybrid, 6),
            })

        df = pd.DataFrame(rows)
        df.sort_values("HybridScore", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _min_max_normalize(values: dict[int, Any]) -> dict[int, float]:
        """Apply min-max normalisation to a node-score mapping.

        Parameters
        ----------
        values : dict[int, Any]
            ``{node: numeric_value}``.

        Returns
        -------
        dict[int, float]
            ``{node: normalised_value}`` in [0, 1].
        """
        vals = [float(v) for v in values.values()]
        vmin = min(vals)
        vmax = max(vals)
        rng = vmax - vmin
        if rng == 0:
            return {n: 0.0 for n in values}
        return {n: (float(v) - vmin) / rng for n, v in values.items()}


# ══════════════════════════════════════════════
# Hybrid analysis & reporting helper
# ══════════════════════════════════════════════

class HybridScorer:
    """Generate hybrid-score artefacts: CSV, correlations, analysis, plot.

    This class is intentionally decoupled from :class:`PlacementStrategies`
    so that future hybrid variants (different weights, distance-based
    scores, etc.) can be created without modifying the strategy class.

    Parameters
    ----------
    strategies : PlacementStrategies
        The parent strategies instance (used to access the graph and
        pre-computed metrics).
    """

    def __init__(self, strategies: PlacementStrategies) -> None:
        self.strat = strategies
        self.df: pd.DataFrame = strategies.get_hybrid_score_table()

    # ── CSV export ────────────────────────────

    def export_scores_csv(self, filepath: Path | str) -> None:
        """Save the full hybrid score table to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        logger.info("Hybrid scores CSV saved → %s", path)

    # ── Pearson correlations ──────────────────

    def export_correlations(self, filepath: Path | str) -> None:
        """Compute and export Pearson correlation matrix.

        Columns: DegreeNorm, BetweennessNorm, ClosenessNorm, HybridScore.
        """
        cols = ["DegreeNorm", "BetweennessNorm", "ClosenessNorm", "HybridScore"]
        corr = self.df[cols].corr(method="pearson")
        corr = corr.round(4)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(path)
        logger.info("Hybrid correlations CSV saved → %s", path)

    # ── Top-nodes bar chart ───────────────────

    def generate_top_nodes_plot(
        self, filepath: Path | str, top_n: int = 10
    ) -> None:
        """Bar chart of the top *top_n* nodes ranked by HybridScore.

        Parameters
        ----------
        filepath : Path | str
            Destination PNG path.
        top_n : int
            Number of top nodes to display.
        """
        top = self.df.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(top))
        bars = ax.bar(
            x,
            top["HybridScore"].values,
            width=0.6,
            color="#DD8452",
            edgecolor="black",
        )
        for bar, val in zip(bars, top["HybridScore"].values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_xlabel("Node ID", fontsize=12)
        ax.set_ylabel("Hybrid Score", fontsize=12)
        ax.set_title(
            f"Top {top_n} Nodes by Hybrid Score",
            fontsize=14, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(top["Node"].values, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Hybrid top-nodes plot saved → %s", path)

    # ── Research analysis text ────────────────

    def export_analysis(
        self, filepath: Path | str, top_n: int = 10
    ) -> None:
        """Write a research-oriented analysis text file.

        Compares the top-*n* nodes across degree, betweenness, closeness,
        and hybrid rankings and highlights overlaps / divergences.
        """
        graph = self.strat.graph

        # Compute per-metric rankings (top_n node sets)
        deg_ranked = sorted(
            dict(graph.degree()).items(), key=lambda x: (-x[1], x[0])
        )[:top_n]
        bc_ranked = sorted(
            nx.betweenness_centrality(graph).items(),
            key=lambda x: (-x[1], x[0]),
        )[:top_n]
        cc_ranked = sorted(
            nx.closeness_centrality(graph).items(),
            key=lambda x: (-x[1], x[0]),
        )[:top_n]
        hybrid_ranked = self.df.head(top_n)["Node"].tolist()

        top_deg = set(n for n, _ in deg_ranked)
        top_bc = set(n for n, _ in bc_ranked)
        top_cc = set(n for n, _ in cc_ranked)
        top_hyb = set(hybrid_ranked)

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("HYBRID PLACEMENT — RESEARCH ANALYSIS")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Weights: Degree={HYBRID_DEGREE_WEIGHT}, "
                      f"Betweenness={HYBRID_BETWEENNESS_WEIGHT}, "
                      f"Closeness={HYBRID_CLOSENESS_WEIGHT}")
        lines.append("")

        # Top-10 tables
        sections = [
            ("Top 10 Hybrid-Ranked Nodes", hybrid_ranked),
            ("Top 10 Degree Nodes", [n for n, _ in deg_ranked]),
            ("Top 10 Betweenness Nodes", [n for n, _ in bc_ranked]),
            ("Top 10 Closeness Nodes", [n for n, _ in cc_ranked]),
        ]
        for title, node_list in sections:
            lines.append(f"  {title}")
            lines.append(f"  {'-' * 40}")
            lines.append(f"    Nodes: {sorted(node_list)}")
            lines.append("")

        # Overlap analysis
        all_sets = {"Degree": top_deg, "Betweenness": top_bc,
                     "Closeness": top_cc, "Hybrid": top_hyb}
        common_all = top_deg & top_bc & top_cc & top_hyb
        lines.append("OVERLAP ANALYSIS")
        lines.append(f"  {'-' * 40}")
        lines.append(f"    Nodes common to ALL four top-10 lists: {sorted(common_all) if common_all else 'None'}")
        lines.append("")

        names = list(all_sets.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = all_sets[names[i]] & all_sets[names[j]]
                lines.append(
                    f"    {names[i]} ∩ {names[j]}: "
                    f"{sorted(overlap)} ({len(overlap)} nodes)"
                )
        lines.append("")

        # Unique to hybrid
        only_hybrid = top_hyb - top_deg - top_bc - top_cc
        lines.append("  Nodes UNIQUE to Hybrid top-10 (not in any single-metric top-10):")
        lines.append(f"    {sorted(only_hybrid) if only_hybrid else 'None'}")
        lines.append("")

        # Interpretation
        lines.append("INTERPRETATION")
        lines.append(f"  {'-' * 40}")
        lines.append(
            "  The hybrid score balances connectivity (degree), bridging "
            "importance (betweenness),"
        )
        lines.append(
            "  and reachability (closeness). Nodes that rank highly in the "
            "hybrid metric but not"
        )
        lines.append(
            "  in any single metric represent balanced candidates that may "
            "offer robust controller"
        )
        lines.append(
            "  placement across multiple network-performance dimensions. "
            "These are strong baseline"
        )
        lines.append(
            "  candidates for NSGA-II seeding or warm-start initialisation."
        )
        lines.append("")
        lines.append("=" * 60)

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Hybrid analysis saved → %s", path)
