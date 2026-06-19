#!/usr/bin/env python3
"""
Main entry point for the SDN Controller Placement framework.

Workflow
--------
1.  Load the GEANT 2010 topology from GML.
2.  Compute all graph metrics (degree, betweenness, closeness, distance matrix).
3.  Export metrics to CSV files.
4.  Select sample controller nodes and evaluate the placement (Milestone 1).
5.  Run baseline placement strategies (Milestone 2).
6.  Compare strategies and generate report, CSV, plots, and summary.
"""

from __future__ import annotations

import logging

from config import (
    BASELINE_COMPARISON_CSV,
    BETWEENNESS_CSV,
    CLOSENESS_CSV,
    DEFAULT_CONTROLLERS,
    DEFAULT_NUM_CONTROLLERS,
    DEGREE_CSV,
    DISTANCE_MATRIX_CSV,
    EVALUATION_RESULTS_CSV,
    HYBRID_ANALYSIS_TXT,
    HYBRID_CORRELATIONS_CSV,
    HYBRID_SCORES_CSV,
    HYBRID_TOP_NODES_PNG,
    KCENTER_ANALYSIS_TXT,
    KCENTER_COVERAGE_CSV,
    KCENTER_COVERAGE_PNG,
    KCENTER_VS_CENTRALITY_TXT,
    OUTPUT_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    STRATEGY_SUMMARY_TXT,
    TOPOLOGY_FILE,
)
from comparison import StrategyComparison
from evaluator import PlacementEvaluator
from kcenter_analysis import KCenterAnalysis
from metrics import GraphMetrics
from strategies import HybridScorer, PlacementStrategies
from topology import TopologyLoader
from utils import ensure_directory, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the full analysis pipeline."""

    # ── Step 0: Bootstrap ────────────────────
    setup_logging()
    ensure_directory(OUTPUT_DIR)
    logger.info("SDN Controller Placement framework started.")

    # ── Step 1: Load topology ────────────────
    loader = TopologyLoader(TOPOLOGY_FILE)
    graph = loader.load_topology()

    # ── Step 2: Compute metrics ──────────────
    metrics = GraphMetrics(graph)
    metrics.compute_all()

    # ── Step 3: Export metrics to CSV ────────
    metrics.export_all(
        degree_path=DEGREE_CSV,
        betweenness_path=BETWEENNESS_CSV,
        closeness_path=CLOSENESS_CSV,
        distance_matrix_path=DISTANCE_MATRIX_CSV,
    )

    # ── Step 4: Evaluate sample placement ────
    controller_nodes: list[int] = list(DEFAULT_CONTROLLERS)
    logger.info("Selected controller nodes: %s", controller_nodes)

    evaluator = PlacementEvaluator(
        graph=graph,
        metrics=metrics,
        controller_nodes=controller_nodes,
    )
    evaluator.evaluate()
    evaluator.print_report()
    evaluator.save_results(EVALUATION_RESULTS_CSV)

    # ══════════════════════════════════════════
    # Milestone 2 – Baseline Strategy Comparison
    # ══════════════════════════════════════════

    # ── Step 5: Generate placements ─────────
    k = DEFAULT_NUM_CONTROLLERS
    strat = PlacementStrategies(graph)
    placements = strat.all_strategies(k, seed=RANDOM_SEED)
    logger.info("Generated placements for %d strategies (k=%d).", len(placements), k)

    # ── Step 6: Compare strategies ───────────
    comparison = StrategyComparison(
        graph=graph,
        distance_matrix=metrics.distance_matrix,  # type: ignore[arg-type]
        betweenness_scores=metrics.betweenness,   # type: ignore[arg-type]
    )
    comparison.evaluate_strategies(placements)

    # ── Step 7: Reports & artefacts ──────────
    comparison.print_report()
    comparison.export_csv(BASELINE_COMPARISON_CSV)
    comparison.export_summary(STRATEGY_SUMMARY_TXT)
    comparison.generate_plots(PLOTS_DIR)

    # ══════════════════════════════════════════
    # Milestone 2.5 – Hybrid Analysis
    # ══════════════════════════════════════════

    # ── Step 8: Hybrid score artefacts ───────
    scorer = HybridScorer(strat)
    scorer.export_scores_csv(HYBRID_SCORES_CSV)
    scorer.export_correlations(HYBRID_CORRELATIONS_CSV)
    scorer.generate_top_nodes_plot(HYBRID_TOP_NODES_PNG)
    scorer.export_analysis(HYBRID_ANALYSIS_TXT)

    # ═══════════════════════════════════════════
    # Milestone 3 – K-Center Analysis
    # ═══════════════════════════════════════════

    # ── Step 9: K-Center detailed analysis ────
    kcenter_controllers = placements["KCenter"]
    kcenter = KCenterAnalysis(
        graph=graph,
        distance_matrix=metrics.distance_matrix,  # type: ignore[arg-type]
        controllers=kcenter_controllers,
        nodes=metrics.nodes,
    )
    kcenter.export_analysis(KCENTER_ANALYSIS_TXT)
    kcenter.export_coverage_csv(KCENTER_COVERAGE_CSV)
    kcenter.generate_coverage_plot(KCENTER_COVERAGE_PNG)

    # ── Step 10: K-Center vs centrality report ─
    kcenter.export_vs_centrality_analysis(
        KCENTER_VS_CENTRALITY_TXT,
        comparison_results=comparison.results,
    )

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
