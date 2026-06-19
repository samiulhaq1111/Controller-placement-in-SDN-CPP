"""
Placement evaluator module for SDN Controller Placement framework.

Orchestrates objective computation, builds a structured report, and
persists results to CSV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pandas as pd

from metrics import GraphMetrics
from objectives import ControllerPlacementObjectives

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Immutable container for a single placement evaluation.

    Attributes
    ----------
    controller_nodes : list[int]
        Selected controller locations.
    average_latency : float
        Mean switch-to-nearest-controller hop distance.
    maximum_latency : float
        Worst-case switch-to-nearest-controller hop distance.
    load_distribution : dict[int, int]
        Switches managed per controller.
    load_variance : float
        Variance of controller loads.
    importance_score : float
        Sum of betweenness centrality for the chosen controllers.
    """

    controller_nodes: list[int]
    average_latency: float
    maximum_latency: float
    load_distribution: dict[int, int] = field(default_factory=dict)
    load_variance: float = 0.0
    importance_score: float = 0.0


class PlacementEvaluator:
    """Evaluate a controller placement and generate a structured report.

    Parameters
    ----------
    graph : nx.Graph
        Network topology graph.
    metrics : GraphMetrics
        Pre-computed graph metrics (must include betweenness & distance matrix).
    controller_nodes : list[int]
        Node IDs selected as controllers.
    """

    def __init__(
        self,
        graph: nx.Graph,  # type: ignore[type-arg]
        metrics: GraphMetrics,
        controller_nodes: list[int],
    ) -> None:
        self.graph = graph
        self.metrics = metrics
        self.controller_nodes = list(controller_nodes)

        # Ensure required metrics exist
        if self.metrics.betweenness is None:
            self.metrics.compute_betweenness()
        if self.metrics.distance_matrix is None:
            self.metrics.compute_distance_matrix()

        self._objectives = ControllerPlacementObjectives(
            graph=graph,
            distance_matrix=self.metrics.distance_matrix,  # type: ignore[arg-type]
            controller_nodes=self.controller_nodes,
        )

        self.result: EvaluationResult | None = None

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def evaluate(self) -> EvaluationResult:
        """Run all objective functions and store the result.

        Returns
        -------
        EvaluationResult
            Dataclass containing all computed metrics.
        """
        logger.info(
            "Evaluating placement with controllers: %s", self.controller_nodes
        )
        raw = self._objectives.evaluate_all(
            betweenness=self.metrics.betweenness,  # type: ignore[arg-type]
        )

        self.result = EvaluationResult(
            controller_nodes=self.controller_nodes,
            average_latency=raw["average_latency"],  # type: ignore[arg-type]
            maximum_latency=raw["maximum_latency"],  # type: ignore[arg-type]
            load_distribution=raw["load_distribution"],  # type: ignore[arg-type]
            load_variance=raw["load_variance"],  # type: ignore[arg-type]
            importance_score=raw["importance_score"],  # type: ignore[arg-type]
        )
        logger.info("Evaluation complete.")
        return self.result

    def print_report(self) -> None:
        """Print a human-readable placement report to stdout."""
        result = self._require_result()
        separator = "=" * 50

        print()
        print(separator)
        print("CONTROLLER PLACEMENT REPORT")
        print(separator)
        print()
        print(f"  Controllers        : {result.controller_nodes}")
        print()
        print(f"  Average Latency    : {result.average_latency:.2f}")
        print(f"  Maximum Latency    : {result.maximum_latency:.0f}")
        print(f"  Load Variance      : {result.load_variance:.2f}")
        print(f"  Importance Score   : {result.importance_score:.3f}")
        print()
        print("  Load Distribution:")
        for ctrl, count in sorted(result.load_distribution.items()):
            print(f"    Controller {ctrl:>3}  →  {count} switches")
        print()
        print(separator)

    def save_results(self, filepath: Path | str) -> None:
        """Persist the evaluation result to a CSV file.

        Columns: ``Controller_Nodes``, ``Average_Latency``,
        ``Maximum_Latency``, ``Load_Variance``, ``Importance_Score``.

        Load distribution is serialised as a string representation.
        """
        result = self._require_result()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "Controller_Nodes": str(result.controller_nodes),
            "Average_Latency": round(result.average_latency, 4),
            "Maximum_Latency": result.maximum_latency,
            "Load_Variance": round(result.load_variance, 4),
            "Importance_Score": round(result.importance_score, 6),
            "Load_Distribution": str(result.load_distribution),
        }
        df = pd.DataFrame([row])
        df.to_csv(path, index=False)
        logger.info("Evaluation results saved → %s", path)

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _require_result(self) -> EvaluationResult:
        """Return the stored result or raise ``RuntimeError``."""
        if self.result is None:
            raise RuntimeError(
                "Evaluation has not been run yet. Call evaluate() first."
            )
        return self.result
