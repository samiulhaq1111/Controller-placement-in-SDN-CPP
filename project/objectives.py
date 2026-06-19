"""
Controller placement objective functions for SDN analysis.

Each objective quantifies a different aspect of a given controller placement:
    1. Average controller latency  – mean switch-to-nearest-controller distance
    2. Maximum controller latency  – worst-case switch-to-nearest-controller distance
    3. Controller load distribution – switches managed per controller
    4. Load variance               – variance of controller loads (balance metric)
    5. Controller importance score – sum of betweenness centrality of controllers
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class ControllerPlacementObjectives:
    """Evaluate a candidate controller placement against multiple objectives.

    Parameters
    ----------
    graph : nx.Graph
        The network topology graph.
    distance_matrix : np.ndarray
        Square matrix of shortest-path hop distances.  Row/column ordering
        must match ``sorted(graph.nodes())``.
    controller_nodes : list[int]
        Node identifiers selected as controller locations.

    Raises
    ------
    ValueError
        If any controller node is not present in the graph.
    """

    def __init__(
        self,
        graph: nx.Graph,  # type: ignore[type-arg]
        distance_matrix: np.ndarray,
        controller_nodes: list[int],
    ) -> None:
        self.graph = graph
        self.distance_matrix = distance_matrix
        self.controller_nodes = list(controller_nodes)

        self.nodes: list = sorted(graph.nodes())
        self._node_index: dict[Any, int] = {n: i for i, n in enumerate(self.nodes)}

        self._validate_controllers()

        # Pre-computed mapping: node → nearest controller & distance
        self._nearest: dict[Any, tuple[Any, float]] | None = None

    # ──────────────────────────────────────────
    # Objective 1 – Average Controller Latency
    # ──────────────────────────────────────────

    def average_controller_latency(self) -> float:
        """Mean hop distance from every switch to its nearest controller.

        Returns
        -------
        float
            Average latency (hop count).
        """
        nearest = self._compute_nearest_controller()
        distances = [dist for _, dist in nearest.values()]
        avg = float(np.mean(distances))
        logger.debug("Average controller latency: %.4f", avg)
        return avg

    # ──────────────────────────────────────────
    # Objective 2 – Maximum Controller Latency
    # ──────────────────────────────────────────

    def maximum_controller_latency(self) -> float:
        """Worst-case (maximum) hop distance from any switch to its nearest controller.

        Returns
        -------
        float
            Maximum latency (hop count).
        """
        nearest = self._compute_nearest_controller()
        distances = [dist for _, dist in nearest.values()]
        max_lat = float(np.max(distances))
        logger.debug("Maximum controller latency: %.4f", max_lat)
        return max_lat

    # ──────────────────────────────────────────
    # Objective 3 – Controller Load Distribution
    # ──────────────────────────────────────────

    def controller_load_distribution(self) -> dict[int, int]:
        """Count of switches managed by each controller (nearest-controller assignment).

        Returns
        -------
        dict[int, int]
            ``{controller_node: number_of_managed_switches}``.
        """
        nearest = self._compute_nearest_controller()
        load: dict[int, int] = {c: 0 for c in self.controller_nodes}
        for _, (ctrl, _) in nearest.items():
            load[ctrl] += 1
        logger.debug("Controller load distribution: %s", load)
        return load

    # ──────────────────────────────────────────
    # Objective 4 – Load Variance
    # ──────────────────────────────────────────

    def load_variance(self) -> float:
        """Variance of controller loads (lower = better balance).

        Returns
        -------
        float
            Population variance of switch counts across controllers.
        """
        load = self.controller_load_distribution()
        values = list(load.values())
        var = float(np.var(values))
        logger.debug("Load variance: %.4f", var)
        return var

    # ──────────────────────────────────────────
    # Objective 5 – Controller Importance Score
    # ──────────────────────────────────────────

    def controller_importance_score(
        self, betweenness: dict[int, float]
    ) -> float:
        """Sum of betweenness centrality values for the selected controllers.

        Parameters
        ----------
        betweenness : dict[int, float]
            Pre-computed betweenness centrality for every node in the graph.

        Returns
        -------
        float
            Aggregate importance score.
        """
        score = sum(
            betweenness.get(c, 0.0) for c in self.controller_nodes
        )
        logger.debug("Controller importance score: %.6f", score)
        return score

    # ──────────────────────────────────────────
    # Aggregate evaluation
    # ──────────────────────────────────────────

    def evaluate_all(
        self, betweenness: dict[int, float]
    ) -> dict[str, float | dict[int, int]]:
        """Compute all objectives and return a combined result dictionary.

        Returns
        -------
        dict
            Keys: ``average_latency``, ``maximum_latency``,
            ``load_distribution``, ``load_variance``,
            ``importance_score``.
        """
        return {
            "average_latency": self.average_controller_latency(),
            "maximum_latency": self.maximum_controller_latency(),
            "load_distribution": self.controller_load_distribution(),
            "load_variance": self.load_variance(),
            "importance_score": self.controller_importance_score(betweenness),
        }

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _validate_controllers(self) -> None:
        """Raise ``ValueError`` if any controller node is not in the graph."""
        graph_nodes = set(self.graph.nodes())
        invalid = [c for c in self.controller_nodes if c not in graph_nodes]
        if invalid:
            logger.error("Invalid controller node(s): %s", invalid)
            raise ValueError(
                f"The following controller nodes are not in the graph: {invalid}"
            )
        if not self.controller_nodes:
            logger.error("Controller node list is empty.")
            raise ValueError("At least one controller node must be specified.")
        logger.debug("Controller nodes validated: %s", self.controller_nodes)

    def _compute_nearest_controller(self) -> dict[Any, tuple[Any, float]]:
        """For every node, find the nearest controller and the distance.

        Returns
        -------
        dict
            ``{node: (nearest_controller, distance)}``
        """
        if self._nearest is not None:
            return self._nearest

        ctrl_indices = [self._node_index[c] for c in self.controller_nodes]
        nearest: dict[Any, tuple[Any, float]] = {}

        for node in self.nodes:
            i = self._node_index[node]
            best_ctrl: Any = None
            best_dist = float("inf")
            for c, ci in zip(self.controller_nodes, ctrl_indices):
                d = self.distance_matrix[i][ci]
                if d < best_dist:
                    best_dist = d
                    best_ctrl = c
            nearest[node] = (best_ctrl, best_dist)

        self._nearest = nearest
        return self._nearest
