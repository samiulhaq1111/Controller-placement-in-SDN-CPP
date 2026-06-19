"""
Graph metrics module for SDN Controller Placement framework.

Computes and exports:
    - Degree centrality (per node)
    - Betweenness centrality (per node)
    - Closeness centrality (per node)
    - All-pairs shortest-path distance matrix
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GraphMetrics:
    """Compute, cache, and export standard graph-theoretic metrics.

    Parameters
    ----------
    graph : nx.Graph
        A connected (or at least loaded) NetworkX undirected graph.

    Attributes
    ----------
    graph : nx.Graph
        The underlying graph.
    nodes : list
        Sorted list of node identifiers.
    degree : dict
        ``{node: degree}`` mapping.
    betweenness : dict
        ``{node: betweenness_centrality}`` mapping.
    closeness : dict
        ``{node: closeness_centrality}`` mapping.
    distance_matrix : np.ndarray
        2-D array of shortest-path hop distances (rows/cols indexed by
        position in ``self.nodes``).
    """

    def __init__(self, graph: nx.Graph) -> None:  # type: ignore[type-arg]
        self.graph: nx.Graph = graph  # type: ignore[type-arg]
        self.nodes: list = sorted(graph.nodes())
        self._node_index: dict = {n: i for i, n in enumerate(self.nodes)}

        # Lazily computed caches
        self.degree: dict | None = None
        self.betweenness: dict | None = None
        self.closeness: dict | None = None
        self.distance_matrix: np.ndarray | None = None

    # ──────────────────────────────────────────
    # Computation
    # ──────────────────────────────────────────

    def compute_all(self) -> None:
        """Run every metric computation in sequence."""
        logger.info("Computing all graph metrics …")
        self.compute_degree()
        self.compute_betweenness()
        self.compute_closeness()
        self.compute_distance_matrix()
        logger.info("All graph metrics computed successfully.")

    def compute_degree(self) -> dict:
        """Compute node degrees.

        Returns
        -------
        dict
            ``{node: degree}`` mapping.
        """
        logger.info("Computing node degrees …")
        self.degree = dict(self.graph.degree())
        logger.debug("Degrees computed for %d nodes.", len(self.degree))
        return self.degree

    def compute_betweenness(self) -> dict:
        """Compute betweenness centrality for every node.

        Returns
        -------
        dict
            ``{node: betweenness_centrality}`` mapping.
        """
        logger.info("Computing betweenness centrality …")
        self.betweenness = nx.betweenness_centrality(self.graph)
        logger.debug(
            "Betweenness centrality computed for %d nodes.",
            len(self.betweenness),
        )
        return self.betweenness

    def compute_closeness(self) -> dict:
        """Compute closeness centrality for every node.

        Returns
        -------
        dict
            ``{node: closeness_centrality}`` mapping.
        """
        logger.info("Computing closeness centrality …")
        self.closeness = nx.closeness_centrality(self.graph)
        logger.debug(
            "Closeness centrality computed for %d nodes.", len(self.closeness)
        )
        return self.closeness

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute the all-pairs shortest-path distance matrix (hop count).

        Uses ``nx.all_pairs_shortest_path_length`` which performs BFS on
        unweighted graphs.

        Returns
        -------
        np.ndarray
            Square matrix where entry ``[i][j]`` is the shortest-path hop
            distance from ``self.nodes[i]`` to ``self.nodes[j]``.
        """
        logger.info("Computing all-pairs shortest-path distance matrix …")
        n = len(self.nodes)
        dist_matrix = np.full((n, n), fill_value=np.inf, dtype=float)

        path_lengths = dict(nx.all_pairs_shortest_path_length(self.graph))
        for src, targets in path_lengths.items():
            i = self._node_index[src]
            for dst, length in targets.items():
                j = self._node_index[dst]
                dist_matrix[i][j] = length

        self.distance_matrix = dist_matrix
        logger.info("Distance matrix computed (%d × %d).", n, n)
        return self.distance_matrix

    # ──────────────────────────────────────────
    # Accessors (single-node convenience)
    # ──────────────────────────────────────────

    def degree_of(self, node: int) -> int:
        """Return the degree of *node*."""
        if self.degree is None:
            self.compute_degree()
        return self.degree[node]  # type: ignore[index]

    def betweenness_of(self, node: int) -> float:
        """Return the betweenness centrality of *node*."""
        if self.betweenness is None:
            self.compute_betweenness()
        return self.betweenness[node]  # type: ignore[index]

    def closeness_of(self, node: int) -> float:
        """Return the closeness centrality of *node*."""
        if self.closeness is None:
            self.compute_closeness()
        return self.closeness[node]  # type: ignore[index]

    # ──────────────────────────────────────────
    # CSV export
    # ──────────────────────────────────────────

    def export_degree(self, filepath: Path | str) -> None:
        """Export node degrees to a CSV file.

        Columns: ``Node``, ``Degree``.
        """
        if self.degree is None:
            self.compute_degree()
        df = pd.DataFrame(
            sorted(self.degree.items()),  # type: ignore[union-attr]
            columns=["Node", "Degree"],
        )
        self._write_csv(df, filepath, "Degree")

    def export_betweenness(self, filepath: Path | str) -> None:
        """Export betweenness centrality to a CSV file.

        Columns: ``Node``, ``Betweenness``.
        """
        if self.betweenness is None:
            self.compute_betweenness()
        df = pd.DataFrame(
            sorted(self.betweenness.items()),  # type: ignore[union-attr]
            columns=["Node", "Betweenness"],
        )
        self._write_csv(df, filepath, "Betweenness centrality")

    def export_closeness(self, filepath: Path | str) -> None:
        """Export closeness centrality to a CSV file.

        Columns: ``Node``, ``Closeness``.
        """
        if self.closeness is None:
            self.compute_closeness()
        df = pd.DataFrame(
            sorted(self.closeness.items()),  # type: ignore[union-attr]
            columns=["Node", "Closeness"],
        )
        self._write_csv(df, filepath, "Closeness centrality")

    def export_distance_matrix(self, filepath: Path | str) -> None:
        """Export the distance matrix to a CSV file.

        Rows = source nodes, Columns = destination nodes.
        """
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        df = pd.DataFrame(
            self.distance_matrix,
            index=self.nodes,
            columns=self.nodes,
        )
        # Replace inf with a readable token for CSV
        df = df.replace(np.inf, -1)
        self._write_csv(df, filepath, "Distance matrix", include_index=True)

    def export_all(
        self,
        degree_path: Path | str,
        betweenness_path: Path | str,
        closeness_path: Path | str,
        distance_matrix_path: Path | str,
    ) -> None:
        """Convenience wrapper to export every metric at once."""
        self.export_degree(degree_path)
        self.export_betweenness(betweenness_path)
        self.export_closeness(closeness_path)
        self.export_distance_matrix(distance_matrix_path)
        logger.info("All metrics exported.")

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _write_csv(
        df: pd.DataFrame,
        filepath: Path | str,
        label: str,
        include_index: bool = False,
    ) -> None:
        """Write a DataFrame to *filepath* and log the action."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=include_index)
        logger.info("%s exported → %s", label, path)
