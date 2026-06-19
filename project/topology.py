"""
Topology loader module for SDN Controller Placement framework.

Responsibilities:
    - Load a network topology from a GML file.
    - Validate the loaded graph.
    - Print summary statistics (nodes, edges, density, components).
    - Expose the underlying NetworkX graph for downstream modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


class TopologyLoader:
    """Loads, validates, and summarises a network topology stored in GML format.

    Parameters
    ----------
    filepath : Path | str
        Absolute or relative path to the ``.gml`` topology file.

    Attributes
    ----------
    graph : nx.Graph
        The loaded NetworkX undirected graph.
    filepath : Path
        Resolved path to the topology file.
    """

    def __init__(self, filepath: Path | str) -> None:
        self.filepath: Path = Path(filepath).resolve()
        self.graph: nx.Graph | None = None  # type: ignore[type-arg]

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def load_topology(self, filepath: Path | str | None = None) -> nx.Graph:
        """Read the GML file and return a NetworkX graph.

        Parameters
        ----------
        filepath : Path | str | None
            Optional override; falls back to the instance ``filepath``.

        Returns
        -------
        nx.Graph
            The loaded undirected graph.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file cannot be parsed as valid GML.
        """
        target = Path(filepath).resolve() if filepath else self.filepath

        if not target.exists():
            logger.error("Topology file not found: %s", target)
            raise FileNotFoundError(f"Topology file not found: {target}")

        logger.info("Loading topology from: %s", target)
        try:
            self.graph = nx.read_gml(str(target), label="id")
        except nx.NetworkXError as exc:
            if "duplicated" in str(exc).lower():
                logger.warning(
                    "GML contains duplicate edges. "
                    "Re-loading with deduplication."
                )
                self.graph = self._read_gml_dedup(str(target))
            else:
                logger.error("Failed to parse GML file: %s", exc)
                raise ValueError(f"Invalid GML file: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse GML file: %s", exc)
            raise ValueError(f"Invalid GML file: {exc}") from exc

        self._validate()
        self.print_topology_info()
        return self.graph

    def get_topology_info(self) -> dict[str, int | float]:
        """Return a dictionary of basic topology statistics.

        Returns
        -------
        dict
            Keys: ``nodes``, ``edges``, ``density``, ``connected_components``.

        Raises
        ------
        RuntimeError
            If the topology has not been loaded yet.
        """
        graph = self._require_graph()
        info: dict[str, int | float] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "connected_components": nx.number_connected_components(graph),
        }
        return info

    def print_topology_info(self) -> None:
        """Log and print a human-readable topology summary."""
        info = self.get_topology_info()
        separator = "=" * 50
        print(separator)
        print("TOPOLOGY SUMMARY")
        print(separator)
        print(f"  File               : {self.filepath.name}")
        print(f"  Nodes              : {info['nodes']}")
        print(f"  Edges              : {info['edges']}")
        print(f"  Density            : {info['density']:.6f}")
        print(f"  Connected components: {info['connected_components']}")
        print(separator)
        logger.info(
            "Topology loaded — %d nodes, %d edges, density=%.6f, components=%d",
            info["nodes"],
            info["edges"],
            info["density"],
            info["connected_components"],
        )

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _require_graph(self) -> nx.Graph:
        """Return the loaded graph or raise ``RuntimeError``."""
        if self.graph is None:
            raise RuntimeError(
                "Topology has not been loaded yet. Call load_topology() first."
            )
        return self.graph

    @staticmethod
    def _read_gml_dedup(filepath: str) -> nx.Graph:
        """Read a GML file while silently dropping duplicate edges.

        This is a fallback for GML files that contain duplicate edge
        entries which ``nx.read_gml`` rejects by default.

        Parameters
        ----------
        filepath : str
            Path to the GML file.

        Returns
        -------
        nx.Graph
            A graph with duplicate edges removed.
        """
        import re
        import tempfile

        with open(filepath, "r") as fh:
            lines = fh.readlines()

        # Track seen (source, target) pairs to filter duplicates
        seen_edges: set[tuple[int, int]] = set()
        filtered_lines: list[str] = []
        in_edge = False
        edge_block: list[str] = []
        source: int | None = None
        target: int | None = None

        for line in lines:
            stripped = line.strip()
            if stripped == "edge [":
                in_edge = True
                edge_block = [line]
                source = None
                target = None
                continue
            if in_edge:
                edge_block.append(line)
                src_match = re.match(r"\s*source\s+(\d+)", stripped)
                tgt_match = re.match(r"\s*target\s+(\d+)", stripped)
                if src_match:
                    source = int(src_match.group(1))
                if tgt_match:
                    target = int(tgt_match.group(1))
                if stripped == "]":
                    in_edge = False
                    if source is not None and target is not None:
                        edge_key = (min(source, target), max(source, target))
                        if edge_key in seen_edges:
                            logger.debug(
                                "Dropping duplicate edge: %d -- %d",
                                source, target,
                            )
                        else:
                            seen_edges.add(edge_key)
                            filtered_lines.extend(edge_block)
                    else:
                        filtered_lines.extend(edge_block)
            else:
                filtered_lines.append(line)

        # Write filtered content to a temp file and re-read
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gml", delete=False
        ) as tmp:
            tmp.writelines(filtered_lines)
            tmp_path = tmp.name

        try:
            graph = nx.read_gml(tmp_path, label="id")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        logger.info(
            "GML loaded with deduplication: %d nodes, %d edges.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    def _validate(self) -> None:
        """Perform basic sanity checks on the loaded graph.

        Raises
        ------
        ValueError
            If the graph is empty or contains no edges.
        """
        graph = self._require_graph()

        if graph.number_of_nodes() == 0:
            logger.error("Loaded topology contains no nodes.")
            raise ValueError("Topology contains no nodes.")

        if graph.number_of_edges() == 0:
            logger.error("Loaded topology contains no edges.")
            raise ValueError("Topology contains no edges.")

        if not nx.is_connected(graph):
            logger.warning(
                "Topology is NOT fully connected (%d components).",
                nx.number_connected_components(graph),
            )
        else:
            logger.info("Topology is connected.")

        logger.info("Topology validation passed.")
