# SDN Controller Placement Research Framework

A modular, research-oriented Python framework for analysing the **Controller Placement Problem (CPP)** in Software Defined Networks.

---

## Project Architecture

```
project/
├── data/
│   └── Geant.gml                  # GEANT 2010 network topology (GML format)
├── output/                        # Auto-generated CSV artefacts
│   ├── degree.csv
│   ├── betweenness.csv
│   ├── closeness.csv
│   ├── distance_matrix.csv
│   └── evaluation_results.csv
├── config.py                      # Centralised paths & constants
├── topology.py                    # TopologyLoader – GML loading & validation
├── metrics.py                     # GraphMetrics – centrality & distance matrix
├── objectives.py                  # ControllerPlacementObjectives – 5 objectives
├── evaluator.py                   # PlacementEvaluator – reporting & CSV export
├── utils.py                       # Logging setup & directory helpers
├── main.py                        # Entry point – full pipeline
├── requirements.txt
└── README.md
```

### Module Responsibilities

| Module | Class | Purpose |
|---|---|---|
| `topology.py` | `TopologyLoader` | Load GML, validate graph, print statistics |
| `metrics.py` | `GraphMetrics` | Compute degree, betweenness, closeness, distance matrix |
| `objectives.py` | `ControllerPlacementObjectives` | Evaluate 5 placement objectives |
| `evaluator.py` | `PlacementEvaluator` | Orchestrate evaluation, generate report, save CSV |
| `config.py` | — | All file paths and project constants |
| `utils.py` | — | Logging configuration and directory helpers |

---

## Metric Definitions

### Degree
The number of edges incident to a node.  Higher degree indicates a more connected switch.

### Betweenness Centrality
For each node, the fraction of all shortest paths in the graph that pass through it.  Nodes with high betweenness act as critical bridges.

### Closeness Centrality
The reciprocal of the sum of shortest-path distances from a node to all other nodes.  Higher closeness means the node can reach the rest of the network more quickly.

### Distance Matrix
An *N × N* matrix where entry *(i, j)* is the shortest-path hop distance from node *i* to node *j*.  Computed via BFS (unweighted graph).

---

## Objective Function Definitions

| # | Objective | Description | Return Type |
|---|---|---|---|
| 1 | **Average Controller Latency** | Mean hop distance from every switch to its nearest controller | `float` |
| 2 | **Maximum Controller Latency** | Worst-case hop distance from any switch to its nearest controller | `float` |
| 3 | **Controller Load Distribution** | Number of switches assigned to each controller (nearest-controller rule) | `dict[int, int]` |
| 4 | **Load Variance** | Population variance of controller loads (lower = better balance) | `float` |
| 5 | **Controller Importance Score** | Sum of betweenness centrality values of the selected controller nodes | `float` |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
cd project/
python main.py
```

### Output

The script prints a formatted report to stdout:

```
==================================================
CONTROLLER PLACEMENT REPORT
==================================================

  Controllers        : [5, 12, 28]

  Average Latency    : 1.82
  Maximum Latency    : 4
  Load Variance      : 2.31
  Importance Score   : 0.547

  Load Distribution:
    Controller   5  →  12 switches
    Controller  12  →  10 switches
    Controller  28  →  14 switches

==================================================
```

All metrics and evaluation results are saved as CSV files in the `output/` directory.

---

## Configuration

All file paths and constants are centralised in `config.py`:

```python
TOPOLOGY_FILE = DATA_DIR / "Geant.gml"
DEFAULT_CONTROLLERS = [5, 12, 28]
```

Modify `DEFAULT_CONTROLLERS` to evaluate different placements without changing any other module.

---

## Future Integration

The architecture is designed for extensibility.  Future optimiser modules will live in:

```
project/
├── optimizer/
│   ├── nsga2.py                   # NSGA-II multi-objective optimisation
│   ├── genetic_algorithm.py       # Single/multi-objective GA
│   └── rl_controller_placement.py # Reinforcement Learning approach
```

Each optimiser will consume the existing `GraphMetrics` and `ControllerPlacementObjectives` classes without requiring modifications to them, following the **Open/Closed Principle**.

---

## Technology Stack

- **Python 3.10+**
- **NetworkX** – graph representation and algorithms
- **NumPy** – numerical operations and distance matrix
- **Pandas** – CSV export and tabular data
- **Matplotlib** – visualisation (future)
- **Logging** – structured logging throughout

---

## License

This project is intended for academic and research use.
