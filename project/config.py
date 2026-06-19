"""
Configuration module for SDN Controller Placement framework.

All file paths and project-wide constants are defined here to avoid
hardcoded values throughout the codebase.
"""

from pathlib import Path


# ──────────────────────────────────────────────
# Project root (parent of this file's directory)
# ──────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# Data directory
# ──────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
TOPOLOGY_FILE: Path = DATA_DIR / "Geant.gml"

# ──────────────────────────────────────────────
# Output directory
# ──────────────────────────────────────────────
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# Output CSV file names
DEGREE_CSV: Path = OUTPUT_DIR / "degree.csv"
BETWEENNESS_CSV: Path = OUTPUT_DIR / "betweenness.csv"
CLOSENESS_CSV: Path = OUTPUT_DIR / "closeness.csv"
DISTANCE_MATRIX_CSV: Path = OUTPUT_DIR / "distance_matrix.csv"
EVALUATION_RESULTS_CSV: Path = OUTPUT_DIR / "evaluation_results.csv"

# Baseline comparison outputs
BASELINE_COMPARISON_CSV: Path = OUTPUT_DIR / "baseline_comparison.csv"
STRATEGY_SUMMARY_TXT: Path = OUTPUT_DIR / "strategy_summary.txt"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"

# Hybrid strategy outputs
HYBRID_SCORES_CSV: Path = OUTPUT_DIR / "hybrid_scores.csv"
HYBRID_CORRELATIONS_CSV: Path = OUTPUT_DIR / "hybrid_correlations.csv"
HYBRID_ANALYSIS_TXT: Path = OUTPUT_DIR / "hybrid_analysis.txt"
HYBRID_TOP_NODES_PNG: Path = PLOTS_DIR / "hybrid_top_nodes.png"

# ──────────────────────────────────────────────
# Default controller placement (sample)
# ──────────────────────────────────────────────
DEFAULT_CONTROLLERS: list[int] = [5, 12, 28]
DEFAULT_NUM_CONTROLLERS: int = 3
RANDOM_SEED: int = 42

# ──────────────────────────────────────────────
# Hybrid placement weights
# ──────────────────────────────────────────────
HYBRID_DEGREE_WEIGHT: float = 0.4
HYBRID_BETWEENNESS_WEIGHT: float = 0.4
HYBRID_CLOSENESS_WEIGHT: float = 0.2

def validate_hybrid_weights() -> None:
    """Ensure hybrid weights sum to 1.0.

    Raises
    ------
    ValueError
        If the sum of weights deviates from 1.0 beyond floating-point tolerance.
    """
    total = HYBRID_DEGREE_WEIGHT + HYBRID_BETWEENNESS_WEIGHT + HYBRID_CLOSENESS_WEIGHT
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Hybrid weights must sum to 1.0, got {total:.6f} "
            f"(degree={HYBRID_DEGREE_WEIGHT}, betweenness={HYBRID_BETWEENNESS_WEIGHT}, "
            f"closeness={HYBRID_CLOSENESS_WEIGHT})."
        )

# ──────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = "INFO"
