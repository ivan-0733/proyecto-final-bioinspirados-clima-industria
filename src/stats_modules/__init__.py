"""
Statistics module for MOEA/D ARM.

Provides statistical analysis tools for tracking evolution:
- HypervolumeTracker: HV evolution over generations
- PopulationStats: Diversity metrics (entropy, Hamming distance)
- ConvergenceMetrics: GD, IGD vs reference front
- ParetoFrontAnalyzer: Spacing, spread, crowding distance
"""
from src.stats_modules.hypervolume_tracker import HypervolumeTracker
from src.stats_modules.population_stats import PopulationStats
from src.stats_modules.convergence_metrics import ConvergenceMetrics
from src.stats_modules.pareto_front_analyzer import ParetoFrontAnalyzer

__all__ = [
    'HypervolumeTracker',
    'PopulationStats',
    'ConvergenceMetrics',
    'ParetoFrontAnalyzer',
]
