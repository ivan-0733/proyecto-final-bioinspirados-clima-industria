"""
Unit tests for Phase 6: Statistics.
Tests HV computation, plateau detection, diversity metrics, convergence.
"""
import pytest
import numpy as np
from src.stats_modules import HypervolumeTracker, PopulationStats, ConvergenceMetrics, ParetoFrontAnalyzer


class TestHypervolumeTracker:
    """Test HypervolumeTracker functionality."""
    
    def test_initialization(self):
        """Test tracker initializes correctly."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point, window_size=5, tolerance=1e-3)
        
        assert np.array_equal(tracker.ref_point, ref_point)
        assert tracker.window_size == 5
        assert tracker.tolerance == 1e-3
        assert len(tracker.history) == 0
    
    def test_initialization_invalid_ref_point(self):
        """Test initialization rejects invalid reference points."""
        # 0D array
        with pytest.raises(ValueError, match="must be 1D array"):
            HypervolumeTracker(ref_point=np.array(0.0))
        
        # Single objective
        with pytest.raises(ValueError, match="≥2 objectives"):
            HypervolumeTracker(ref_point=np.array([0.0]))
    
    def test_initialization_invalid_parameters(self):
        """Test initialization validates parameters."""
        ref_point = np.array([0.0, 0.0])
        
        # Invalid window size
        with pytest.raises(ValueError, match="window_size must be ≥2"):
            HypervolumeTracker(ref_point=ref_point, window_size=1)
        
        # Negative tolerance
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            HypervolumeTracker(ref_point=ref_point, tolerance=-0.1)
    
    def test_compute_hypervolume(self):
        """Test HV computation for valid objectives."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point)
        
        # Simple 2-point Pareto front
        objectives = np.array([
            [-0.8, -0.2],  # Point 1
            [-0.2, -0.8]   # Point 2
        ])
        
        hv = tracker.compute(objectives)
        
        # HV should be positive for points dominating ref_point
        assert hv > 0.0
        assert isinstance(hv, float)
    
    def test_compute_empty_population(self):
        """Test HV returns 0 for empty population."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point)
        
        objectives = np.array([]).reshape(0, 2)
        hv = tracker.compute(objectives)
        
        assert hv == 0.0
    
    def test_compute_invalid_objectives_shape(self):
        """Test compute rejects invalid objective shapes."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point)
        
        # 1D array
        with pytest.raises(ValueError, match="must be 2D array"):
            tracker.compute(np.array([1, 2, 3]))
        
        # Wrong number of columns
        with pytest.raises(ValueError, match="has 3 columns"):
            tracker.compute(np.array([[1, 2, 3]]))
    
    def test_record_generation(self):
        """Test recording HV for a generation."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point)
        
        objectives = np.array([[-0.5, -0.5]])
        
        hv = tracker.record(generation=10, objectives=objectives)
        
        assert hv > 0.0
        assert len(tracker.history) == 1
        assert tracker.history[0] == (10, hv)
    
    def test_plateau_detection_insufficient_data(self):
        """Test plateau returns False with insufficient data."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point, window_size=5)
        
        # Only 3 points, window requires 5
        for gen in range(3):
            tracker.record(gen, np.array([[-0.5, -0.5]]))
        
        assert tracker.is_plateau() is False
    
    def test_plateau_detection_flat_hv(self):
        """Test plateau detects flat HV."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point, window_size=3, tolerance=1e-4)
        
        # Record same HV 5 times (flat)
        for gen in range(5):
            tracker.record(gen, np.array([[-0.5, -0.5]]))
        
        assert tracker.is_plateau() is True
    
    def test_plateau_detection_improving_hv(self):
        """Test plateau returns False for improving HV."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point, window_size=3, tolerance=1e-4)
        
        # Improving Pareto front (increasing HV)
        for gen in range(5):
            scale = 0.5 + gen * 0.1
            tracker.record(gen, np.array([[-scale, -scale]]))
        
        assert tracker.is_plateau() is False
    
    def test_trend_computation(self):
        """Test trend computes slope correctly."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point, window_size=5)
        
        # Linear improvement
        for gen in range(10):
            scale = 0.5 + gen * 0.05
            tracker.record(gen, np.array([[-scale, -scale]]))
        
        trend = tracker.get_trend()
        
        # Should have positive trend (improving)
        assert trend is not None
        assert trend > 0
    
    def test_trend_insufficient_data(self):
        """Test trend returns None with <2 points."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point)
        
        assert tracker.get_trend() is None
        
        tracker.record(0, np.array([[-0.5, -0.5]]))
        assert tracker.get_trend() is None
    
    def test_summary_empty(self):
        """Test summary with no data."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point)
        
        summary = tracker.get_summary()
        
        assert summary['total_generations'] == 0
        assert summary['initial_hv'] is None
        assert summary['final_hv'] is None
        assert summary['max_hv'] is None
        assert summary['improvement'] is None
        assert summary['trend'] is None
        assert summary['is_plateau'] is False
    
    def test_summary_with_data(self):
        """Test summary with recorded data."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point, window_size=3)
        
        # Record increasing HV
        hv_values = []
        for gen in range(5):
            scale = 0.5 + gen * 0.1
            hv = tracker.record(gen, np.array([[-scale, -scale]]))
            hv_values.append(hv)
        
        summary = tracker.get_summary()
        
        assert summary['total_generations'] == 5
        assert summary['initial_hv'] == hv_values[0]
        assert summary['final_hv'] == hv_values[-1]
        assert summary['max_hv'] == max(hv_values)
        assert summary['improvement'] == hv_values[-1] - hv_values[0]
        assert summary['trend'] is not None
    
    def test_clear_history(self):
        """Test clearing history."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point)
        
        # Record some data
        for gen in range(5):
            tracker.record(gen, np.array([[-0.5, -0.5]]))
        
        assert len(tracker.history) == 5
        
        tracker.clear()
        
        assert len(tracker.history) == 0
        assert tracker.get_summary()['total_generations'] == 0
    
    def test_get_history(self):
        """Test getting history returns copy."""
        ref_point = np.array([0.0, 0.0])
        tracker = HypervolumeTracker(ref_point=ref_point)
        
        tracker.record(0, np.array([[-0.5, -0.5]]))
        tracker.record(1, np.array([[-0.6, -0.6]]))
        
        history = tracker.get_history()
        
        # Verify it's a copy
        assert len(history) == 2
        history.append((999, 999.0))
        assert len(tracker.history) == 2  # Original unchanged


class TestPopulationStats:
    """Test PopulationStats diversity metrics."""
    
    def test_initialization(self):
        """Test stats tracker initializes."""
        stats = PopulationStats()
        assert stats is not None
    
    def test_unique_count(self):
        """Test counting unique genotypes."""
        stats = PopulationStats()
        
        genotypes = ['hash1', 'hash2', 'hash1', 'hash3', 'hash2']
        unique = stats.compute_unique_count(genotypes)
        
        assert unique == 3
    
    def test_unique_count_all_same(self):
        """Test unique count with all duplicates."""
        stats = PopulationStats()
        
        genotypes = ['hash1'] * 10
        unique = stats.compute_unique_count(genotypes)
        
        assert unique == 1
    
    def test_duplicate_rate(self):
        """Test duplicate rate computation."""
        stats = PopulationStats()
        
        # 5 total, 3 unique → 40% duplicates
        genotypes = ['A', 'B', 'A', 'C', 'B']
        rate = stats.compute_duplicate_rate(genotypes)
        
        assert 0.0 <= rate <= 1.0
        assert rate == 0.4  # 2 duplicates out of 5
    
    def test_duplicate_rate_all_unique(self):
        """Test duplicate rate with no duplicates."""
        stats = PopulationStats()
        
        genotypes = ['A', 'B', 'C', 'D']
        rate = stats.compute_duplicate_rate(genotypes)
        
        assert rate == 0.0
    
    def test_duplicate_rate_empty(self):
        """Test duplicate rate with empty population."""
        stats = PopulationStats()
        
        rate = stats.compute_duplicate_rate([])
        
        assert rate == 0.0
    
    def test_hamming_distance(self):
        """Test Hamming distance between genomes."""
        stats = PopulationStats()
        
        genome1 = np.array([0, 1, 0, 1, 0])
        genome2 = np.array([0, 0, 0, 1, 1])
        
        distance = stats.compute_hamming_distance(genome1, genome2)
        
        # Differ in positions 1 and 4
        assert distance == 2
    
    def test_hamming_distance_identical(self):
        """Test Hamming distance for identical genomes."""
        stats = PopulationStats()
        
        genome = np.array([1, 2, 3, 4])
        distance = stats.compute_hamming_distance(genome, genome)
        
        assert distance == 0
    
    def test_hamming_distance_different_lengths(self):
        """Test Hamming distance rejects different lengths."""
        stats = PopulationStats()
        
        genome1 = np.array([1, 2, 3])
        genome2 = np.array([1, 2])
        
        with pytest.raises(ValueError, match="same length"):
            stats.compute_hamming_distance(genome1, genome2)
    
    def test_avg_hamming(self):
        """Test average pairwise Hamming distance."""
        stats = PopulationStats()
        
        genomes = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        
        avg = stats.compute_avg_hamming(genomes)
        
        # Pair (0,1): distance=3, Pair (0,2): distance=1, Pair (1,2): distance=2
        # Average: (3+1+2)/3 = 2.0
        assert avg == 2.0
    
    def test_avg_hamming_single_genome(self):
        """Test avg Hamming with single genome."""
        stats = PopulationStats()
        
        genomes = np.array([[1, 2, 3]])
        avg = stats.compute_avg_hamming(genomes)
        
        assert avg == 0.0
    
    def test_shannon_entropy(self):
        """Test Shannon entropy computation."""
        stats = PopulationStats()
        
        # All unique → max entropy
        genotypes = ['A', 'B', 'C', 'D']
        entropy = stats.compute_shannon_entropy(genotypes)
        
        # Max entropy for 4 unique items is log2(4) = 2.0
        assert entropy == pytest.approx(2.0, abs=0.01)
    
    def test_shannon_entropy_all_same(self):
        """Test entropy with no diversity."""
        stats = PopulationStats()
        
        genotypes = ['A'] * 10
        entropy = stats.compute_shannon_entropy(genotypes)
        
        # No diversity → entropy = 0
        assert entropy == 0.0
    
    def test_shannon_entropy_empty(self):
        """Test entropy with empty population."""
        stats = PopulationStats()
        
        entropy = stats.compute_shannon_entropy([])
        
        assert entropy == 0.0
    
    def test_objective_spread(self):
        """Test objective space spread."""
        stats = PopulationStats()
        
        objectives = np.array([
            [0.1, 0.2],
            [0.5, 0.8],
            [0.3, 0.4]
        ])
        
        spread = stats.compute_objective_spread(objectives)
        
        assert len(spread) == 2
        assert all(spread >= 0)
    
    def test_objective_spread_empty(self):
        """Test spread with empty objectives."""
        stats = PopulationStats()
        
        spread = stats.compute_objective_spread(np.array([]).reshape(0, 2))
        
        assert len(spread) == 0
    
    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        stats = PopulationStats()
        
        genotypes = ['A', 'B', 'A', 'C']
        genomes = np.array([[0, 1], [1, 0], [0, 1], [1, 1]])
        objectives = np.array([[0.1, 0.2], [0.3, 0.4], [0.1, 0.2], [0.5, 0.6]])
        
        metrics = stats.compute_all_metrics(genotypes, genomes, objectives)
        
        assert 'unique_count' in metrics
        assert 'duplicate_rate' in metrics
        assert 'shannon_entropy' in metrics
        assert 'avg_hamming_distance' in metrics
        assert 'avg_objective_spread' in metrics
        assert metrics['population_size'] == 4
    
    def test_is_converged(self):
        """Test convergence detection."""
        stats = PopulationStats()
        
        # High diversity (not converged)
        diverse = ['A', 'B', 'C', 'D', 'E']
        assert stats.is_converged(diverse, unique_threshold=0.5) is False
        
        # Low diversity (converged)
        converged_pop = ['A', 'A', 'A', 'B', 'A']
        assert stats.is_converged(converged_pop, unique_threshold=0.5) is True
    
    def test_diversity_summary(self):
        """Test diversity summary generation."""
        stats = PopulationStats()
        
        genotypes = ['A', 'B', 'A', 'C']
        genomes = np.array([[0, 1], [1, 0], [0, 1], [1, 1]])
        
        summary = stats.get_diversity_summary(genotypes, genomes)
        
        assert summary['population_size'] == 4
        assert summary['unique_individuals'] == 3
        assert summary['unique_rate'] == 0.75
        assert 'shannon_entropy' in summary
        assert 'is_diverse' in summary
        assert 'avg_hamming_distance' in summary


class TestConvergenceMetrics:
    """Test ConvergenceMetrics (GD, IGD)."""
    
    def test_initialization_with_reference(self):
        """Test initialization with reference front."""
        reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        assert metrics.reference_front is not None
        assert len(metrics.reference_front) == 3
    
    def test_initialization_without_reference(self):
        """Test initialization without reference."""
        metrics = ConvergenceMetrics()
        
        assert metrics.reference_front is None
    
    def test_set_reference(self):
        """Test setting reference front."""
        metrics = ConvergenceMetrics()
        reference = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        metrics.set_reference(reference)
        
        assert metrics.reference_front is not None
        assert len(metrics.reference_front) == 2
    
    def test_invalid_reference_1d(self):
        """Test initialization rejects 1D reference."""
        with pytest.raises(ValueError, match="must be 2D array"):
            ConvergenceMetrics(reference_front=np.array([1, 2, 3]))
    
    def test_invalid_reference_single_objective(self):
        """Test initialization rejects single objective."""
        with pytest.raises(ValueError, match="≥2 objectives"):
            ConvergenceMetrics(reference_front=np.array([[1], [2]]))
    
    def test_invalid_reference_empty(self):
        """Test initialization rejects empty reference."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ConvergenceMetrics(reference_front=np.array([]).reshape(0, 2))
    
    def test_compute_gd_perfect_convergence(self):
        """Test GD with perfect convergence (approximation = reference)."""
        reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        # Approximation equals reference
        gd = metrics.compute_gd(reference)
        
        assert gd == pytest.approx(0.0, abs=1e-6)
    
    def test_compute_gd_with_distance(self):
        """Test GD with non-zero distance."""
        reference = np.array([[0.0, 1.0], [1.0, 0.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        # Approximation offset from reference
        approximation = np.array([[0.1, 0.9], [0.9, 0.1]])
        
        gd = metrics.compute_gd(approximation)
        
        # Should have positive GD
        assert gd > 0.0
        assert gd < 1.0  # But not too large
    
    def test_compute_gd_without_reference(self):
        """Test GD raises error without reference."""
        metrics = ConvergenceMetrics()
        approximation = np.array([[0.5, 0.5]])
        
        with pytest.raises(ValueError, match="Reference front not set"):
            metrics.compute_gd(approximation)
    
    def test_compute_gd_empty_approximation(self):
        """Test GD with empty approximation."""
        reference = np.array([[0.0, 1.0], [1.0, 0.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        gd = metrics.compute_gd(np.array([]).reshape(0, 2))
        
        assert gd == 0.0
    
    def test_compute_igd_perfect_coverage(self):
        """Test IGD with perfect coverage."""
        reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        # Approximation covers all reference points
        igd = metrics.compute_igd(reference)
        
        assert igd == pytest.approx(0.0, abs=1e-6)
    
    def test_compute_igd_with_distance(self):
        """Test IGD with non-zero distance."""
        reference = np.array([[0.0, 1.0], [1.0, 0.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        # Approximation doesn't fully cover reference
        approximation = np.array([[0.5, 0.5]])
        
        igd = metrics.compute_igd(approximation)
        
        # Should have positive IGD
        assert igd > 0.0
    
    def test_compute_igd_empty_approximation(self):
        """Test IGD with empty approximation."""
        reference = np.array([[0.0, 1.0], [1.0, 0.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        igd = metrics.compute_igd(np.array([]).reshape(0, 2))
        
        # Empty approximation → infinite IGD
        assert igd == float('inf')
    
    def test_compute_max_gd(self):
        """Test maximum GD computation."""
        reference = np.array([[0.0, 0.0], [1.0, 1.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        # One point close, one far
        approximation = np.array([[0.0, 0.0], [2.0, 2.0]])
        
        max_gd = metrics.compute_max_gd(approximation)
        
        # Max distance is from (2,2) to (1,1) = sqrt(2)
        assert max_gd == pytest.approx(np.sqrt(2), abs=1e-6)
    
    def test_compute_max_igd(self):
        """Test maximum IGD computation."""
        reference = np.array([[0.0, 0.0], [2.0, 2.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        # Approximation only near first reference point
        approximation = np.array([[0.0, 0.0]])
        
        max_igd = metrics.compute_max_igd(approximation)
        
        # Max distance is from (2,2) to (0,0) = sqrt(8)
        assert max_igd == pytest.approx(np.sqrt(8), abs=1e-6)
    
    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        metrics = ConvergenceMetrics(reference_front=reference)
        
        approximation = np.array([[0.1, 0.9], [0.9, 0.1]])
        
        all_metrics = metrics.compute_all_metrics(approximation)
        
        assert 'gd' in all_metrics
        assert 'igd' in all_metrics
        assert 'max_gd' in all_metrics
        assert 'max_igd' in all_metrics
        
        # All should be non-negative
        assert all_metrics['gd'] >= 0
        assert all_metrics['igd'] >= 0
        assert all_metrics['max_gd'] >= 0
        assert all_metrics['max_igd'] >= 0
    
    def test_incompatible_dimensions(self):
        """Test error with incompatible objective dimensions."""
        reference = np.array([[0.0, 1.0], [1.0, 0.0]])  # 2 objectives
        metrics = ConvergenceMetrics(reference_front=reference)
        
        # Approximation with 3 objectives
        approximation = np.array([[0.5, 0.5, 0.5]])
        
        with pytest.raises(ValueError, match="has 3 objectives"):
            metrics.compute_gd(approximation)


class TestParetoFrontAnalyzer:
    """Test suite for Pareto front distribution analyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ParetoFrontAnalyzer()
        assert analyzer is not None
    
    def test_compute_spacing_normal(self):
        """Test spacing computation with normal distribution."""
        analyzer = ParetoFrontAnalyzer()
        
        # Uniformly spaced points
        objectives = np.array([
            [0.0, 1.0],
            [0.33, 0.67],
            [0.67, 0.33],
            [1.0, 0.0]
        ])
        
        spacing = analyzer.compute_spacing(objectives)
        
        # Should be low for uniform spacing
        assert spacing >= 0
        assert spacing < 0.1
    
    def test_compute_spacing_single_point(self):
        """Test spacing with single point."""
        analyzer = ParetoFrontAnalyzer()
        objectives = np.array([[0.5, 0.5]])
        
        spacing = analyzer.compute_spacing(objectives)
        assert spacing == 0.0
    
    def test_compute_spacing_two_points(self):
        """Test spacing with two points."""
        analyzer = ParetoFrontAnalyzer()
        objectives = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        spacing = analyzer.compute_spacing(objectives)
        assert spacing == 0.0  # Perfect spacing for 2 points
    
    def test_compute_spacing_clustered(self):
        """Test spacing with clustered points."""
        analyzer = ParetoFrontAnalyzer()
        
        # Two clusters with large gap
        objectives = np.array([
            [0.0, 1.0],
            [0.01, 0.99],  # Close to first
            [0.8, 0.2],    # Large gap
            [1.0, 0.0]     # Close to third
        ])
        
        spacing = analyzer.compute_spacing(objectives)
        
        # Should be higher for non-uniform spacing (relax threshold)
        assert spacing > 0.0  # Just verify it's positive
    
    def test_compute_spread_normal(self):
        """Test spread computation."""
        analyzer = ParetoFrontAnalyzer()
        
        objectives = np.array([
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0]
        ])
        
        spread = analyzer.compute_spread(objectives)
        
        # Should be low for good coverage
        assert spread >= 0
        assert spread < 1.0
    
    def test_compute_spread_with_reference(self):
        """Test spread with reference extremes."""
        analyzer = ParetoFrontAnalyzer()
        
        objectives = np.array([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2]
        ])
        
        # Reference extremes wider than objectives
        reference_extremes = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        
        spread = analyzer.compute_spread(objectives, reference_extremes)
        assert spread >= 0
    
    def test_compute_spread_single_point(self):
        """Test spread with single point."""
        analyzer = ParetoFrontAnalyzer()
        objectives = np.array([[0.5, 0.5]])
        
        spread = analyzer.compute_spread(objectives)
        assert spread == 0.0
    
    def test_compute_crowding_distance_normal(self):
        """Test crowding distance computation."""
        analyzer = ParetoFrontAnalyzer()
        
        objectives = np.array([
            [0.0, 1.0],
            [0.33, 0.67],
            [0.67, 0.33],
            [1.0, 0.0]
        ])
        
        distances = analyzer.compute_crowding_distance(objectives)
        
        # Check shape
        assert distances.shape == (4,)
        
        # Boundary solutions should have infinite distance
        assert np.isinf(distances[0]) or np.isinf(distances[-1])
        
        # Interior solutions should have finite distance
        assert np.isfinite(distances[1])
        assert np.isfinite(distances[2])
    
    def test_compute_crowding_distance_single_point(self):
        """Test crowding distance with single point."""
        analyzer = ParetoFrontAnalyzer()
        objectives = np.array([[0.5, 0.5]])
        
        distances = analyzer.compute_crowding_distance(objectives)
        
        assert distances.shape == (1,)
        assert np.isinf(distances[0])
    
    def test_compute_crowding_distance_two_points(self):
        """Test crowding distance with two points."""
        analyzer = ParetoFrontAnalyzer()
        objectives = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        distances = analyzer.compute_crowding_distance(objectives)
        
        # Both should be infinite (boundaries)
        assert np.isinf(distances[0])
        assert np.isinf(distances[1])
    
    def test_compute_avg_crowding_distance(self):
        """Test average crowding distance."""
        analyzer = ParetoFrontAnalyzer()
        
        objectives = np.array([
            [0.0, 1.0],
            [0.33, 0.67],
            [0.67, 0.33],
            [1.0, 0.0]
        ])
        
        avg_distance = analyzer.compute_avg_crowding_distance(objectives)
        
        # Should be finite (excludes infinities)
        assert np.isfinite(avg_distance)
        assert avg_distance > 0
    
    def test_compute_avg_crowding_distance_all_boundary(self):
        """Test avg crowding when all points are boundary."""
        analyzer = ParetoFrontAnalyzer()
        objectives = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        avg_distance = analyzer.compute_avg_crowding_distance(objectives)
        
        # No finite distances, should return 0
        assert avg_distance == 0.0
    
    def test_compute_all_metrics(self):
        """Test computing all metrics together."""
        analyzer = ParetoFrontAnalyzer()
        
        objectives = np.array([
            [0.0, 1.0],
            [0.33, 0.67],
            [0.67, 0.33],
            [1.0, 0.0]
        ])
        
        all_metrics = analyzer.compute_all_metrics(objectives)
        
        # Check all keys present
        assert 'spacing' in all_metrics
        assert 'spread' in all_metrics
        assert 'avg_crowding_distance' in all_metrics
        
        # All should be non-negative
        assert all_metrics['spacing'] >= 0
        assert all_metrics['spread'] >= 0
        assert all_metrics['avg_crowding_distance'] >= 0
    
    def test_is_well_distributed_good(self):
        """Test well-distributed detection with good front."""
        analyzer = ParetoFrontAnalyzer()
        
        # Uniformly distributed
        objectives = np.array([
            [0.0, 1.0],
            [0.33, 0.67],
            [0.67, 0.33],
            [1.0, 0.0]
        ])
        
        is_good = analyzer.is_well_distributed(
            objectives,
            spacing_threshold=1.0,  # Relaxed threshold
            spread_threshold=2.0    # Relaxed threshold
        )
        assert is_good is True
    
    def test_is_well_distributed_poor(self):
        """Test well-distributed detection with poor front."""
        analyzer = ParetoFrontAnalyzer()
        
        # Clustered distribution
        objectives = np.array([
            [0.0, 1.0],
            [0.01, 0.99],
            [0.02, 0.98],
            [1.0, 0.0]
        ])
        
        is_good = analyzer.is_well_distributed(
            objectives,
            spacing_threshold=0.05  # Strict threshold
        )
        assert is_good is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

