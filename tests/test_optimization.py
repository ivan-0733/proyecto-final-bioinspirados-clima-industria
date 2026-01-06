"""
Unit tests for Phase 3: Optimization components.
Tests AdaptiveControl, StuckDetector, and LazyParetoArchiver.
"""
import pytest
import numpy as np
from src.optimization import AdaptiveControl, ProbabilityConfig, StuckDetector, LazyParetoArchiver
from src.core.exceptions import MOEADDeadlockError
from src.representation import Rule
class TestProbabilityConfig:
    """Test ProbabilityConfig dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        config = ProbabilityConfig(initial=0.5, min_val=0.3, max_val=0.7)
        assert config.initial == 0.5
        assert config.min_val == 0.3
        assert config.max_val == 0.7
        assert config.increase_factor == 1.1
        assert config.decrease_factor == 0.9
    
    def test_custom_factors(self):
        """Test custom increase/decrease factors."""
        config = ProbabilityConfig(
            initial=0.6,
            min_val=0.4,
            max_val=0.8,
            increase_factor=1.2,
            decrease_factor=0.85
        )
        assert config.increase_factor == 1.2
        assert config.decrease_factor == 0.85


class TestAdaptiveControl:
    """Test AdaptiveControl with 1/5 success rule."""
    
    def test_initialization(self):
        """Test proper initialization."""
        mutation_config = ProbabilityConfig(initial=0.5, min_val=0.3, max_val=0.7)
        crossover_config = ProbabilityConfig(initial=0.6, min_val=0.4, max_val=0.8)
        
        control = AdaptiveControl(mutation_config, crossover_config)
        
        assert control.mut_prob == 0.5
        assert control.cx_prob == 0.6
        assert len(control.success_history) == 0
    
    def test_record_generation(self):
        """Test recording generation success."""
        mutation_config = ProbabilityConfig(initial=0.5, min_val=0.3, max_val=0.7)
        crossover_config = ProbabilityConfig(initial=0.6, min_val=0.4, max_val=0.8)
        
        control = AdaptiveControl(mutation_config, crossover_config)
        control.record_generation(num_success=10, num_attempts=20)
        
        assert len(control.success_history) == 1
        assert control.success_history[0] == 0.5  # 10/20
    
    def test_increase_probabilities_high_success(self):
        """Test probability increase when success_rate > 0.2."""
        mutation_config = ProbabilityConfig(initial=0.5, min_val=0.3, max_val=0.7)
        crossover_config = ProbabilityConfig(initial=0.6, min_val=0.4, max_val=0.8)
        
        control = AdaptiveControl(mutation_config, crossover_config)
        control.record_generation(num_success=6, num_attempts=10)  # 0.6 > 0.2
        
        new_mut, new_cx = control.update_probabilities()
        
        # Mutation should increase (0.5 * 1.1 = 0.55)
        assert new_mut > 0.5
        assert new_mut <= 0.7  # Capped at max
        
        # Crossover should decrease (0.6 * 0.9 = 0.54)
        assert new_cx < 0.6
        assert new_cx >= 0.4  # Floored at min
    
    def test_decrease_probabilities_low_success(self):
        """Test probability decrease when success_rate < 0.2."""
        mutation_config = ProbabilityConfig(initial=0.5, min_val=0.3, max_val=0.7)
        crossover_config = ProbabilityConfig(initial=0.6, min_val=0.4, max_val=0.8)
        
        control = AdaptiveControl(mutation_config, crossover_config)
        control.record_generation(num_success=1, num_attempts=10)  # 0.1 < 0.2
        
        new_mut, new_cx = control.update_probabilities()
        
        # Mutation should decrease (0.5 * 0.9 = 0.45)
        assert new_mut < 0.5
        assert new_mut >= 0.3
        
        # Crossover should increase (0.6 * 1.1 = 0.66)
        assert new_cx > 0.6
        assert new_cx <= 0.8
    
    def test_probability_bounds(self):
        """Test probabilities are clamped to min/max."""
        mutation_config = ProbabilityConfig(initial=0.69, min_val=0.3, max_val=0.7)
        crossover_config = ProbabilityConfig(initial=0.41, min_val=0.4, max_val=0.8)
        
        control = AdaptiveControl(mutation_config, crossover_config)
        control.record_generation(num_success=8, num_attempts=10)  # High success
        
        new_mut, new_cx = control.update_probabilities()
        
        # Mutation: 0.69 * 1.1 = 0.759, should be clamped to 0.7
        assert new_mut == 0.7
        
        # Crossover: 0.41 * 0.9 = 0.369, should be clamped to 0.4
        assert new_cx == 0.4


class TestStuckDetector:
    """Test StuckDetector with multi-modal detection."""
    
    def test_initialization(self):
        """Test proper initialization."""
        detector = StuckDetector(
            max_runtime_minutes=10.0,
            window_size=5,
            min_new_per_window=2,
            hv_tolerance=1e-6,
            hv_period=10
        )
        
        assert detector.max_runtime_minutes == 10.0
        assert detector.window_size == 5
        assert detector.min_new_per_window == 2
    
    def test_runtime_limit_detection(self):
        """Test stuck detection via runtime limit."""
        detector = StuckDetector(max_runtime_minutes=0.001)  # 0.06 seconds
        
        import time
        time.sleep(0.1)  # Exceed limit
        
        detector.record_generation(num_new=5, hypervolume=0.5)
        
        is_stuck, reason = detector.check_stuck(generation=1)
        assert is_stuck
        assert reason is not None
        assert "runtime" in reason.lower() or "limit" in reason.lower()
    
    def test_window_based_detection(self):
        """Test stuck detection via new rules window."""
        detector = StuckDetector(
            max_runtime_minutes=None,
            window_size=2,  # Smaller window
            min_new_per_window=10  # Higher threshold
        )
        
        # Record generations with few new rules (need 3 iterations for stuck_streak >= 3)
        is_stuck = False
        reason = None
        for i in range(4):  # Need 4 iterations to reach stuck_streak=3
            detector.record_generation(num_new=1, hypervolume=None)
            is_stuck, reason = detector.check_stuck(generation=i+1)
        
        # After stuck_streak >= 3, should be stuck
        assert is_stuck
        assert reason is not None
        assert "new rules" in reason.lower()
    
    def test_hypervolume_stagnation_detection(self):
        """Test stuck detection via HV stagnation."""
        detector = StuckDetector(
            max_runtime_minutes=None,
            hv_tolerance=0.001,
            hv_period=3
        )
        
        # Record same HV for 3 generations
        detector.record_generation(num_new=10, hypervolume=0.5)
        detector.record_generation(num_new=10, hypervolume=0.5001)
        detector.record_generation(num_new=10, hypervolume=0.5002)
        
        is_stuck, reason = detector.check_stuck(generation=3)
        assert is_stuck
        assert reason is not None
        assert "hypervolume" in reason.lower()
    
    def test_not_stuck(self):
        """Test not stuck when conditions not met."""
        detector = StuckDetector(
            max_runtime_minutes=10.0,
            window_size=5,
            min_new_per_window=2
        )
        
        # Record good progress
        for i in range(5):
            detector.record_generation(num_new=5, hypervolume=0.5 + i * 0.1)
        
        is_stuck, reason = detector.check_stuck(generation=5)
        assert not is_stuck
        assert reason is None
    
    def test_raise_if_stuck(self):
        """Test raise_if_stuck throws exception."""
        detector = StuckDetector(max_runtime_minutes=0.001)
        
        import time
        time.sleep(0.1)
        
        detector.record_generation(num_new=1, hypervolume=None)
        
        with pytest.raises(MOEADDeadlockError) as exc_info:
            detector.raise_if_stuck(generation=1)
        
        # Verify exception attributes
        assert exc_info.value.generation == 1
        assert exc_info.value.stuck_streak >= 0


class TestLazyParetoArchiver:
    """Test LazyParetoArchiver with deferred deduplication."""
    
    def test_initialization(self):
        """Test proper initialization."""
        archiver = LazyParetoArchiver()
        assert len(archiver.candidates) == 0
    
    def test_add_candidate(self):
        """Test adding candidates."""
        archiver = LazyParetoArchiver()
        
        X = np.array([1, 2, 3, 4])
        F = np.array([0.5, 0.6, 0.7])
        rule = Rule.from_items([(0, 1)], [(1, 0)])  # A->B
        
        archiver.add_candidate(X, F, rule)
        
        assert len(archiver.candidates) == 1
        assert archiver.candidates[0][2] == rule.hash
    
    def test_deduplicate_removes_duplicates(self):
        """Test deduplication removes duplicate hashes."""
        archiver = LazyParetoArchiver()
        
        # Add duplicate rules
        X1 = np.array([1, 2, 3, 4])
        F1 = np.array([0.5, 0.6, 0.7])
        rule1 = Rule.from_items([(0, 1)], [(1, 0)])  # A->B
        
        X2 = np.array([5, 6, 7, 8])
        F2 = np.array([0.6, 0.7, 0.8])
        rule2 = Rule.from_items([(2, 1)], [(3, 0)])  # C->D
        
        archiver.add_candidate(X1, F1, rule1)
        archiver.add_candidate(X2, F2, rule2)
        archiver.add_candidate(X1, F1, rule1)  # Duplicate
        
        num_removed = archiver.deduplicate()
        
        assert num_removed == 1
        X_arr, F_arr, hashes = archiver.get_archive()
        assert len(hashes) == 2  # Only 2 unique rules
    
    def test_get_archive(self):
        """Test getting deduplicated archive."""
        archiver = LazyParetoArchiver()
        
        X1 = np.array([1, 2, 3, 4])
        F1 = np.array([0.5, 0.6, 0.7])
        rule1 = Rule.from_items([(0, 1)], [(1, 0)])
        
        X2 = np.array([5, 6, 7, 8])
        F2 = np.array([0.6, 0.7, 0.8])
        rule2 = Rule.from_items([(2, 1)], [(3, 0)])
        
        archiver.add_candidate(X1, F1, rule1)
        archiver.add_candidate(X2, F2, rule2)
        
        X_arr, F_arr, hashes = archiver.get_archive()
        
        assert len(hashes) == 2
        assert np.array_equal(X_arr[0], X1)
        assert np.array_equal(X_arr[1], X2)
    
    def test_clear(self):
        """Test clearing archive."""
        archiver = LazyParetoArchiver()
        
        rule = Rule.from_items([(0, 1)], [(1, 0)])
        archiver.add_candidate(np.array([1, 2]), np.array([0.5]), rule)
        archiver.clear()
        
        assert len(archiver.candidates) == 0
    
    def test_get_statistics(self):
        """Test statistics reporting."""
        archiver = LazyParetoArchiver()
        
        rule1 = Rule.from_items([(0, 1)], [(1, 0)])
        rule2 = Rule.from_items([(2, 1)], [(3, 0)])
        
        archiver.add_candidate(np.array([1, 2]), np.array([0.5]), rule1)
        archiver.add_candidate(np.array([3, 4]), np.array([0.6]), rule2)
        archiver.add_candidate(np.array([1, 2]), np.array([0.5]), rule1)  # Duplicate
        
        archiver.deduplicate()
        stats = archiver.get_statistics()
        
        # After dedup, total_candidates == unique_candidates
        assert stats["total_candidates"] == 2
        assert stats["unique_candidates"] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
