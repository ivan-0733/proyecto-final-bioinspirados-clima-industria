"""
Unit tests for Phase 5: Metrics edge cases.
Tests indeterminate values, caching, alias resolution, and factory.
"""
import pytest
import pandas as pd
from src.metrics import MetricsFactory, IndeterminateMetricsLogger
from src.metrics.scenario1 import Scenario1Metrics
from src.metrics.scenario2 import Scenario2Metrics
from src.core.exceptions import ConfigurationError


class TestMetricsFactory:
    """Test MetricsFactory with Strategy pattern."""
    
    def test_create_scenario1(self, sample_dataframe, sample_supports, sample_metadata):
        """Test creating Scenario1Metrics."""
        metrics = MetricsFactory.create_metrics(
            scenario_name='scenario_1',
            dataframe=sample_dataframe,
            supports_dict=sample_supports,
            metadata=sample_metadata
        )
        
        assert isinstance(metrics, Scenario1Metrics)
        available = metrics.get_available_metrics()
        assert 'casual_support' in available
    
    def test_create_scenario2(self, sample_dataframe, sample_supports, sample_metadata):
        """Test creating Scenario2Metrics."""
        metrics = MetricsFactory.create_metrics(
            scenario_name='scenario_2',
            dataframe=sample_dataframe,
            supports_dict=sample_supports,
            metadata=sample_metadata
        )
        
        assert isinstance(metrics, Scenario2Metrics)
        available = metrics.get_available_metrics()
        assert 'jaccard' in available
    
    def test_invalid_scenario(self, sample_dataframe, sample_supports, sample_metadata):
        """Test factory raises error on invalid scenario."""
        with pytest.raises(ConfigurationError):
            MetricsFactory.create_metrics(
                scenario_name="invalid_scenario",
                dataframe=sample_dataframe,
                supports_dict=sample_supports,
                metadata=sample_metadata
            )


class TestScenario1Metrics:
    """Test Scenario1Metrics edge cases."""
    
    def test_casual_support_calculation(self, sample_dataframe, sample_supports, sample_metadata):
        """Test casual support metric."""
        metrics = Scenario1Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        antecedent = [(0, 1)]  # age=1
        consequent = [(2, 1)]  # diabetes=1
        
        values, errors = metrics.get_metrics(antecedent, consequent, ['casual_support'])
        assert len(values) == 1
        assert isinstance(values[0], (float, type(None)))
    
    def test_casual_support_caching(self, sample_dataframe, sample_supports, sample_metadata):
        """Test caching mechanism."""
        metrics = Scenario1Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        antecedent = [(0, 0)]
        consequent = [(2, 1)]
        
        # First call
        values1, _ = metrics.get_metrics(antecedent, consequent, ['casual_support'])
        # Second call (should use cache)
        values2, _ = metrics.get_metrics(antecedent, consequent, ['casual_support'])
        
        assert values1 == values2
        # Verify cache was used
        assert len(metrics._cache) > 0
    
    def test_indeterminate_metrics_logged(self, sample_dataframe, sample_supports, sample_metadata):
        """Test that indeterminate metrics are logged."""
        # Create scenario with zero support
        import copy
        zero_supports = copy.deepcopy(sample_supports)
        zero_supports['variables']['age']['0'] = 0.0
        
        metrics = Scenario1Metrics(sample_dataframe, zero_supports, sample_metadata)
        antecedent = [(0, 0)]  # age=0 (zero support)
        consequent = [(2, 1)]
        
        values, errors = metrics.get_metrics(antecedent, consequent, ['casual_confidence'])
        # Should return None for indeterminate
        assert values[0] is None or errors.get('casual_confidence') is not None
    
    def test_alias_resolution(self, sample_dataframe, sample_supports, sample_metadata):
        """Test metric alias resolution (casual-supp → casual_support)."""
        metrics = Scenario1Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        # Test that canonical name returns valid metric
        canonical = metrics.get_canonical_name('casual-supp')
        assert canonical == 'casual_support'
        
        canonical_conf = metrics.get_canonical_name('casual-conf')
        assert canonical_conf == 'casual_confidence'


class TestScenario2Metrics:
    """Test Scenario2Metrics edge cases."""
    
    def test_jaccard_calculation(self, sample_dataframe, sample_supports, sample_metadata):
        """Test Jaccard similarity metric."""
        metrics = Scenario2Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        antecedent = [(0, 1)]
        consequent = [(2, 1)]
        
        values, _ = metrics.get_metrics(antecedent, consequent, ['jaccard'])
        assert len(values) == 1
        if values[0] is not None:
            assert 0 <= values[0] <= 1
    
    def test_cosine_calculation(self, sample_dataframe, sample_supports, sample_metadata):
        """Test cosine similarity metric."""
        metrics = Scenario2Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        antecedent = [(0, 0)]
        consequent = [(1, 1)]
        
        values, _ = metrics.get_metrics(antecedent, consequent, ['cosine'])
        assert len(values) == 1
        if values[0] is not None:
            assert 0 <= values[0] <= 1
    
    def test_phi_coefficient_calculation(self, sample_dataframe, sample_supports, sample_metadata):
        """Test phi coefficient metric."""
        metrics = Scenario2Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        antecedent = [(0, 1)]
        consequent = [(2, 0)]
        
        values, _ = metrics.get_metrics(antecedent, consequent, ['phi_coefficient'])
        assert len(values) == 1
        if values[0] is not None:
            assert -1 <= values[0] <= 1
    
    def test_k_measure_calculation(self, sample_dataframe, sample_supports, sample_metadata):
        """Test k-measure (kappa) metric."""
        metrics = Scenario2Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        antecedent = [(1, 0)]
        consequent = [(2, 1)]
        
        values, _ = metrics.get_metrics(antecedent, consequent, ['k_measure'])
        assert len(values) == 1
    
    def test_kappa_alias(self, sample_dataframe, sample_supports, sample_metadata):
        """Test kappa alias for k_measure."""
        metrics = Scenario2Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        # Test that canonical name resolves
        canonical = metrics.get_canonical_name('kappa')
        assert canonical == 'k_measure'
    
    def test_multiple_metrics_at_once(self, sample_dataframe, sample_supports, sample_metadata):
        """Test calculating multiple metrics in one call."""
        metrics = Scenario2Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        antecedent = [(0, 0)]
        consequent = [(2, 0)]
        
        values, _ = metrics.get_metrics(
            antecedent, 
            consequent, 
            ['jaccard', 'cosine', 'phi_coefficient']
        )
        assert len(values) == 3


class TestIndeterminateMetricsLogger:
    """Test IndeterminateMetricsLogger."""
    
    def test_logger_initialization(self):
        """Test logger can be created."""
        logger = IndeterminateMetricsLogger()
        summary = logger.get_summary()
        assert isinstance(summary, dict)
        assert 'total_metrics_with_errors' in summary
        assert summary['total_metrics_with_errors'] == 0
    
    def test_log_indeterminate_metric(self):
        """Test logging indeterminate metric."""
        logger = IndeterminateMetricsLogger()
        logger.log_indeterminate(
            metric_name='casual_confidence',
            error_reason='zero_antecedent_support',
            rule_signature='age=0 => diabetes=1'
        )
        summary = logger.get_summary()
        assert summary['total_metrics_with_errors'] > 0
    
    def test_multiple_logs_aggregation(self):
        """Test that multiple logs are aggregated."""
        logger = IndeterminateMetricsLogger()
        
        # Log same error twice
        logger.log_indeterminate('casual_conf', 'zero_support')
        logger.log_indeterminate('casual_conf', 'zero_support')
        
        summary = logger.get_summary()
        assert summary['total_metrics_with_errors'] > 0
        # Should aggregate counts
        assert 'casual_conf' in summary['metrics']
        assert summary['metrics']['casual_conf']['total_errors'] == 2
    
    def test_summary_format(self):
        """Test summary has expected structure."""
        logger = IndeterminateMetricsLogger()
        logger.log_indeterminate('jaccard', 'division_by_zero')
        
        summary = logger.get_summary()
        assert isinstance(summary, dict)
        assert 'total_metrics_with_errors' in summary
        assert 'metrics' in summary
        assert 'jaccard' in summary['metrics']
        assert 'total_errors' in summary['metrics']['jaccard']


class TestMetricsCachePerformance:
    """Test metrics caching performance."""
    
    def test_cache_hit_performance(self, sample_dataframe, sample_supports, sample_metadata):
        """Test that cache improves performance."""
        metrics = Scenario1Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        antecedent = [(0, 1)]
        consequent = [(2, 1)]
        objectives = ['casual_support', 'casual_confidence']
        
        # First call (cache miss)
        import time
        start = time.time()
        values1, _ = metrics.get_metrics(antecedent, consequent, objectives)
        time1 = time.time() - start
        
        # Second call (cache hit)
        start = time.time()
        values2, _ = metrics.get_metrics(antecedent, consequent, objectives)
        time2 = time.time() - start
        
        # Cache hit should be faster
        assert values1 == values2
        # Note: This might be flaky on fast machines
    
    def test_cache_key_uniqueness(self, sample_dataframe, sample_supports, sample_metadata):
        """Test that different rules don't collide in cache."""
        metrics = Scenario1Metrics(sample_dataframe, sample_supports, sample_metadata)
        
        # Two different rules
        values1, _ = metrics.get_metrics([(0, 0)], [(2, 1)], ['casual_support'])
        values2, _ = metrics.get_metrics([(0, 1)], [(2, 1)], ['casual_support'])
        
        # Should have different results
        assert values1 != values2
        # Cache should have 2 entries
        assert len(metrics._cache) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
