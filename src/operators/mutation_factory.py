"""
Factory for creating mutation operators based on config.
"""
from typing import Dict, Any

from src.core.logging_config import get_logger
from .mutation import ARMMutation
from .guided_mutation import GuidedMutation
from .conservative_mutation import ConservativeMutation
from .template_mutation import TemplateMutation
from .fallback_mutation import FallbackMutation


def create_mutation(
    mutation_type: str,
    metadata: Dict[str, Any],
    validator,
    logger,
    config: Dict[str, Any]
) -> Any:
    """
    Create mutation operator based on type.
    
    Args:
        mutation_type: One of:
            - "mixed" (default ARMMutation)
            - "guided" (recombine from pool)
            - "conservative" (minimal changes)
            - "template" (predefined patterns)
            - "fallback" (fast timeout + pool)
        metadata: Feature metadata
        validator: ARMValidator instance
        logger: Logger instance
        config: Mutation configuration dict
    
    Returns:
        Mutation operator instance
    """
    log = get_logger(__name__)
    
    prob = config.get('probability', {}).get('initial', 0.7)
    
    if mutation_type == "guided":
        log.info("creating_guided_mutation", prob=prob)
        return GuidedMutation(
            metadata=metadata,
            validator=validator,
            prob=prob
        )
    
    elif mutation_type == "conservative":
        log.info("creating_conservative_mutation", prob=prob)
        return ConservativeMutation(
            metadata=metadata,
            validator=validator,
            prob=prob
        )
    
    elif mutation_type == "template":
        log.info("creating_template_mutation", prob=prob)
        return TemplateMutation(
            metadata=metadata,
            validator=validator,
            prob=prob
        )
    
    elif mutation_type == "fallback":
        timeout = config.get('timeout', 2.0)
        max_operations = config.get('max_operations', 500)
        reproducible_mode = config.get('reproducible_mode', True)
        log.info(
            "creating_fallback_mutation",
            prob=prob,
            timeout=timeout,
            max_operations=max_operations,
            reproducible_mode=reproducible_mode
        )
        return FallbackMutation(
            metadata=metadata,
            validator=validator,
            prob=prob,
            timeout=timeout,
            max_operations=max_operations,
            reproducible_mode=reproducible_mode
        )
    
    else:  # "mixed" or default
        log.info("creating_mixed_mutation", prob=prob)
        return ARMMutation(
            metadata=metadata,
            validator=validator,
            logger=logger,
            prob=prob,
            active_ops=config.get('active_ops', ['extension', 'contraction', 'replacement']),
            repair_attempts=config.get('repair_attempts', 10),
            duplicate_attempts=config.get('duplicate_attempts', 15),
            timeout_per_attempt=config.get('timeout_per_attempt', 10.0)
        )
