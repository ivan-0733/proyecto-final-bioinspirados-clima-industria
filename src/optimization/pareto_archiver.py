"""
Lazy Pareto archiver with deferred deduplication.
"""
from typing import List, Set, Optional
import numpy as np
from src.representation import Rule
from src.core.logging_config import get_logger


class LazyParetoArchiver:
    """
    Lazy archiver that stores candidates and deduplicates only on save.
    
    This reduces overhead during evolution by deferring expensive
    deduplication (SHA256 hashing + set operations) until final save.
    """
    
    def __init__(self):
        """Initialize lazy archiver."""
        self.candidates: List[tuple] = []  # (X, F, rule_hash)
        self.seen_hashes: Set[str] = set()
        
        self.log = get_logger(__name__)
        self.log.info("lazy_pareto_archiver_initialized")
    
    def add_candidate(self, X: np.ndarray, F: np.ndarray, rule: Rule) -> bool:
        """
        Add candidate to archive (no deduplication yet).
        
        Args:
            X: Genome
            F: Objectives
            rule: Rule representation
        
        Returns:
            True if added (always True, dedup happens later)
        """
        self.candidates.append((X.copy(), F.copy(), rule.hash))
        return True
    
    def deduplicate(self) -> int:
        """
        Deduplicate candidates by rule hash.
        
        Returns:
            Number of duplicates removed
        """
        unique_candidates = []
        seen = set()
        duplicates = 0
        
        for X, F, rule_hash in self.candidates:
            if rule_hash not in seen:
                unique_candidates.append((X, F, rule_hash))
                seen.add(rule_hash)
            else:
                duplicates += 1
        
        self.candidates = unique_candidates
        self.seen_hashes = seen
        
        self.log.info(
            "candidates_deduplicated",
            total_before=len(self.candidates) + duplicates,
            unique=len(unique_candidates),
            duplicates_removed=duplicates,
            dedup_rate=f"{duplicates / (len(self.candidates) + duplicates) * 100:.1f}%"
        )
        
        return duplicates
    
    def get_unique_count(self) -> int:
        """
        Get number of unique candidates (after dedup).
        
        Returns:
            Number of unique candidates
        """
        return len(self.candidates)
    
    def get_archive(self) -> tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get deduplicated archive.
        
        Returns:
            Tuple of (X_array, F_array, rule_hashes)
        """
        if not self.candidates:
            return np.array([]), np.array([]), []
        
        X_list = [X for X, F, _ in self.candidates]
        F_list = [F for X, F, _ in self.candidates]
        hashes = [h for _, _, h in self.candidates]
        
        return np.array(X_list), np.array(F_list), hashes
    
    def clear(self) -> None:
        """Clear the archive."""
        count = len(self.candidates)
        self.candidates.clear()
        self.seen_hashes.clear()
        
        self.log.info("archive_cleared", candidates_removed=count)
    
    def get_statistics(self) -> dict:
        """
        Get archiver statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_candidates = len(self.candidates)
        unique_hashes = len(self.seen_hashes)
        
        return {
            "total_candidates": total_candidates,
            "unique_candidates": unique_hashes,
            "duplication_rate": f"{(1 - unique_hashes / total_candidates) * 100:.1f}%" if total_candidates > 0 else "0.0%"
        }
