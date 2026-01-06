"""
Refactored Individual class using new Rule representation.
Maintains backward compatibility with legacy code.
"""
import numpy as np
import random
from typing import Dict, Any, Optional, Tuple, List
from pymoo.core.individual import Individual as PymooIndividual

from .rule import Rule, RuleDecoder, RuleEncoder


class RuleIndividual(PymooIndividual):
    """
    Refactored individual with diploid genome representation.
    
    Key improvements over legacy:
    - Uses Rule class with SHA256 hashing
    - Integrates with new validators
    - Better separation of concerns
    - Type hints for clarity
    
    Genome structure (1D array):
        [role_0, role_1, ..., role_N, val_0, val_1, ..., val_N]
        
    Roles:
        0 = ignore variable
        1 = antecedent
        2 = consequent
    """
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize individual with metadata.
        
        Args:
            metadata: Dataset metadata with variable information
            **kwargs: Additional arguments for pymoo Individual
        """
        super().__init__(**kwargs)
        self.metadata = metadata
        self._decoder: Optional[RuleDecoder] = None
        self._encoder: Optional[RuleEncoder] = None
        
        if self.metadata:
            self.variables_info = self._parse_variables()
            self.num_genes = len(self.variables_info)
            self.X = np.zeros(2 * self.num_genes, dtype=int)
            
            # Initialize encoder/decoder
            self._decoder = RuleDecoder(metadata)
            self._encoder = RuleEncoder(metadata)
    
    def _parse_variables(self) -> List[Dict[str, Any]]:
        """
        Extract variable information from metadata.
        
        Returns:
            List of variable info dicts with name, cardinality, labels
        """
        if not self.metadata:
            return []
        
        variables_info = []
        order = list(self.metadata.get('feature_order', []))
        
        # Add target variable if not in order
        target_name = self.metadata.get('target_variable')
        if isinstance(target_name, dict):
            target_name = target_name.get('name')
        if target_name and target_name not in order:
            order.append(target_name)
        
        variables_dict = self.metadata.get('variables', {})
        
        for name in order:
            if name in variables_dict:
                info = variables_dict[name]
                if 'cardinality' in info:
                    variables_info.append({
                        'name': name,
                        'cardinality': info['cardinality'],
                        'labels': info.get('labels', []),
                        'encoding': info.get('encoding', {})
                    })
                else:
                    raise ValueError(f"Variable '{name}' lacks 'cardinality' information.")
        
        return variables_info
    
    def initialize(self, sparsity: float = 0.6) -> None:
        """
        Randomly initialize the individual with configurable sparsity.
        
        Args:
            sparsity: Probability of ignoring a variable (0=all active, 1=all ignore)
        """
        # Calculate probabilities for role assignment
        p_ignore = sparsity
        p_antecedent = (1 - sparsity) / 2
        p_consequent = (1 - sparsity) / 2
        
        for i in range(self.num_genes):
            role_idx = i
            val_idx = self.num_genes + i
            
            # Assign role
            self.X[role_idx] = np.random.choice(
                [0, 1, 2],
                p=[p_ignore, p_antecedent, p_consequent]
            )
            
            # Assign value if active
            if self.X[role_idx] != 0:
                card = self.variables_info[i]['cardinality']
                self.X[val_idx] = np.random.randint(0, card)  # FIX: Usar np.random en lugar de random
            else:
                self.X[val_idx] = 0
    
    def repair(self) -> None:
        """
        Repair individual to ensure consistency.
        
        Rules enforced:
        1. If role=0, value must be 0
        2. Values must be within cardinality bounds
        3. At least one antecedent and one consequent (structural repair)
        """
        # Rule 1 & 2: Consistency and bounds
        for i in range(self.num_genes):
            role_idx = i
            val_idx = self.num_genes + i
            
            if self.X[role_idx] == 0:
                self.X[val_idx] = 0
            else:
                card = self.variables_info[i]['cardinality']
                # Clamp to valid range
                self.X[val_idx] = np.clip(self.X[val_idx], 0, card - 1)
        
        # Rule 3: Structural repair (ensure non-empty sides)
        self._repair_structure()
    
    def _repair_structure(self) -> None:
        """
        Ensure rule has at least one antecedent and one consequent.
        """
        roles = self.X[:self.num_genes]
        has_ant = np.any(roles == 1)
        has_con = np.any(roles == 2)
        
        # Repair missing antecedent
        if not has_ant:
            zeros = np.where(roles == 0)[0]
            if len(zeros) > 0:
                idx = np.random.choice(zeros)
                self.X[idx] = 1
                card = self.variables_info[idx]['cardinality']
                self.X[self.num_genes + idx] = np.random.randint(0, card)
            else:
                # Flip a consequent to antecedent
                cons = np.where(roles == 2)[0]
                if len(cons) > 0:
                    idx = np.random.choice(cons)
                    self.X[idx] = 1
        
        # Repair missing consequent
        roles = self.X[:self.num_genes]  # Refresh after potential change
        has_con = np.any(roles == 2)
        
        if not has_con:
            zeros = np.where(roles == 0)[0]
            if len(zeros) > 0:
                idx = np.random.choice(zeros)
                self.X[idx] = 2
                card = self.variables_info[idx]['cardinality']
                self.X[self.num_genes + idx] = np.random.randint(0, card)
            else:
                # Flip an antecedent to consequent
                ants = np.where(roles == 1)[0]
                if len(ants) > 0:
                    idx = np.random.choice(ants)
                    self.X[idx] = 2
    
    def get_rule_items(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Extract rule structure as lists of (var_idx, val_idx) tuples.
        
        Returns:
            Tuple of (antecedent_items, consequent_items)
        """
        antecedent = []
        consequent = []
        
        for i in range(self.num_genes):
            role = self.X[i]
            val = self.X[self.num_genes + i]
            
            if role == 1:
                antecedent.append((i, int(val)))
            elif role == 2:
                consequent.append((i, int(val)))
        
        return antecedent, consequent
    
    def to_rule(self) -> Rule:
        """
        Convert genome to Rule object with SHA256 hashing.
        
        Returns:
            Rule instance
        """
        if self._decoder is None:
            raise ValueError("Decoder not initialized (metadata required)")
        
        return self._decoder.decode(self.X)
    
    def decode_parts(self) -> Tuple[str, str]:
        """
        Decode into human-readable (antecedent_str, consequent_str).
        
        Returns:
            Tuple of (antecedent_string, consequent_string)
        """
        if self._encoder is None:
            raise ValueError("Encoder not initialized (metadata required)")
        
        rule = self.to_rule()
        return self._encoder.encode(rule)
    
    def decode(self) -> str:
        """
        Decode into full rule string: "Antecedent => Consequent".
        
        Returns:
            Human-readable rule string
        """
        if self._encoder is None:
            raise ValueError("Encoder not initialized (metadata required)")
        
        rule = self.to_rule()
        return self._encoder.encode_full(rule)
    
    def __repr__(self) -> str:
        """String representation of individual."""
        try:
            return self.decode()
        except Exception:
            return f"Individual(genes={self.num_genes}, X_len={len(self.X) if self.X is not None else 0})"
    
    def __hash__(self) -> int:
        """
        Hash based on Rule (SHA256).
        Enables O(1) duplicate detection.
        """
        try:
            rule = self.to_rule()
            return hash(rule)
        except Exception:
            # Fallback to genome hash if Rule conversion fails
            return hash(tuple(self.X.flatten()))
    
    def __eq__(self, other) -> bool:
        """
        Equality based on Rule (order-independent).
        """
        if not isinstance(other, RuleIndividual):
            return False
        
        try:
            return self.to_rule() == other.to_rule()
        except Exception:
            # Fallback to genome comparison
            return np.array_equal(self.X, other.X)
