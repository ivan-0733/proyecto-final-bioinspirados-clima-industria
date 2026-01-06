"""
Rule representation with cryptographic hashing for deduplication.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Tuple, FrozenSet, Optional, Dict, Any


@dataclass(frozen=True)
class Rule:
    """
    Immutable association rule with canonical hashing.
    
    Attributes:
        antecedent: Frozenset of (var_idx, val_idx) tuples
        consequent: Frozenset of (var_idx, val_idx) tuples
        _hash: Cached SHA256 hash for O(1) equality checks
    """
    antecedent: FrozenSet[Tuple[int, int]]
    consequent: FrozenSet[Tuple[int, int]]
    _hash: str = field(init=False, repr=False, compare=False)
    
    def __post_init__(self):
        """Compute and cache the cryptographic hash."""
        object.__setattr__(self, '_hash', self._compute_hash())
    
    def _compute_hash(self) -> str:
        """
        Compute SHA256 hash of canonicalized rule.
        
        Canonicalization ensures:
        - Sorted items within each side
        - Consistent serialization format
        - Order-independent hashing
        """
        canonical = {
            'antecedent': sorted(list(self.antecedent)),
            'consequent': sorted(list(self.consequent))
        }
        serialized = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    
    @property
    def hash(self) -> str:
        """Get the cached hash."""
        return self._hash
    
    def __hash__(self) -> int:
        """Use first 8 bytes of SHA256 for Python hash()."""
        return int(self._hash[:16], 16)
    
    def __eq__(self, other) -> bool:
        """O(1) equality check via hash comparison."""
        if not isinstance(other, Rule):
            return False
        return self._hash == other._hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'antecedent': sorted(list(self.antecedent)),
            'consequent': sorted(list(self.consequent)),
            'hash': self._hash
        }
    
    @classmethod
    def from_items(cls, antecedent: List[Tuple[int, int]], consequent: List[Tuple[int, int]]) -> "Rule":
        """
        Create rule from lists (auto-converts to frozensets).
        
        Args:
            antecedent: List of (var_idx, val_idx) tuples
            consequent: List of (var_idx, val_idx) tuples
        
        Returns:
            Rule instance
        """
        return cls(
            antecedent=frozenset(antecedent),
            consequent=frozenset(consequent)
        )


class RuleEncoder:
    """
    Encodes Rule objects into human-readable strings using metadata.
    """
    def __init__(self, metadata: Dict[str, Any]):
        """
        Initialize encoder with dataset metadata.
        
        Args:
            metadata: Dataset metadata with variable info
        """
        self.metadata = metadata
        self.variables_info = self._parse_variables()
    
    def _parse_variables(self) -> List[Dict[str, Any]]:
        """Extract ordered variable information from metadata."""
        order = list(self.metadata.get('feature_order', []))
        target_name = self.metadata.get('target_variable')
        
        if isinstance(target_name, dict):
            target_name = target_name.get('name')
        if target_name and target_name not in order:
            order.append(target_name)
        
        variables_dict = self.metadata.get('variables', {})
        # Agregar el nombre de la variable al diccionario de info
        result = []
        for name in order:
            var_info = variables_dict.get(name, {}).copy()  # Copiar para no modificar original
            var_info['name'] = name  # Asegurar que el nombre esté presente
            result.append(var_info)
        return result
    
    def encode(self, rule: Rule) -> Tuple[str, str]:
        """
        Encode rule into (antecedent_str, consequent_str).
        
        Args:
            rule: Rule to encode
        
        Returns:
            Tuple of (antecedent_string, consequent_string)
        """
        def format_items(items: FrozenSet[Tuple[int, int]]) -> str:
            parts = []
            for var_idx, val_idx in sorted(items):
                if var_idx >= len(self.variables_info):
                    parts.append(f"VAR{var_idx}={val_idx}")
                    continue
                
                var_info = self.variables_info[var_idx]
                var_name = var_info.get('name', f'VAR{var_idx}')
                labels = var_info.get('labels', [])
                
                if labels and 0 <= val_idx < len(labels):
                    val_str = labels[val_idx]
                else:
                    val_str = str(val_idx)
                
                parts.append(f"{var_name}={val_str}")
            
            return " ^ ".join(parts) if parts else "{}"
        
        ant_str = format_items(rule.antecedent)
        con_str = format_items(rule.consequent)
        return ant_str, con_str
    
    def encode_full(self, rule: Rule) -> str:
        """
        Encode rule into full string format: "Antecedent => Consequent".
        
        Args:
            rule: Rule to encode
        
        Returns:
            Full rule string
        """
        ant_str, con_str = self.encode(rule)
        return f"{ant_str} => {con_str}"


class RuleDecoder:
    """
    Decodes diploid genomes into Rule objects.
    """
    def __init__(self, metadata: Dict[str, Any]):
        """
        Initialize decoder with dataset metadata.
        
        Args:
            metadata: Dataset metadata with variable info
        """
        self.metadata = metadata
        self.num_genes = len(metadata.get('feature_order', []))
        target = metadata.get('target_variable')
        if isinstance(target, dict):
            target = target.get('name')
        if target and target not in metadata.get('feature_order', []):
            self.num_genes += 1
    
    def decode(self, genome: Any) -> Rule:
        """
        Decode diploid genome into Rule.
        
        Genome structure: [role_0, ..., role_N, val_0, ..., val_N]
        - Roles: 0=ignore, 1=antecedent, 2=consequent
        - Values: integer indices
        
        Args:
            genome: 1D array or sequence of length 2*num_genes
        
        Returns:
            Rule object
        """
        if len(genome) != 2 * self.num_genes:
            raise ValueError(
                f"Genome length {len(genome)} doesn't match expected {2 * self.num_genes}"
            )
        
        antecedent = []
        consequent = []
        
        for i in range(self.num_genes):
            role = int(genome[i])
            val = int(genome[self.num_genes + i])
            
            if role == 1:
                antecedent.append((i, val))
            elif role == 2:
                consequent.append((i, val))
        
        return Rule.from_items(antecedent, consequent)
