"""
Refactored crossover operator.
"""
import numpy as np
from pymoo.core.crossover import Crossover


class DiploidNPointCrossover(Crossover):
    """
    N-Point crossover for diploid individuals (refactored).
    
    Preserves vertical gene structure (role + value stay together).

    N-points are randomly selected between 1 and the length of the genome - 1.
    """
    
    def __init__(self, prob: float = 0.9, **kwargs):
        """
        Initialize crossover operator.
        
        Args:
            n_points: Number of crossover points
            prob: Crossover probability
            **kwargs: Additional pymoo arguments
        """
        super().__init__(2, 2, **kwargs)
        self.prob = prob
    
    def _do(self, problem, X, **kwargs):
        """
        Perform n-point crossover.
        
        Args:
            problem: Optimization problem
            X: Parent genomes (n_matings, n_parents, n_var)
            **kwargs: Additional arguments
        
        Returns:
            Offspring genomes (n_matings, n_parents, n_var)
        """
        n_matings, n_parents, n_var = X.shape
        n_genes = n_var // 2
        
        # Reshape to (n_matings, n_parents, 2, n_genes)
        X_reshaped = X.reshape(n_matings, n_parents, 2, n_genes)
        Y_reshaped = np.zeros_like(X_reshaped)
        
        for i in range(n_matings):
            parent1 = X_reshaped[i, 0]
            parent2 = X_reshaped[i, 1]
            
            # Select random number of cut points (1 to n_genes-1)
            if n_genes > 1:
                # Random number of points between 1 and n_genes-1 (fully random)
                actual_n_points = np.random.randint(1, n_genes)
                
                cut_points = np.sort(
                    np.random.choice(
                        np.arange(1, n_genes),
                        actual_n_points,
                        replace=False
                    )
                )
                points = np.concatenate(([0], cut_points, [n_genes]))
            else:
                points = np.array([0, n_genes])
            
            # Create offspring by alternating segments
            off1 = np.zeros_like(parent1)
            off2 = np.zeros_like(parent2)
            
            swap = False
            for j in range(len(points) - 1):
                start, end = points[j], points[j + 1]
                
                if not swap:
                    off1[:, start:end] = parent1[:, start:end]
                    off2[:, start:end] = parent2[:, start:end]
                else:
                    off1[:, start:end] = parent2[:, start:end]
                    off2[:, start:end] = parent1[:, start:end]
                
                swap = not swap
            
            Y_reshaped[i, 0] = off1
            Y_reshaped[i, 1] = off2
        
        # Flatten back
        Y = Y_reshaped.reshape(n_matings, n_parents, n_var)
        return Y
