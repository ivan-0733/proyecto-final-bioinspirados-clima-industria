"""
Benchmark script for multi-seed analysis of MOEA/D ARM.

Executes multiple runs with different random seeds (first 10 primes) and generates:
- Statistical tables (min/max/avg/std) per metric at generations 50, 100, 150, 200, 300
- Top 3 rules per metric per scenario
- Knee point rules (best trade-off solutions)
- Seed closest to median performance
- CSV exports and terminal output with tables

Usage:
    python src/benchmark_seeds.py --scenario escenario_1 [--debug]
    python src/benchmark_seeds.py --scenario escenario_2 [--debug]
    python src/benchmark_seeds.py --all [--debug]
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from orchestrator import Orchestrator

console = Console()

# First 10 prime numbers as seeds
PRIME_SEEDS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
DEBUG_SEEDS = [2, 3, 5]  # Only 3 seeds for debug mode

# Report every N generations (then skip to end when values stabilize)
REPORT_INTERVAL = 5


class BenchmarkRunner:
    """Orchestrates multi-seed benchmark execution and analysis."""
    
    def __init__(self, scenario: str, debug: bool = False, output_dir: str = "results/benchmark"):
        self.scenario = scenario
        self.debug = debug
        self.output_dir = Path(output_dir)
        self.seeds = DEBUG_SEEDS if debug else PRIME_SEEDS
        
        # Results storage
        self.runs_data: Dict[int, Dict] = {}  # seed -> {gen -> stats}
        self.pareto_fronts: Dict[int, pd.DataFrame] = {}  # seed -> final pareto
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "debug" if debug else "full"
        self.run_dir = self.output_dir / f"{scenario}_{mode}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = self.run_dir / "benchmark.log"
        self._setup_logging()
        
        console.print(Panel(
            f"[bold cyan]Benchmark Runner Initialized[/bold cyan]\n"
            f"Scenario: {scenario}\n"
            f"Mode: {'DEBUG (3 seeds, 30 gens)' if debug else 'FULL (10 seeds, 300 gens)'}\n"
            f"Seeds: {self.seeds}\n"
            f"Output: {self.run_dir}",
            expand=False
        ))
    
    def _setup_logging(self):
        """Configure logging to file and console."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_all_seeds(self):
        """Execute algorithm with all seeds."""
        console.print("\n[bold green]Starting Multi-Seed Execution[/bold green]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"[cyan]Running {len(self.seeds)} seeds...",
                total=len(self.seeds)
            )
            
            for seed in self.seeds:
                progress.update(task, description=f"[cyan]Seed {seed}...")
                self.logger.info(f"Starting execution with seed {seed}")
                
                try:
                    run_data = self._execute_single_seed(seed)
                    self.runs_data[seed] = run_data
                    
                    # Load final pareto HISTORICAL (all unique solutions across generations)
                    pareto_path = self._get_pareto_historical_path(seed)
                    if pareto_path.exists():
                        self.pareto_fronts[seed] = pd.read_csv(pareto_path)
                    else:
                        # Fallback to final_pareto.csv if historical doesn't exist
                        pareto_path_fallback = self._get_pareto_path(seed)
                        if pareto_path_fallback.exists():
                            self.pareto_fronts[seed] = pd.read_csv(pareto_path_fallback)
                    
                    self.logger.info(f"Completed seed {seed} successfully")
                    
                except Exception as e:
                    self.logger.error(f"Failed seed {seed}: {e}", exc_info=True)
                    console.print(f"[red][FAIL] Seed {seed} failed: {e}[/red]")
                
                progress.advance(task)
        
        console.print(f"\n[bold green][OK] Completed {len(self.runs_data)}/{len(self.seeds)} seeds[/bold green]\n")
    
    def _execute_single_seed(self, seed: int) -> Dict:
        """Execute algorithm with specific seed and extract statistics."""
        # Load config and modify seed
        config_path = project_root / "config" / f"{self.scenario}.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Modify config for this run
        config['experiment']['random_seed'] = seed
        config['experiment']['output_root'] = str(self.run_dir / f"seed_{seed}")
        
        if self.debug:
            # Reduce generations for debug mode
            config['algorithm']['generations'] = 30
            config['algorithm']['logging_interval'] = 10
        
        # Save modified config
        seed_config_path = self.run_dir / f"config_seed_{seed}.json"
        with open(seed_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Execute orchestrator
        orchestrator = Orchestrator(str(seed_config_path))
        orchestrator.run()
        
        # Extract statistics from evolution_stats.csv
        stats_path = self._get_stats_path(seed)
        
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        
        stats_df = pd.read_csv(stats_path)
        
        # Organize by generation
        run_data = {}
        for _, row in stats_df.iterrows():
            gen = int(row['generation'])
            run_data[gen] = {
                'hypervolume': row['hypervolume']
            }
            
            # Extract metric stats (min/max/mean for each objective)
            for col in stats_df.columns:
                if col not in ['generation', 'hypervolume']:
                    run_data[gen][col] = row[col]
        
        return run_data
    
    def _get_stats_path(self, seed: int) -> Path:
        """Get path to evolution_stats.csv for a seed."""
        seed_dir = self.run_dir / f"seed_{seed}"
        # Find exp_XXX directory (search recursively because Orchestrator creates experiment_name/exp_XXX)
        exp_dirs = list(seed_dir.glob("**/exp_*"))
        if not exp_dirs:
            raise FileNotFoundError(f"No exp_* directory found in {seed_dir}")
        
        return exp_dirs[0] / "stats" / "evolution_stats.csv"
    
    def _get_pareto_path(self, seed: int) -> Path:
        """Get path to final_pareto.csv for a seed."""
        seed_dir = self.run_dir / f"seed_{seed}"
        # Find exp_XXX directory (search recursively)
        exp_dirs = list(seed_dir.glob("**/exp_*"))
        if not exp_dirs:
            raise FileNotFoundError(f"No exp_* directory found in {seed_dir}")
        
        return exp_dirs[0] / "final_pareto.csv"
    
    def _get_pareto_historical_path(self, seed: int) -> Path:
        """Get path to final_pareto_historical.csv for a seed (all unique solutions)."""
        seed_dir = self.run_dir / f"seed_{seed}"
        # Find exp_XXX directory (search recursively)
        exp_dirs = list(seed_dir.glob("**/exp_*"))
        if not exp_dirs:
            raise FileNotFoundError(f"No exp_* directory found in {seed_dir}")
        
        return exp_dirs[0] / "final_pareto_historical.csv"
    
    def compute_statistics(self) -> pd.DataFrame:
        """Compute min/max/mean directly from ALL pareto front values across all seeds.
        
        Reports every REPORT_INTERVAL generations, but applies uniform sampling
        to limit to maximum 10 rows per metric in final CSV.
        """
        console.print("\n[bold blue]Computing Statistics[/bold blue]\n")
        
        # Find max generation across all seeds
        max_gen = max(max(run_data.keys()) for run_data in self.runs_data.values())
        
        # Collect all objective columns from pareto fronts
        if not self.pareto_fronts:
            self.logger.warning("No pareto fronts available")
            return pd.DataFrame()
        
        first_pareto = list(self.pareto_fronts.values())[0]
        exclude_cols = {'rule', 'seed', 'generation', 'genome_hash', 'antecedent', 'consequent', 'encoded_rule', 'id'}
        objective_cols = [col for col in first_pareto.columns 
                         if col not in exclude_cols and pd.api.types.is_numeric_dtype(first_pareto[col])]
        
        # First pass: collect ALL generations data
        all_results = []
        current_gen = REPORT_INTERVAL
        while current_gen <= max_gen:
            row = {'generation': current_gen}
            
            # Collect ALL pareto values at this generation from all seeds
            for obj_col in objective_cols:
                all_values = []
                
                for seed, pareto_df in self.pareto_fronts.items():
                    # Filter by generation (handle early stopping)
                    available_gens = sorted(pareto_df['generation'].unique()) if 'generation' in pareto_df.columns else [max_gen]
                    
                    if current_gen in available_gens:
                        gen = current_gen
                    elif current_gen > max(available_gens):
                        gen = max(available_gens)
                    else:
                        gen = min(available_gens, key=lambda x: abs(x - current_gen))
                    
                    gen_pareto = pareto_df[pareto_df['generation'] == gen] if 'generation' in pareto_df.columns else pareto_df
                    
                    if obj_col in gen_pareto.columns:
                        all_values.extend(gen_pareto[obj_col].dropna().tolist())
                
                # Compute direct statistics from all values
                if all_values:
                    row[f"{obj_col}_min"] = float(np.min(all_values))
                    row[f"{obj_col}_max"] = float(np.max(all_values))
                    row[f"{obj_col}_mean"] = float(np.mean(all_values))
                    row[f"{obj_col}_std"] = float(np.std(all_values))
            
            all_results.append(row)
            current_gen += REPORT_INTERVAL
        
        full_df = pd.DataFrame(all_results)
        
        # Second pass: apply uniform sampling (max 10 rows)
        MAX_ROWS = 10
        if len(full_df) <= MAX_ROWS:
            sampled_df = full_df
        else:
            # Sample uniformly: indices spread across range
            indices = np.linspace(0, len(full_df) - 1, MAX_ROWS, dtype=int)
            sampled_df = full_df.iloc[indices].reset_index(drop=True)
        
        # Save sampled data to CSV
        csv_path = self.run_dir / "statistics_summary.csv"
        sampled_df.to_csv(csv_path, index=False)
        console.print(f"[green][OK] Statistics saved to: {csv_path}[/green]")
        
        return sampled_df
    
    def generate_tables(self, stats_df: pd.DataFrame):
        """Generate formatted tables for terminal display and save to file.
        
        Displays sampled data from stats_df (already limited to max 10 rows).
        Each metric shows min/max/mean values directly from ALL pareto fronts.
        """
        console.print("\n[bold blue]Generating Tables[/bold blue]\n")
        
        # Extract unique base metric names from columns ending with _min
        key_metrics = sorted(list(set([col.replace('_min', '') for col in stats_df.columns 
                                       if col.endswith('_min')])))
        
        # Create table per metric
        output_lines = []
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"BENCHMARK RESULTS - {self.scenario.upper()}")
        output_lines.append(f"{'='*80}\n")
        
        for metric in key_metrics:
            if f"{metric}_min" not in stats_df.columns:
                continue
            
            # Use ALL rows from sampled DataFrame (already limited to 10)
            metric_rows = stats_df.to_dict('records')
            
            # Rich table for console
            table = Table(title=f"{metric.upper()} Statistics", show_header=True, header_style="bold magenta")
            table.add_column("Generación", style="cyan", justify="center")
            table.add_column("min", justify="right")
            table.add_column("max", justify="right")
            table.add_column("Prom", justify="right")
            table.add_column("Prom(desv)", justify="right")
            
            # Text table for file
            output_lines.append(f"\n{metric.upper()}")
            output_lines.append("-" * 80)
            output_lines.append(f"{'Generación':<15} {'min':<15} {'max':<15} {'Prom':<15} {'Prom(desv)':<15}")
            output_lines.append("-" * 80)
            
            for row in metric_rows:
                gen = int(row['generation'])
                min_val = row[f"{metric}_min"]
                max_val = row[f"{metric}_max"]
                mean_val = row[f"{metric}_mean"]
                std_val = row[f"{metric}_std"]
                
                # Add to rich table
                table.add_row(
                    str(gen),
                    f"{min_val:.4f}",
                    f"{max_val:.4f}",
                    f"{mean_val:.4f}",
                    f"({std_val:.4f})"
                )
                
                # Add to text table
                output_lines.append(
                    f"{gen:<15} {min_val:<15.4f} {max_val:<15.4f} "
                    f"{mean_val:<15.4f} ({std_val:.4f})"
                )
            
            console.print(table)
            output_lines.append("")        # Save to file
        tables_file = self.run_dir / "tables_output.txt"
        with open(tables_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        console.print(f"\n[green][OK] Tables saved to: {tables_file}[/green]")
    
    def find_top_rules(self):
        """Find top 3 rules per metric and knee point rule per scenario."""
        console.print("\n[bold blue]Finding Top Rules and Knee Points[/bold blue]\n")
        
        output_lines = []
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"TOP RULES - {self.scenario.upper()}")
        output_lines.append(f"{'='*80}\n")
        
        # Combine all pareto fronts
        all_rules = []
        for seed, pareto_df in self.pareto_fronts.items():
            pareto_df['seed'] = seed
            all_rules.append(pareto_df)
        
        if not all_rules:
            console.print("[yellow][WARN] No pareto fronts available[/yellow]")
            return
        
        combined_df = pd.concat(all_rules, ignore_index=True)
        
        # Identify objective columns (only numeric columns that are actual metrics)
        exclude_cols = ['rule', 'seed', 'generation', 'genome_hash', 
                       'antecedent', 'consequent', 'encoded_rule', 'id']
        obj_cols = [col for col in combined_df.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(combined_df[col])]
        
        # Top 3 per metric (deduplicated by rule text)
        for metric in obj_cols:
            output_lines.append(f"\n{'─'*80}")
            output_lines.append(f"TOP 3 RULES - {metric.upper()}")
            output_lines.append(f"{'─'*80}")
            
            # Sort by metric (nlargest because we want BEST = HIGHEST after negation reversal)
            # In pymoo we negate, so the output CSVs have the REAL values (higher is better)
            sorted_df = combined_df.sort_values(metric, ascending=False)
            
            # Deduplicate by rule text, keep first (best) occurrence
            unique_rules = sorted_df.drop_duplicates(subset=['rule'], keep='first')
            top_3 = unique_rules.head(3)
            
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                output_lines.append(f"\n#{i} (Seed {row['seed']}, {metric}={row[metric]:.4f})")
                output_lines.append(f"  Rule: {row.get('rule', 'N/A')}")
                
                # Show all objectives (only numeric columns)
                obj_str = ", ".join([f"{obj}={row[obj]:.4f}" for obj in obj_cols 
                                    if isinstance(row[obj], (int, float, np.number))])
                output_lines.append(f"  Objectives: {obj_str}")
            
            console.print(f"[cyan][OK] Top 3 for {metric}[/cyan]")
        
        # Knee point (best trade-off)
        output_lines.append(f"\n{'='*80}")
        output_lines.append("KNEE POINT (Best Trade-off Rule)")
        output_lines.append(f"{'='*80}")
        
        knee_rule = self._find_knee_point(combined_df, obj_cols)
        
        if knee_rule is not None:
            output_lines.append(f"\nSeed: {knee_rule['seed']}")
            output_lines.append(f"Rule: {knee_rule.get('rule', 'N/A')}")
            obj_str = ", ".join([f"{obj}={knee_rule[obj]:.4f}" for obj in obj_cols 
                                if isinstance(knee_rule[obj], (int, float, np.number))])
            output_lines.append(f"Objectives: {obj_str}")
            
            console.print("[green][OK] Knee point identified[/green]")
        else:
            output_lines.append("\n[WARN] Could not identify knee point")
            console.print("[yellow][WARN] Could not identify knee point[/yellow]")
        
        # Save to file
        rules_file = self.run_dir / "top_rules.txt"
        with open(rules_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        console.print(f"\n[green][OK] Top rules saved to: {rules_file}[/green]")
    
    def _find_knee_point(self, pareto_df: pd.DataFrame, obj_cols: List[str]) -> Optional[pd.Series]:
        """
        Find knee point using distance to ideal point.
        
        The knee point is the solution closest to the ideal point (min of all objectives).
        """
        if pareto_df.empty or not obj_cols:
            return None
        
        # Extract objectives matrix
        objectives = pareto_df[obj_cols].values
        
        # Ideal point (minimum of each objective)
        ideal = np.min(objectives, axis=0)
        
        # Normalize objectives to [0, 1]
        obj_min = np.min(objectives, axis=0)
        obj_max = np.max(objectives, axis=0)
        
        # Avoid division by zero
        ranges = obj_max - obj_min
        ranges[ranges == 0] = 1.0
        
        normalized = (objectives - obj_min) / ranges
        ideal_normalized = np.zeros(len(obj_cols))  # Ideal is at origin in normalized space
        
        # Euclidean distance to ideal
        distances = np.linalg.norm(normalized - ideal_normalized, axis=1)
        
        # Knee point is closest to ideal
        knee_idx = np.argmin(distances)
        
        return pareto_df.iloc[knee_idx]
    
    def find_median_seed(self, stats_df: pd.DataFrame):
        """Identify seed closest to median hypervolume performance."""
        console.print("\n[bold blue]Finding Median Seed[/bold blue]\n")
        
        # Use final hypervolume per seed (last generation available)
        seed_hvs = {}
        
        for seed in self.runs_data.keys():
            run_data = self.runs_data[seed]
            # Get last generation
            last_gen = max(run_data.keys())
            seed_hvs[seed] = run_data[last_gen]['hypervolume']
        
        # Find median
        median_hv = np.median(list(seed_hvs.values()))
        
        # Closest seed to median
        closest_seed = min(seed_hvs.keys(), key=lambda s: abs(seed_hvs[s] - median_hv))
        
        output = []
        output.append(f"\n{'='*80}")
        output.append("MEDIAN SEED ANALYSIS")
        output.append(f"{'='*80}\n")
        output.append(f"Median Hypervolume: {median_hv:.6f}")
        output.append(f"Closest Seed: {closest_seed} (HV={seed_hvs[closest_seed]:.6f})")
        output.append(f"\nAll Seeds HV:")
        
        for seed, hv in sorted(seed_hvs.items()):
            marker = " ← MEDIAN" if seed == closest_seed else ""
            output.append(f"  Seed {seed:2d}: {hv:.6f}{marker}")
        
        output_text = '\n'.join(output)
        console.print(output_text)
        
        # Save to file
        median_file = self.run_dir / "median_seed.txt"
        with open(median_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        console.print(f"\n[green][OK] Median seed analysis saved to: {median_file}[/green]")
    
    def run_full_analysis(self):
        """Execute complete benchmark workflow."""
        try:
            # 1. Run all seeds
            self.run_all_seeds()
            
            if not self.runs_data:
                console.print("[red][FAIL] No successful runs. Aborting.[/red]")
                return
            
            # 2. Compute statistics
            stats_df = self.compute_statistics()
            
            # 3. Generate tables
            self.generate_tables(stats_df)
            
            # 4. Find top rules
            self.find_top_rules()
            
            # 5. Find median seed
            self.find_median_seed(stats_df)
            
            # Final summary
            console.print(Panel(
                f"[bold green][OK] Benchmark Complete![/bold green]\n\n"
                f"Results saved to: {self.run_dir}\n"
                f"- statistics_summary.csv\n"
                f"- tables_output.txt\n"
                f"- top_rules.txt\n"
                f"- median_seed.txt\n"
                f"- benchmark.log",
                title="Success",
                expand=False
            ))
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}", exc_info=True)
            console.print(f"[red][FAIL] Benchmark failed: {e}[/red]")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed benchmark for MOEA/D ARM scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['escenario_1', 'escenario_2', 'all'],
        default='all',
        help='Scenario to benchmark (default: all)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: 3 seeds, 30 generations (fast testing)'
    )
    
    args = parser.parse_args()
    
    scenarios = ['escenario_1', 'escenario_2'] if args.scenario == 'all' else [args.scenario]
    
    for scenario in scenarios:
        console.print(f"\n[bold magenta]{'='*80}[/bold magenta]")
        console.print(f"[bold magenta]Starting Benchmark: {scenario.upper()}[/bold magenta]")
        console.print(f"[bold magenta]{'='*80}[/bold magenta]\n")
        
        runner = BenchmarkRunner(scenario=scenario, debug=args.debug)
        runner.run_full_analysis()


if __name__ == "__main__":
    main()
