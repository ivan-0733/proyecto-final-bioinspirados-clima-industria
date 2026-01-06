#!/usr/bin/env python3
"""
CSV to LaTeX Table Converter for Benchmark Results

Converts statistics_summary.csv from benchmark results into LaTeX table format.
Supports both Scenario 1 (casual-supp, casual-conf, maxConf) and 
Scenario 2 (jaccard, cosine, phi, kappa) metrics.

Usage:
    python src/csv_to_latex.py [--benchmark-dir PATH] [--decimals N]

Examples:
    # Use latest benchmark (default)
    python src/csv_to_latex.py
    
    # Specify benchmark directory
    python src/csv_to_latex.py --benchmark-dir results/benchmark/escenario_1_full_20251208_123456
    
    # Change decimal precision
    python src/csv_to_latex.py --decimals 3
"""

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime


def find_latest_benchmark(results_dir: Path = Path("results/benchmark")) -> Path:
    """Find the most recent benchmark directory."""
    if not results_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {results_dir}")
    
    # Get all benchmark directories
    benchmark_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    if not benchmark_dirs:
        raise FileNotFoundError(f"No benchmark directories found in {results_dir}")
    
    # Sort by modification time (most recent first)
    benchmark_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return benchmark_dirs[0]


def detect_scenario(df: pd.DataFrame) -> str:
    """Detect which scenario based on column names."""
    columns = set(df.columns)
    
    # Check for scenario 1 metrics (casual)
    if 'casual_supp_min' in columns or 'casual-supp_min' in columns:
        return "scenario_1"
    
    # Check for scenario 2 metrics (correlation)
    if 'jaccard_min' in columns:
        return "scenario_2"
    
    raise ValueError("Could not detect scenario from CSV columns")


def format_value(value: float, decimals: int = 2, no_leading_zero: bool = False) -> str:
    """
    Format a float value for LaTeX output.
    
    Args:
        value: The value to format
        decimals: Number of decimal places
        no_leading_zero: If True, omit leading zero (e.g., .08 instead of 0.08)
    """
    formatted = f"{value:.{decimals}f}"
    
    if no_leading_zero and formatted.startswith("0."):
        formatted = formatted[1:]  # Remove leading "0"
    elif no_leading_zero and formatted.startswith("-0."):
        formatted = "-" + formatted[2:]  # Keep negative sign, remove "0"
    
    return formatted


def generate_scenario1_latex(df: pd.DataFrame, decimals: int = 2, caption: str = None) -> str:
    """
    Generate LaTeX table for Scenario 1 (Casual ARM).
    
    Metrics: casual-supp, casual-conf, maxConf, hypervolume
    """
    # Try both naming conventions (underscore and hyphen)
    metric_columns = {
        'casual_supp': ['casual_supp_min', 'casual_supp_max', 'casual_supp_mean', 'casual_supp_std'],
        'casual_conf': ['casual_conf_min', 'casual_conf_max', 'casual_conf_mean', 'casual_conf_std'],
        'maxconf': ['maxconf_min', 'maxconf_max', 'maxconf_mean', 'maxconf_std'],
        'hypervolume': ['hypervolume_min', 'hypervolume_max', 'hypervolume_mean', 'hypervolume_std']
    }
    
    # Check if columns use hyphens instead
    if 'casual-supp_min' in df.columns:
        metric_columns['casual_supp'] = ['casual-supp_min', 'casual-supp_max', 'casual-supp_mean', 'casual-supp_std']
        metric_columns['casual_conf'] = ['casual-conf_min', 'casual-conf_max', 'casual-conf_mean', 'casual-conf_std']
    
    # Check if maxConf uses capital C
    if 'maxConf_min' in df.columns:
        metric_columns['maxconf'] = ['maxConf_min', 'maxConf_max', 'maxConf_mean', 'maxConf_std']
    
    if caption is None:
        caption = "Escenario 1: Métricas de ARM Casual"
    
    # Start building LaTeX
    latex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"",
        r"    \begin{tabular}{c|ccc|ccc|ccc|ccc}",
        r"    \toprule",
        r"    \textbf{Gen}",
        r"    & \multicolumn{3}{c|}{\textbf{Casual-supp}}",
        r"    & \multicolumn{3}{c|}{\textbf{Casual-conf}}",
        r"    & \multicolumn{3}{c|}{\textbf{Max-conf}}",
        r"    & \multicolumn{3}{c}{\textbf{Hipervolumen}} \\",
        r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}",
        r"    & min & max & Prom (desv)",
        r"    & min & max & Prom (desv)",
        r"    & min & max & Prom (desv)",
        r"    & min & max & Prom (desv) \\",
        r"    \midrule",
    ]
    
    # Add data rows
    for _, row in df.iterrows():
        gen = int(row['generation'])
        
        # Casual-supp
        cs_min = format_value(row[metric_columns['casual_supp'][0]], decimals)
        cs_max = format_value(row[metric_columns['casual_supp'][1]], decimals)
        cs_mean = format_value(row[metric_columns['casual_supp'][2]], decimals)
        cs_std = format_value(row[metric_columns['casual_supp'][3]], decimals)
        
        # Casual-conf
        cc_min = format_value(row[metric_columns['casual_conf'][0]], decimals)
        cc_max = format_value(row[metric_columns['casual_conf'][1]], decimals)
        cc_mean = format_value(row[metric_columns['casual_conf'][2]], decimals)
        cc_std = format_value(row[metric_columns['casual_conf'][3]], decimals)
        
        # MaxConf
        mc_min = format_value(row[metric_columns['maxconf'][0]], decimals)
        mc_max = format_value(row[metric_columns['maxconf'][1]], decimals)
        mc_mean = format_value(row[metric_columns['maxconf'][2]], decimals)
        mc_std = format_value(row[metric_columns['maxconf'][3]], decimals)
        
        # Hypervolume
        hv_min = format_value(row[metric_columns['hypervolume'][0]], decimals)
        hv_max = format_value(row[metric_columns['hypervolume'][1]], decimals)
        hv_mean = format_value(row[metric_columns['hypervolume'][2]], decimals)
        hv_std = format_value(row[metric_columns['hypervolume'][3]], decimals)
        
        latex.append(
            f"    {gen:3d}\n"
            f"    & {cs_min} & {cs_max} & {cs_mean} ({cs_std})\n"
            f"    & {cc_min} & {cc_max} & {cc_mean} ({cc_std})\n"
            f"    & {mc_min} & {mc_max} & {mc_mean} ({mc_std})\n"
            f"    & {hv_min} & {hv_max} & {hv_mean} ({hv_std})\n"
            f"    \\\\"
        )
    
    # Close table
    latex.extend([
        r"    \bottomrule",
        r"    \end{tabular}",
        f"\\caption{{{caption}}}",
        r"\end{table}",
    ])
    
    return "\n".join(latex)


def generate_scenario2_latex(df: pd.DataFrame, decimals: int = 2, caption: str = None) -> str:
    """
    Generate LaTeX table for Scenario 2 (Correlation metrics).
    
    Metrics: jaccard, cosine, phi, kappa, hypervolume
    """
    if caption is None:
        caption = "Escenario 2: Métricas de Correlación"
    
    # Start building LaTeX
    latex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt} % compactar columnas",
        r"",
        r"\begin{tabular}{@{}c|ccc|ccc|ccc|ccc|ccc@{}}",
        r"\toprule",
        r"\textbf{Gen}",
        r"& \multicolumn{3}{c|}{\textbf{Jaccard}}",
        r"& \multicolumn{3}{c|}{\textbf{Coseno}}",
        r"& \multicolumn{3}{c|}{\textbf{Phi}}",
        r"& \multicolumn{3}{c|}{\textbf{Kappa}}",
        r"& \multicolumn{3}{c}{\textbf{Hipervolumen}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13} \cmidrule(lr){14-16}",
        r"& min & max & Prom (desv)",
        r"& min & max & Prom (desv)",
        r"& min & max & Prom (desv)",
        r"& min & max & Prom (desv)",
        r"& min & max & Prom (desv) \\",
        r"\midrule",
    ]
    
    # Add data rows (use no leading zero format for scenario 2)
    for _, row in df.iterrows():
        gen = int(row['generation'])
        
        # Jaccard
        j_min = format_value(row['jaccard_min'], decimals, no_leading_zero=True)
        j_max = format_value(row['jaccard_max'], decimals, no_leading_zero=True)
        j_mean = format_value(row['jaccard_mean'], decimals, no_leading_zero=True)
        j_std = format_value(row['jaccard_std'], decimals, no_leading_zero=True)
        
        # Cosine
        c_min = format_value(row['cosine_min'], decimals, no_leading_zero=True)
        c_max = format_value(row['cosine_max'], decimals, no_leading_zero=True)
        c_mean = format_value(row['cosine_mean'], decimals, no_leading_zero=True)
        c_std = format_value(row['cosine_std'], decimals, no_leading_zero=True)
        
        # Phi
        p_min = format_value(row['phi_min'], decimals, no_leading_zero=True)
        p_max = format_value(row['phi_max'], decimals, no_leading_zero=True)
        p_mean = format_value(row['phi_mean'], decimals, no_leading_zero=True)
        p_std = format_value(row['phi_std'], decimals, no_leading_zero=True)
        
        # Kappa
        k_min = format_value(row['kappa_min'], decimals, no_leading_zero=True)
        k_max = format_value(row['kappa_max'], decimals, no_leading_zero=True)
        k_mean = format_value(row['kappa_mean'], decimals, no_leading_zero=True)
        k_std = format_value(row['kappa_std'], decimals, no_leading_zero=True)
        
        # Hypervolume
        hv_min = format_value(row['hypervolume_min'], decimals, no_leading_zero=True)
        hv_max = format_value(row['hypervolume_max'], decimals, no_leading_zero=True)
        hv_mean = format_value(row['hypervolume_mean'], decimals, no_leading_zero=True)
        hv_std = format_value(row['hypervolume_std'], decimals, no_leading_zero=True)
        
        latex.append(
            f"{gen}\n"
            f"& {j_min} & {j_max} & {j_mean} ({j_std})\n"
            f"& {c_min} & {c_max} & {c_mean} ({c_std})\n"
            f"& {p_min} & {p_max} & {p_mean} ({p_std})\n"
            f"& {k_min} & {k_max} & {k_mean} ({k_std})\n"
            f"& {hv_min} & {hv_max} & {hv_mean} ({hv_std})\n"
            f"\\\\"
        )
    
    # Close table
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        f"\\caption{{{caption}}}",
        r"\end{table}",
    ])
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark CSV statistics to LaTeX table format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        help="Path to benchmark directory (default: latest in results/benchmark/)"
    )
    
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places (default: 2)"
    )
    
    parser.add_argument(
        "--caption",
        type=str,
        help="Custom caption for the table"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: print to console)"
    )
    
    args = parser.parse_args()
    
    # Find benchmark directory
    if args.benchmark_dir:
        benchmark_dir = args.benchmark_dir
    else:
        print("🔍 Finding latest benchmark...")
        benchmark_dir = find_latest_benchmark()
    
    print(f"📁 Using benchmark: {benchmark_dir.name}")
    
    # Load CSV
    csv_path = benchmark_dir / "statistics_summary.csv"
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found")
        return 1
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from CSV")
    
    # Detect scenario
    scenario = detect_scenario(df)
    print(f"🎯 Detected: {scenario}")
    
    # Generate LaTeX
    if scenario == "scenario_1":
        latex_output = generate_scenario1_latex(df, args.decimals, args.caption)
    else:  # scenario_2
        latex_output = generate_scenario2_latex(df, args.decimals, args.caption)
    
    # Output
    if args.output:
        args.output.write_text(latex_output, encoding='utf-8')
        print(f"💾 LaTeX table saved to: {args.output}")
    else:
        print("\n" + "="*80)
        print("LATEX OUTPUT")
        print("="*80 + "\n")
        print(latex_output)
        print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
