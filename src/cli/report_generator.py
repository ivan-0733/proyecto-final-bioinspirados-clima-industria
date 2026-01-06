"""
Executive Summary Report Generator for MOEA/D ARM Evolution.

Generates comprehensive HTML report with:
- Evolution statistics (HV, diversity, convergence)
- Pareto front visualizations
- Rule distribution analysis
- Performance metrics
"""
from pathlib import Path
from typing import Optional
import json
import pandas as pd
from datetime import datetime
from src.core.logging_config import get_logger

log = get_logger(__name__)


def generate_executive_summary(exp_dir: Path) -> Path:
    """
    Generate executive summary report for completed evolution.
    
    Args:
        exp_dir: Experiment directory path
    
    Returns:
        Path to generated HTML report
    """
    exp_dir = Path(exp_dir)
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    log.info("generating_executive_summary", exp_dir=str(exp_dir))
    
    # Load data
    config = _load_config(exp_dir)
    stats = _load_statistics(exp_dir)
    pareto = _load_pareto_front(exp_dir)
    
    # Generate report sections
    html_content = _build_html_report(
        exp_dir=exp_dir,
        config=config,
        stats=stats,
        pareto=pareto
    )
    
    # Save report
    report_path = exp_dir / "executive_summary.html"
    report_path.write_text(html_content, encoding='utf-8')
    
    log.info("executive_summary_generated", report_path=str(report_path))
    
    return report_path


def _load_config(exp_dir: Path) -> dict:
    """Load configuration snapshot."""
    config_path = exp_dir / "config_snapshot.json"
    
    if not config_path.exists():
        log.warning("config_snapshot_not_found", exp_dir=str(exp_dir))
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)


def _load_statistics(exp_dir: Path) -> Optional[pd.DataFrame]:
    """Load evolution statistics."""
    stats_path = exp_dir / "stats" / "evolution_stats.csv"
    
    if not stats_path.exists():
        log.warning("evolution_stats_not_found", exp_dir=str(exp_dir))
        return None
    
    return pd.read_csv(stats_path)


def _load_pareto_front(exp_dir: Path) -> Optional[pd.DataFrame]:
    """Load final Pareto front."""
    pareto_path = exp_dir / "final_pareto.csv"
    
    if not pareto_path.exists():
        log.warning("final_pareto_not_found", exp_dir=str(exp_dir))
        return None
    
    return pd.read_csv(pareto_path)


def _build_html_report(
    exp_dir: Path,
    config: dict,
    stats: Optional[pd.DataFrame],
    pareto: Optional[pd.DataFrame]
) -> str:
    """Build HTML report content."""
    
    # Extract key metrics
    if stats is not None and not stats.empty:
        final_hv = stats['hypervolume'].iloc[-1] if 'hypervolume' in stats else "N/A"
        final_gen = stats['generation'].iloc[-1] if 'generation' in stats else "N/A"
        avg_diversity = stats['diversity'].mean() if 'diversity' in stats else "N/A"
    else:
        final_hv = final_gen = avg_diversity = "N/A"
    
    if pareto is not None and not pareto.empty:
        pareto_size = len(pareto)
    else:
        pareto_size = 0
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOEA/D ARM - Executive Summary</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 2rem;
        }}
        
        .section {{
            margin-bottom: 3rem;
        }}
        
        .section h2 {{
            color: #667eea;
            font-size: 1.8rem;
            margin-bottom: 1rem;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .metric-card h3 {{
            color: #667eea;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }}
        
        .metric-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }}
        
        .config-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        .config-table th,
        .config-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .config-table th {{
            background: #f5f7fa;
            color: #667eea;
            font-weight: 600;
        }}
        
        .config-table tr:hover {{
            background: #f9f9f9;
        }}
        
        .footer {{
            background: #f5f7fa;
            padding: 1.5rem 2rem;
            text-align: center;
            color: #666;
            font-size: 0.9rem;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 MOEA/D ARM Evolution Report</h1>
            <p>Multi-Objective Evolutionary Algorithm based on Decomposition</p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </p>
        </div>
        
        <div class="content">
            <!-- Key Metrics Section -->
            <div class="section">
                <h2>🎯 Key Metrics</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Final Generation</h3>
                        <div class="value">{final_gen}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Hypervolume</h3>
                        <div class="value">{final_hv}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Pareto Front Size</h3>
                        <div class="value">{pareto_size}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Avg Diversity</h3>
                        <div class="value">{avg_diversity}</div>
                    </div>
                </div>
            </div>
            
            <!-- Configuration Section -->
            <div class="section">
                <h2>⚙️ Configuration</h2>
                <table class="config-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Experiment Name</strong></td>
                            <td>{config.get('experiment_name', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td><strong>Population Size</strong></td>
                            <td>{config.get('algorithm', {}).get('population_size', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td><strong>Generations</strong></td>
                            <td>{config.get('termination', {}).get('n_gen', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td><strong>Objectives</strong></td>
                            <td>{', '.join(config.get('objectives', []))}</td>
                        </tr>
                        <tr>
                            <td><strong>Scenario</strong></td>
                            <td>
                                {config.get('scenario', 'Not specified')}
                                <span class="badge badge-info">{len(config.get('objectives', []))} objectives</span>
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Decomposition</strong></td>
                            <td>{config.get('algorithm', {}).get('decomposition', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td><strong>Neighborhood Size</strong></td>
                            <td>{config.get('algorithm', {}).get('n_neighbors', 'N/A')}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- Files Section -->
            <div class="section">
                <h2>📁 Generated Files</h2>
                <table class="config-table">
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>final_pareto.csv</code></td>
                            <td>
                                <span class="badge {'badge-success' if pareto is not None else 'badge-danger'}">
                                    {'✓ Available' if pareto is not None else '✗ Missing'}
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td><code>stats/evolution_stats.csv</code></td>
                            <td>
                                <span class="badge {'badge-success' if stats is not None else 'badge-danger'}">
                                    {'✓ Available' if stats is not None else '✗ Missing'}
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td><code>plots/</code> (visualizations)</td>
                            <td>
                                <span class="badge {'badge-success' if (exp_dir / 'plots').exists() else 'badge-danger'}">
                                    {'✓ Available' if (exp_dir / 'plots').exists() else '✗ Missing'}
                                </span>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- Experiment Path -->
            <div class="section">
                <h2>📍 Experiment Location</h2>
                <p style="background: #f5f7fa; padding: 1rem; border-radius: 8px; font-family: monospace; word-break: break-all;">
                    {exp_dir.absolute()}
                </p>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by MOEA/D ARM Evolution System</p>
            <p style="margin-top: 0.5rem; font-size: 0.8rem;">
                Clean Architecture • SOLID Principles • Comprehensive Testing
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    return html


if __name__ == '__main__':
    # Test with latest experiment
    import sys
    if len(sys.argv) > 1:
        exp_dir = Path(sys.argv[1])
    else:
        # Find latest experiment
        results_dir = Path('results')
        exps = sorted(results_dir.glob('*/exp_*'))
        if not exps:
            print("No experiments found")
            sys.exit(1)
        exp_dir = exps[-1]
    
    report = generate_executive_summary(exp_dir)
    print(f"Report generated: {report}")
