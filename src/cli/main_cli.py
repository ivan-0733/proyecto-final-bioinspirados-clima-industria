"""
Typer-based CLI for MOEA/D ARM Evolution.

Modern command-line interface with:
- Rich formatting and progress bars
- Interactive configuration selection
- Executive summary generation
"""
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint

app = typer.Typer(
    name="moead-arm",
    help="MOEA/D for Association Rule Mining on Diabetes Dataset",
    add_completion=False
)
console = Console()


def resolve_config_path(config_name: str) -> Path:
    """
    Resolve configuration file path.
    
    Tries:
    1. Exact path if it exists
    2. config/<config_name> if not found
    3. config/<config_name>.json if not found
    
    Args:
        config_name: Config file name or path
        
    Returns:
        Resolved Path object
        
    Raises:
        typer.BadParameter: If file not found
    """
    # Try as-is first
    path = Path(config_name)
    if path.exists():
        return path
    
    # Try in config/ directory
    config_dir = Path('config')
    path = config_dir / config_name
    if path.exists():
        return path
    
    # Try adding .json extension
    if not config_name.endswith('.json'):
        path = config_dir / f"{config_name}.json"
        if path.exists():
            return path
    
    # Not found anywhere
    raise typer.BadParameter(
        f"Configuration file not found: {config_name}\n"
        f"Tried: {config_name}, config/{config_name}, config/{config_name}.json"
    )


def check_libraries() -> None:
    """Check if required libraries are installed."""
    import importlib.util
    
    required = ['numpy', 'pandas', 'pymoo', 'matplotlib', 'seaborn', 'typer', 'rich']
    missing = []
    
    for lib in required:
        if importlib.util.find_spec(lib) is None:
            missing.append(lib)
    
    if missing:
        console.print(f"[red]Error:[/red] Missing required libraries: {', '.join(missing)}")
        console.print(f"Install with: [cyan]pip install {' '.join(missing)}[/cyan]")
        raise typer.Exit(code=1)
    
    console.print("[green]OK[/green] Libraries check passed")


def check_directories() -> None:
    """Check if essential directories exist."""
    required_dirs = ['data', 'config', 'src', 'results']
    
    for d in required_dirs:
        path = Path(d)
        if not path.exists():
            console.print(f"[yellow]Warning:[/yellow] Directory '{d}' not found. Creating it...")
            path.mkdir(parents=True, exist_ok=True)
    
    console.print("[green]OK[/green] Directory structure check passed")


def list_configs() -> list[Path]:
    """List available configuration files."""
    config_dir = Path('config')
    configs = sorted(config_dir.glob('*.json'))
    
    if not configs:
        console.print(f"[red]Error:[/red] No configuration files found in '{config_dir}'")
        raise typer.Exit(code=1)
    
    return configs


def select_config_interactive() -> Path:
    """Interactive configuration selection with Rich."""
    configs = list_configs()
    
    # Create table
    table = Table(title="Available Configurations", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Configuration File", style="green")
    table.add_column("Path", style="dim")
    
    for i, config in enumerate(configs, 1):
        # Handle relative path safely
        try:
            rel_path = config.relative_to(Path.cwd())
        except ValueError:
            rel_path = config
        table.add_row(str(i), config.name, str(rel_path))
    
    console.print(table)
    
    # Prompt selection
    choice = Prompt.ask(
        "\n[bold]Select configuration[/bold]",
        default="1",
        show_default=True
    )
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(configs):
            raise ValueError
        selected = configs[idx]
    except ValueError:
        console.print("[yellow]Invalid selection. Using default (1)[/yellow]")
        selected = configs[0]
    
    console.print(f"\n[green]OK[/green] Selected: [cyan]{selected.name}[/cyan]\n")
    return selected


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (JSON)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Enable interactive mode for configuration selection"
    ),
    generate_report: bool = typer.Option(
        True,
        "--report/--no-report",
        help="Generate executive summary report after execution"
    )
):
    """
    Run MOEA/D evolution for association rule mining.
    
    Examples:
        moead-arm run                           # Interactive mode
        moead-arm run -c config/escenario_1.json  # Specify config
        moead-arm run --no-report               # Skip report generation
    """
    console.print(Panel.fit(
        "[bold cyan]MOEA/D for Association Rule Mining[/bold cyan]\n"
        "[dim]Multi-Objective Evolutionary Algorithm based on Decomposition[/dim]",
        border_style="cyan"
    ))
    
    try:
        # Pre-flight checks
        check_libraries()
        check_directories()
        
        # Configuration selection
        if config is None:
            if interactive:
                config = select_config_interactive()
            else:
                configs = list_configs()
                config = configs[0]
                console.print(f"[yellow]No config specified. Using:[/yellow] {config.name}")
        else:
            console.print(f"[green]OK[/green] Using config: [cyan]{config.name}[/cyan]\n")
        
        # Run orchestrator
        from orchestrator import Orchestrator
        
        console.print("\n[bold]Starting Evolution...[/bold]\n")
        
        orch = Orchestrator(str(config))
        orch.run()
        
        console.print("\n[green]OK[/green] Evolution completed successfully!")
        
        # Generate executive summary
        if generate_report:
            console.print("\n[bold]Generating Executive Summary...[/bold]")
            try:
                from src.cli.report_generator import generate_executive_summary
                report_path = generate_executive_summary(Path(orch.experiment_dir))
                console.print(f"[green]OK[/green] Report saved to: [cyan]{report_path}[/cyan]")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Report generation failed: {e}")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Execution cancelled by user.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]Error during execution:[/red] {e}")
        if typer.confirm("Show full traceback?", default=False):
            import traceback
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def list(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """List available configuration files."""
    configs = list_configs()
    
    table = Table(title="Available Configurations", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("File", style="green")
    
    if verbose:
        table.add_column("Full Path", style="dim")
    
    for i, config in enumerate(configs, 1):
        if verbose:
            table.add_row(str(i), config.name, str(config))
        else:
            table.add_row(str(i), config.name)
    
    console.print(table)


@app.command()
def validate(
    config: str = typer.Argument(
        ...,
        help="Configuration file name (e.g., 'escenario_1.json' or 'escenario_1')"
    )
):
    """
    Validate a configuration file.
    
    Examples:
        python main.py validate escenario_1.json
        python main.py validate escenario_1
        python main.py validate config/escenario_1.json
    """
    try:
        config_path = resolve_config_path(config)
    except typer.BadParameter as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    
    import json
    
    console.print(f"[bold]Validating:[/bold] {config_path.name}")
    
    try:
        # Load as JSON (legacy system doesn't use Pydantic yet)
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        console.print("[green]✓ Configuration file is valid JSON![/green]")
        
        # Show summary
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        # Extract info from legacy structure
        experiment = cfg.get('experiment', {})
        algorithm = cfg.get('algorithm', {})
        objectives_cfg = cfg.get('objectives', {})
        
        table.add_row("Experiment Name", experiment.get('name', 'N/A'))
        table.add_row("Scenario", experiment.get('scenario', 'N/A'))
        table.add_row("Population Size", str(algorithm.get('population_size', 'N/A')))
        table.add_row("Generations", str(algorithm.get('generations', 'N/A')))
        
        objectives = objectives_cfg.get('selected', [])
        table.add_row("Objectives", ", ".join(objectives) if objectives else 'N/A')
        
        mutation = algorithm.get('operators', {}).get('mutation', {})
        table.add_row("Mutation Method", mutation.get('method', 'N/A'))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Validation failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def info():
    """Show system and environment information."""
    import platform
    import numpy as np
    import pandas as pd
    import pymoo
    
    table = Table(title="System Information", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")
    
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())
    table.add_row("NumPy", np.__version__)
    table.add_row("Pandas", pd.__version__)
    table.add_row("pymoo", pymoo.__version__)
    
    try:
        import typer as t
        table.add_row("Typer", t.__version__)
    except:
        pass
    
    try:
        import rich as r
        table.add_row("Rich", r.__version__)
    except:
        pass
    
    console.print(table)


def main():
    """Entry point for CLI."""
    # Si no hay argumentos, ejecutar modo interactivo por defecto
    if len(sys.argv) == 1:
        # Simular: python main.py run --interactive
        sys.argv.extend(['run', '--interactive'])
    app()


if __name__ == "__main__":
    main()
