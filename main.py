#!/usr/bin/env python
"""
MOEA/D ARM Evolution - Modern CLI Entry Point.

Uses Typer for command-line interface with Rich formatting.
Defaults to interactive mode when called without arguments.
"""
from src.cli.main_cli import main

if __name__ == "__main__":
    main()
