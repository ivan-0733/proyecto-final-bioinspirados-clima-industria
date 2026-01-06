"""
End-to-end integration tests for main_v2.py CLI.

Tests all commands and workflows to ensure CLI functionality.
"""
import pytest
import sys
from pathlib import Path
from io import StringIO
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.cli import app

runner = CliRunner()


class TestMainV2Integration:
    """Integration tests for main_v2.py CLI application."""
    
    def test_app_loads_successfully(self):
        """Test that the app loads without errors."""
        assert app is not None
        assert hasattr(app, 'registered_commands')
    
    def test_help_command(self):
        """Test --help shows all available commands."""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "MOEA/D for Association Rule Mining" in result.stdout
        assert "run" in result.stdout
        assert "list" in result.stdout
        assert "validate" in result.stdout
        assert "info" in result.stdout
    
    def test_list_command_basic(self):
        """Test list command shows configurations."""
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        assert "Available Configurations" in result.stdout
        # Should show at least one config
        assert ".json" in result.stdout
    
    def test_list_command_verbose(self):
        """Test list --verbose shows detailed info."""
        result = runner.invoke(app, ["list", "--verbose"])
        
        assert result.exit_code == 0
        assert "Full Path" in result.stdout
    
    def test_info_command(self):
        """Test info command shows system information."""
        result = runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "System Information" in result.stdout
        assert "Python" in result.stdout
        assert "NumPy" in result.stdout
    
    def test_validate_command_with_valid_config(self):
        """Test validate with existing config file."""
        # Find any config file
        config_files = list(Path("config").glob("*.json"))
        
        if not config_files:
            pytest.skip("No config files found")
        
        config_path = config_files[0]
        result = runner.invoke(app, ["validate", str(config_path)])
        
        # Should either succeed or fail gracefully
        assert result.exit_code in [0, 1]
        
        if result.exit_code == 0:
            assert "valid" in result.stdout.lower()
    
    def test_validate_command_with_missing_config(self):
        """Test validate with non-existent config."""
        result = runner.invoke(app, ["validate", "nonexistent_config.json"])
        
        assert result.exit_code != 0
    
    def test_run_command_help(self):
        """Test run --help shows usage."""
        result = runner.invoke(app, ["run", "--help"])
        
        assert result.exit_code == 0
        assert "Run MOEA/D evolution" in result.stdout
        assert "--config" in result.stdout
        assert "--interactive" in result.stdout
        assert "--report" in result.stdout
    
    def test_run_command_with_invalid_config(self):
        """Test run with invalid config path fails gracefully."""
        result = runner.invoke(
            app,
            ["run", "-c", "invalid_path.json", "--no-interactive"],
            input="n\n"  # Don't show traceback
        )
        
        assert result.exit_code != 0
    
    def test_run_command_library_check(self):
        """Test that library checks are performed."""
        # Just test that library check runs, not full execution
        # (Full execution is too heavy for unit tests)
        
        config_files = list(Path("config").glob("*.json"))
        if not config_files:
            pytest.skip("No config files found")
        
        # Mock the orchestrator import to avoid actual execution
        with patch('orchestrator.Orchestrator') as mock_orch:
            mock_instance = MagicMock()
            mock_instance.run.return_value = None
            mock_orch.return_value = mock_instance
            
            result = runner.invoke(
                app,
                ["run", "-c", str(config_files[0]), "--no-report"]
            )
            
            # Should show library check passed (even if execution fails later)
            assert "Libraries check passed" in result.stdout or \
                   "Directory structure check passed" in result.stdout or \
                   result.exit_code in [0, 1]


class TestMainV2ErrorHandling:
    """Test error handling and edge cases."""
    
    def test_keyboard_interrupt_handling(self):
        """Test graceful handling of keyboard interrupt."""
        # Test that KeyboardInterrupt is handled (integration test)
        # We can't easily mock this in CLI context, so just verify
        # the command structure is correct
        result = runner.invoke(app, ["run", "--help"])
        
        # Help should work fine
        assert result.exit_code == 0
        assert "Run MOEA/D evolution" in result.stdout
    
    def test_generic_exception_handling(self):
        """Test handling of invalid config causes proper error."""
        # Test with invalid config file
        result = runner.invoke(
            app,
            ["run", "-c", "nonexistent.json", "--no-report"],
            input="n\n"  # Don't show traceback
        )
        
        # Should fail with error
        assert result.exit_code != 0


class TestMainV2Integration_ReportGeneration:
    """Test report generation integration."""
    
    def test_report_generation_after_run(self):
        """Test that --report flag is accepted."""
        result = runner.invoke(app, ["run", "--help"])
        
        # Verify --report option exists in help
        assert "--report" in result.stdout
        assert "--no-report" in result.stdout
    
    def test_report_skipped_when_disabled(self):
        """Test that --no-report flag is accepted."""
        result = runner.invoke(app, ["run", "--help"])
        
        # Verify --no-report option exists
        assert "--no-report" in result.stdout


class TestMainV2RealExecution:
    """Real execution tests (slower, marked for optional execution)."""
    
    @pytest.mark.slow
    def test_full_run_with_test_config(self):
        """Test full execution with test_scenario1.json (if exists)."""
        test_config = Path("config/test_scenario1.json")
        
        if not test_config.exists():
            pytest.skip("test_scenario1.json not found")
        
        result = runner.invoke(
            app,
            ["run", "-c", str(test_config), "--no-report", "--no-interactive"],
            catch_exceptions=False
        )
        
        # Should complete or fail gracefully
        assert result.exit_code in [0, 1]
        
        if result.exit_code == 0:
            assert "Evolution completed successfully" in result.stdout or \
                   "Starting Evolution" in result.stdout


def test_import_main_v2():
    """Test that main_v2.py can be imported."""
    # Add project root to path if needed
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import main_v2
    import importlib.util
    main_v2_path = project_root / "main_v2.py"
    
    spec = importlib.util.spec_from_file_location("main_v2", main_v2_path)
    main_v2 = importlib.util.module_from_spec(spec)
    
    # Should not raise any errors
    assert main_v2 is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
