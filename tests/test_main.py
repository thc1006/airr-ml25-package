#!/usr/bin/env python3
"""
Tests for main.py entry point
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import main


class TestEnvironmentCheck:
    """Test environment checking functionality."""

    def test_check_environment_all_packages_installed(self, capsys):
        """Test that check_environment succeeds when all packages are installed."""
        # This should not raise SystemExit
        main.check_environment()
        captured = capsys.readouterr()
        assert "✅ All required packages are installed" in captured.out

    @patch('main.__import__')
    def test_check_environment_missing_packages(self, mock_import, capsys):
        """Test that check_environment fails when packages are missing."""
        # Simulate missing pandas
        def side_effect(name):
            if name == 'pandas':
                raise ImportError(f"No module named '{name}'")
            return MagicMock()

        mock_import.side_effect = side_effect

        with pytest.raises(SystemExit) as exc_info:
            main.check_environment()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Missing required packages" in captured.out
        assert "pandas" in captured.out


class TestDataCheck:
    """Test data checking functionality."""

    def test_check_data_missing_directories(self, capsys, tmp_path, monkeypatch):
        """Test that check_data fails when data directories are missing."""
        # Change to temp directory where data doesn't exist
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            main.check_data()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Missing competition data" in captured.out
        assert "data/train_datasets" in captured.out

    def test_check_data_success(self, tmp_path, monkeypatch, capsys):
        """Test that check_data succeeds when data exists."""
        # Create mock data structure
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'data' / 'train_datasets' / 'train_datasets').mkdir(parents=True)
        (tmp_path / 'data' / 'test_datasets' / 'test_datasets').mkdir(parents=True)
        (tmp_path / 'data' / 'sample_submissions.csv').touch()

        # Should not raise SystemExit
        main.check_data()
        captured = capsys.readouterr()
        assert "✅ Competition data found" in captured.out


class TestMainFunction:
    """Test main entry point."""

    @patch('main.check_environment')
    @patch('main.check_data')
    @patch('main.champion_main')
    def test_main_success(self, mock_champion, mock_check_data, mock_check_env, capsys):
        """Test that main function runs successfully."""
        # Mock the champion_v8 import
        with patch.dict('sys.modules', {'src.champion_v8': MagicMock(main=mock_champion)}):
            main.main()

        # Verify all checks were called
        mock_check_env.assert_called_once()
        mock_check_data.assert_called_once()
        mock_champion.assert_called_once()

        captured = capsys.readouterr()
        assert "Champion V8 Pipeline" in captured.out
        assert "Pipeline completed successfully" in captured.out

    @patch('main.check_environment')
    @patch('main.check_data')
    def test_main_import_error(self, mock_check_data, mock_check_env, capsys):
        """Test that main handles import errors gracefully."""
        # Simulate import error
        with patch.dict('sys.modules', {'src.champion_v8': None}):
            with pytest.raises(SystemExit) as exc_info:
                main.main()

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Error running pipeline" in captured.out


class TestRequiredPackages:
    """Test that all required packages are actually importable."""

    @pytest.mark.parametrize('package', [
        'pandas', 'numpy', 'scikit-learn', 'scipy',
        'xgboost', 'lightgbm', 'catboost', 'tqdm'
    ])
    def test_package_importable(self, package):
        """Test that each required package can be imported."""
        # Special handling for scikit-learn
        import_name = 'sklearn' if package == 'scikit-learn' else package

        try:
            __import__(import_name)
        except ImportError:
            pytest.skip(f"Package {package} not installed")


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_readme_quick_start_commands(self):
        """Test that README quick start commands are valid."""
        # Verify that main.py exists and is executable
        main_path = Path(__file__).parent.parent / 'main.py'
        assert main_path.exists(), "main.py should exist"

        # Verify requirements.txt exists
        req_path = Path(__file__).parent.parent / 'requirements.txt'
        assert req_path.exists(), "requirements.txt should exist"

    def test_project_structure(self):
        """Test that required project structure exists."""
        root = Path(__file__).parent.parent

        # Check required directories
        assert (root / 'src').exists(), "src/ directory should exist"
        assert (root / 'submissions').exists(), "submissions/ directory should exist"

        # Check required files
        assert (root / 'main.py').exists(), "main.py should exist"
        assert (root / 'requirements.txt').exists(), "requirements.txt should exist"
        assert (root / 'README.md').exists(), "README.md should exist"

        # Check src files
        assert (root / 'src' / 'champion_v8.py').exists(), "champion_v8.py should exist"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
