import pytest
import sys
from pathlib import Path


class TestInfrastructureValidation:
    """Validation tests to ensure the testing infrastructure is properly set up."""
    
    @pytest.mark.unit
    def test_pytest_is_installed(self):
        """Verify pytest is installed and importable."""
        import pytest
        assert pytest.__version__
    
    @pytest.mark.unit
    def test_pytest_cov_is_installed(self):
        """Verify pytest-cov is installed and importable."""
        import pytest_cov
        assert pytest_cov
    
    @pytest.mark.unit
    def test_pytest_mock_is_installed(self):
        """Verify pytest-mock is installed and importable."""
        import pytest_mock
        assert pytest_mock
    
    @pytest.mark.unit
    def test_project_root_in_pythonpath(self):
        """Verify the project root is accessible."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / "code").exists()
    
    @pytest.mark.unit
    def test_conftest_fixtures_available(self, temp_dir, mock_config):
        """Verify conftest fixtures are available."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert mock_config is not None
        assert hasattr(mock_config, "batch_size")
    
    @pytest.mark.unit
    def test_markers_are_registered(self, pytestconfig):
        """Verify custom markers are properly registered."""
        markers = pytestconfig.getini("markers")
        marker_names = [marker.split(":")[0].strip() for marker in markers]
        assert "unit" in marker_names
        assert "integration" in marker_names
        assert "slow" in marker_names
    
    @pytest.mark.integration
    def test_coverage_configuration(self):
        """Verify coverage is properly configured."""
        from coverage import Coverage
        cov = Coverage()
        config = cov.config
        assert config is not None
    
    @pytest.mark.unit
    def test_temp_dir_fixture_cleanup(self, temp_dir):
        """Verify temp_dir fixture creates and cleans up properly."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        # Cleanup happens after test completes
    
    @pytest.mark.unit
    def test_mock_fixtures_work(self, mock_model, mock_data_loader):
        """Verify mock fixtures are functional."""
        assert mock_model is not None
        assert callable(mock_model.train)
        assert callable(mock_model.eval)
        
        assert mock_data_loader is not None
        assert hasattr(mock_data_loader, "__iter__")
        assert hasattr(mock_data_loader, "__len__")
    
    @pytest.mark.unit 
    def test_json_fixture(self, sample_json_data):
        """Verify JSON fixture provides expected data."""
        assert "videos" in sample_json_data
        assert "labels" in sample_json_data
        assert len(sample_json_data["videos"]) == 2
        assert len(sample_json_data["labels"]) == 4
    
    @pytest.mark.unit
    def test_capture_logs_fixture(self, capture_logs):
        """Verify log capture fixture works."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Test log message")
        
        log_output = capture_logs.getvalue()
        assert "Test log message" in log_output
    
    @pytest.mark.slow
    @pytest.mark.integration 
    def test_mock_dataset_dir_fixture(self, mock_dataset_dir):
        """Verify mock dataset directory structure is created properly."""
        assert mock_dataset_dir.exists()
        assert (mock_dataset_dir / "videos").exists()
        assert (mock_dataset_dir / "annotations").exists()
        assert (mock_dataset_dir / "features").exists()
        
        videos = list((mock_dataset_dir / "videos").glob("*.mp4"))
        assert len(videos) == 2
        
        labels_file = mock_dataset_dir / "annotations" / "labels.json"
        assert labels_file.exists()
        
        import json
        labels = json.loads(labels_file.read_text())
        assert "sample1" in labels
        assert "sample2" in labels