import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    config = Mock()
    config.batch_size = 32
    config.learning_rate = 0.001
    config.epochs = 10
    config.device = "cpu"
    config.data_path = "/tmp/test_data"
    config.output_path = "/tmp/test_output"
    return config


@pytest.fixture
def sample_json_data():
    """Provide sample JSON data for testing."""
    return {
        "videos": [
            {
                "id": "test_video_1",
                "label": "hello",
                "path": "/path/to/video1.mp4"
            },
            {
                "id": "test_video_2",
                "label": "world",
                "path": "/path/to/video2.mp4"
            }
        ],
        "labels": ["hello", "world", "sign", "language"]
    }


@pytest.fixture
def mock_video_file(temp_dir):
    """Create a mock video file in the temporary directory."""
    video_path = temp_dir / "test_video.mp4"
    video_path.write_bytes(b"mock video content")
    return video_path


@pytest.fixture
def mock_dataset_dir(temp_dir):
    """Create a mock dataset directory structure."""
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()
    
    # Create subdirectories
    (dataset_dir / "videos").mkdir()
    (dataset_dir / "annotations").mkdir()
    (dataset_dir / "features").mkdir()
    
    # Create some mock files
    (dataset_dir / "videos" / "sample1.mp4").write_bytes(b"video1")
    (dataset_dir / "videos" / "sample2.mp4").write_bytes(b"video2")
    (dataset_dir / "annotations" / "labels.json").write_text(
        json.dumps({"sample1": "hello", "sample2": "world"})
    )
    
    return dataset_dir


@pytest.fixture
def mock_model():
    """Provide a mock model object."""
    model = Mock()
    model.train = Mock(return_value=None)
    model.eval = Mock(return_value=None)
    model.forward = Mock(return_value=Mock(shape=(32, 10)))
    model.parameters = Mock(return_value=[Mock()])
    return model


@pytest.fixture
def mock_data_loader():
    """Provide a mock data loader."""
    loader = Mock()
    loader.__iter__ = Mock(return_value=iter([
        (Mock(shape=(32, 3, 224, 224)), Mock(shape=(32,))),
        (Mock(shape=(32, 3, 224, 224)), Mock(shape=(32,)))
    ]))
    loader.__len__ = Mock(return_value=2)
    return loader


@pytest.fixture
def env_setup(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent))
    

@pytest.fixture
def capture_logs():
    """Capture log messages during tests."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Get root logger
    logger = logging.getLogger()
    original_handlers = logger.handlers[:]
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    # Restore original handlers
    logger.handlers = original_handlers


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically clean up any test files created during tests."""
    test_files = []
    yield test_files
    for file_path in test_files:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


@pytest.fixture
def mock_ini_config(temp_dir):
    """Create a mock INI configuration file."""
    config_path = temp_dir / "test_config.ini"
    config_content = """[DEFAULT]
batch_size = 32
learning_rate = 0.001
epochs = 10

[paths]
data_dir = /tmp/test_data
output_dir = /tmp/test_output

[model]
num_classes = 100
pretrained = True
"""
    config_path.write_text(config_content)
    return config_path