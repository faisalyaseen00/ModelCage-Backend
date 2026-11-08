import pytest

from app import policy


@pytest.fixture
def hashing_enabled(monkeypatch):
    """Exercise SHA-256 identification independently of the deployed hashing policy."""
    monkeypatch.setattr(policy, "MODEL_HASHING_ENABLED", True)
