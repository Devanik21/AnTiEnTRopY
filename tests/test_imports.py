import pytest

def test_imports():
    try:
        import clock
        import entropy
        import hrf_epigenetic
        import immortality
        import reversal
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
