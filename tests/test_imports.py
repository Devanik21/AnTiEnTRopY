import pytest

def test_imports():
    try:
        import CloCk
        import EnTRopY
        import HRF_EpIgEnEtIc
        import ImMoRtAlItY
        import ReVeRsAL
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
