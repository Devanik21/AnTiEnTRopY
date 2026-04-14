import numpy as np
from EnTRopY  import EpigeneticEntropy

def test_entropy_initialization():
    entropy_engine = EpigeneticEntropy()
    assert entropy_engine is not None

def test_binary_entropy():
    from entropy import _binary_entropy

    # Test completely unmethylated/methylated (should be 0 entropy)
    assert np.isclose(_binary_entropy(np.array([0.0]))[0], 0.0, atol=1e-5)
    assert np.isclose(_binary_entropy(np.array([1.0]))[0], 0.0, atol=1e-5)

    # Test maximum entropy at 0.5
    assert np.isclose(_binary_entropy(np.array([0.5]))[0], 1.0, atol=1e-5)

def test_get_sample_entropy_at():
    entropy_engine = EpigeneticEntropy()

    # Vector of betas at 0.5 should give max system entropy
    betas_max = np.array([0.5, 0.5, 0.5, 0.5])
    metrics_max = entropy_engine.get_sample_entropy_at(betas_max)
    assert np.isclose(metrics_max['mean_entropy'], 1.0, atol=1e-5)

    # Vector of betas at 0.0 or 1.0 should give min system entropy
    betas_min = np.array([0.0, 1.0, 0.0, 1.0])
    metrics_min = entropy_engine.get_sample_entropy_at(betas_min)
    assert np.isclose(metrics_min['mean_entropy'], 0.0, atol=1e-5)
