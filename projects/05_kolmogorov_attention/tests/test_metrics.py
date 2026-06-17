import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.analysis.complexity import (compression_ratio, attention_entropy,
                                     attention_gini, attention_sparsity)


def test_compression_low_vs_high():
    low = "a" * 1000
    high = "".join(chr(0x41 + (i * 13) % 26) for i in range(1000))
    # high-entropy text should not be much more compressible than the low-entropy one
    assert compression_ratio(low) < compression_ratio(high)


def test_uniform_attention_high_entropy():
    n = 64
    e_unif = attention_entropy(np.ones(n) / n)
    e_one = attention_entropy(np.eye(1, n)[0])
    assert e_unif > e_one


def test_gini_concentrated():
    n = 64
    g_unif = attention_gini(np.ones(n) / n)
    g_conc = attention_gini(np.eye(1, n)[0])
    assert g_conc > g_unif


def test_sparsity_concentrated_lower():
    n = 100
    s_unif = attention_sparsity(np.ones(n) / n)
    s_conc = attention_sparsity(np.array([0.99] + [0.01/(n-1)]*(n-1)))
    assert s_conc < s_unif
