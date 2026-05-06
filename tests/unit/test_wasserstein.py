"""Unit tests for spatialmt.eval.metrics — Wasserstein-1 transport.

RED targets
-----------
_transport_constraints  Does not exist yet → ImportError on collection.
wasserstein_1 (POT)     Uses scipy LP today; after implementation uses ot.emd2.
                        Correctness tests are regression guards that stay GREEN
                        across both backends. The cache tests are RED until
                        _transport_constraints is added.

Test taxonomy per function
--------------------------
_transport_constraints:
  happy path    — correct shape and entry values for K=2, K=3
  boundary      — K=1 degenerate (single-class transport)
  cache         — same K returns identical object; different K returns distinct object

wasserstein_1:
  happy path    — three analytically known EMD values (K=2 and K=3)
  boundary      — zero cost matrix, K=1, distribution symmetry
  edge cases    — float32 inputs, slightly off-sum float32 (the normalization bug)
  failure       — all-zero distribution, mismatched distribution lengths
"""
import math

import numpy as np
import pytest

from spatialmt.eval.metrics import _transport_constraints, wasserstein_1


# ---------------------------------------------------------------------------
# _transport_constraints
# ---------------------------------------------------------------------------

class TestTransportConstraints:

    # --- happy path ---

    def test_shape_k2(self):
        assert _transport_constraints(2).shape == (4, 4)

    def test_shape_k3(self):
        assert _transport_constraints(3).shape == (6, 9)

    def test_each_row_sums_to_k(self):
        for K in (2, 3, 5):
            A = _transport_constraints(K)
            np.testing.assert_allclose(A.sum(axis=1), K,
                                       err_msg=f"K={K}: row sums != {K}")

    def test_source_marginal_rows_are_row_blocks(self):
        """First K rows must select consecutive row-blocks [i*K:(i+1)*K] of flattened T."""
        K = 3
        A = _transport_constraints(K)
        for i in range(K):
            expected = np.zeros(K * K)
            expected[i * K:(i + 1) * K] = 1.0
            np.testing.assert_array_equal(A[i], expected,
                                          err_msg=f"source row {i} wrong")

    def test_target_marginal_rows_are_column_strides(self):
        """Last K rows must select column-strides j::K of flattened T."""
        K = 3
        A = _transport_constraints(K)
        for j in range(K):
            expected = np.zeros(K * K)
            expected[j::K] = 1.0
            np.testing.assert_array_equal(A[K + j], expected,
                                          err_msg=f"target row {j} wrong")

    # --- boundary ---

    def test_k1_degenerate_shape_and_values(self):
        A = _transport_constraints(1)
        assert A.shape == (2, 1)
        np.testing.assert_array_equal(A, np.ones((2, 1)))

    # --- cache identity ---

    def test_same_k_returns_cached_object(self):
        a = _transport_constraints(6)
        b = _transport_constraints(6)
        assert a is b, "_transport_constraints(K) must return the same object on repeated calls"

    def test_different_k_returns_distinct_objects(self):
        a = _transport_constraints(2)
        b = _transport_constraints(7)
        assert a is not b


# ---------------------------------------------------------------------------
# wasserstein_1
# ---------------------------------------------------------------------------

class TestWasserstein1:

    # --- happy path: analytically known EMD values ---

    def test_identical_distributions_zero_emd(self):
        p = np.array([0.3, 0.5, 0.2])
        M = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float64)
        assert wasserstein_1(p, p.copy(), M) == pytest.approx(0.0, abs=1e-9)

    def test_unit_mass_k2(self):
        """All mass moves from class 0 → class 1; cost = 1."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        M = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert wasserstein_1(p, q, M) == pytest.approx(1.0, abs=1e-9)

    def test_partial_transport_k2(self):
        """Move 0.4 units across cost=2 gap; EMD = 0.8."""
        p = np.array([0.7, 0.3])
        q = np.array([0.3, 0.7])
        M = np.array([[0.0, 2.0], [2.0, 0.0]])
        assert wasserstein_1(p, q, M) == pytest.approx(0.8, abs=1e-9)

    def test_k3_endpoints_geodesic_cost(self):
        """All mass from class 0 → class 2 via geodesic distance 2."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        M = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float64)
        assert wasserstein_1(p, q, M) == pytest.approx(2.0, abs=1e-9)

    # --- boundary ---

    def test_zero_cost_matrix_gives_zero(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.6, 0.2])
        M = np.zeros((3, 3))
        assert wasserstein_1(p, q, M) == pytest.approx(0.0, abs=1e-9)

    def test_k1_degenerate(self):
        assert wasserstein_1(np.array([1.0]), np.array([1.0]), np.array([[0.0]])) \
               == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_in_p_q(self):
        """EMD(p, q, M) == EMD(q, p, M) for a symmetric cost matrix."""
        p = np.array([0.6, 0.1, 0.3])
        q = np.array([0.1, 0.7, 0.2])
        M = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=np.float64)
        assert wasserstein_1(p, q, M) == pytest.approx(wasserstein_1(q, p, M), abs=1e-9)

    # --- edge cases ---

    def test_float32_inputs_accepted(self):
        p = np.array([0.6, 0.4], dtype=np.float32)
        q = np.array([0.4, 0.6], dtype=np.float32)
        M = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        result = wasserstein_1(p, q, M)
        assert math.isfinite(result)

    def test_slightly_off_sum_float32_does_not_crash(self):
        """float32 soft labels from the model may not sum to exactly 1 — must not crash."""
        p = np.array([0.3333334, 0.3333333, 0.3333334], dtype=np.float32)
        q = np.array([0.5000001, 0.2999999, 0.2000000], dtype=np.float32)
        M = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float64)
        result = wasserstein_1(p, q, M)
        assert math.isfinite(result)

    # --- failure / invalid inputs ---

    def test_all_zero_distribution_raises(self):
        """A distribution of all zeros has no valid normalization."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([0.3, 0.4, 0.3])
        M = np.ones((3, 3)) - np.eye(3)
        with pytest.raises(Exception):
            wasserstein_1(p, q, M)

    def test_mismatched_distribution_lengths_raises(self):
        """len(p) != len(q) is incoherent for a transport problem."""
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.4, 0.3])
        M = np.ones((3, 3))
        with pytest.raises(Exception):
            wasserstein_1(p, q, M)
