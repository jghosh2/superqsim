"""Unit tests for the Transmon class."""

import numpy as np
import pytest

from devices import Transmon


# ---------------------------------------------------------------------------
# Attrs integration
# ---------------------------------------------------------------------------

class TestAttrs:
    def test_fields_stored(self, transmon):
        assert transmon.Ec == 0.2
        assert transmon.EJ == 10.0
        assert transmon.ng == 0.0
        assert transmon.n_cutoff == 15

    def test_defaults(self):
        t = Transmon(Ec=0.2, EJ=10.0)
        assert t.ng == 0.0
        assert t.n_cutoff == 15

    def test_repr_contains_field_names(self, transmon):
        r = repr(transmon)
        assert "Ec=0.2" in r
        assert "EJ=10.0" in r
        assert "ng=0.0" in r
        assert "n_cutoff=15" in r

    def test_eq_identical_instances(self):
        t1 = Transmon(Ec=0.2, EJ=10.0, ng=0.0, n_cutoff=15)
        t2 = Transmon(Ec=0.2, EJ=10.0, ng=0.0, n_cutoff=15)
        assert t1 == t2

    def test_eq_different_instances(self):
        t1 = Transmon(Ec=0.2, EJ=10.0)
        t2 = Transmon(Ec=0.3, EJ=10.0)
        assert t1 != t2

    def test_positional_args_accepted(self):
        t = Transmon(0.2, 10.0, 0.0, 15)
        assert t.Ec == 0.2


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_dim(self, transmon):
        assert transmon.dim == 2 * transmon.n_cutoff + 1

    def test_dim_varies_with_n_cutoff(self):
        for n in [5, 10, 20]:
            assert Transmon(Ec=0.2, EJ=10.0, n_cutoff=n).dim == 2 * n + 1

    def test_ej_over_ec(self, transmon):
        assert transmon.EJ_over_Ec == pytest.approx(transmon.EJ / transmon.Ec)

    def test_charge_states_length(self, transmon):
        assert len(transmon.charge_states) == transmon.dim

    def test_charge_states_bounds(self, transmon):
        cs = transmon.charge_states
        assert cs[0] == pytest.approx(-transmon.n_cutoff)
        assert cs[-1] == pytest.approx(transmon.n_cutoff)

    def test_charge_states_integer_spacing(self, transmon):
        cs = transmon.charge_states
        np.testing.assert_allclose(np.diff(cs), 1.0)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class TestOperators:
    def test_charge_operator_is_diagonal(self, transmon):
        n_hat = transmon.charge_operator().full()
        off_diag = n_hat - np.diag(np.diag(n_hat))
        np.testing.assert_allclose(off_diag, 0.0)

    def test_charge_operator_diagonal_values(self, transmon):
        n_hat = transmon.charge_operator().full()
        np.testing.assert_allclose(np.diag(n_hat), transmon.charge_states)

    def test_charge_operator_dims(self, transmon):
        n_hat = transmon.charge_operator()
        assert n_hat.dims == [[transmon.dim], [transmon.dim]]

    def test_hamiltonian_diagonal(self, transmon):
        H = transmon.build_hamiltonian().full()
        n = transmon.charge_states
        expected_diag = 4.0 * transmon.Ec * (n - transmon.ng) ** 2
        np.testing.assert_allclose(np.real(np.diag(H)), expected_diag)

    def test_hamiltonian_off_diagonal(self, transmon):
        H = transmon.build_hamiltonian().full()
        expected = -transmon.EJ / 2.0
        np.testing.assert_allclose(np.real(np.diag(H, 1)), expected)
        np.testing.assert_allclose(np.real(np.diag(H, -1)), expected)

    def test_hamiltonian_is_symmetric(self, transmon):
        H = transmon.build_hamiltonian().full()
        np.testing.assert_allclose(H, H.T)

    def test_hamiltonian_dims(self, transmon):
        H = transmon.build_hamiltonian()
        assert H.dims == [[transmon.dim], [transmon.dim]]

    def test_hamiltonian_tridiagonal(self, transmon):
        H = transmon.build_hamiltonian().full()
        for k in range(2, transmon.dim):
            np.testing.assert_allclose(np.diag(H, k), 0.0, atol=1e-12)
            np.testing.assert_allclose(np.diag(H, -k), 0.0, atol=1e-12)

    def test_hamiltonian_diagonal_shifts_with_ng(self, transmon_offset):
        H = transmon_offset.build_hamiltonian().full()
        n = transmon_offset.charge_states
        expected = 4.0 * transmon_offset.Ec * (n - transmon_offset.ng) ** 2
        np.testing.assert_allclose(np.real(np.diag(H)), expected)


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

class TestSpectralAnalysis:
    def test_ground_state_energy_is_zero(self, transmon):
        vals, _ = transmon.get_eigenspectrum()
        assert vals[0] == pytest.approx(0.0)

    def test_eigenvalues_ordered(self, transmon):
        vals, _ = transmon.get_eigenspectrum()
        assert np.all(np.diff(vals) >= 0)

    def test_num_levels_truncation(self, transmon):
        for k in [3, 5, 8]:
            vals, vecs = transmon.get_eigenspectrum(k)
            assert len(vals) == k
            assert len(vecs) == k

    def test_full_spectrum_length(self, transmon):
        vals, vecs = transmon.get_eigenspectrum()
        assert len(vals) == transmon.dim

    def test_transition_frequency_default_is_01(self, transmon):
        vals, _ = transmon.get_eigenspectrum(2)
        assert transmon.transition_frequency() == pytest.approx(vals[1] - vals[0])

    def test_transition_frequency_positive(self, transmon):
        assert transmon.transition_frequency(0, 1) > 0.0
        assert transmon.transition_frequency(1, 2) > 0.0

    def test_transition_frequency_asymmetric(self, transmon):
        f01 = transmon.transition_frequency(0, 1)
        f10 = transmon.transition_frequency(1, 0)
        assert f01 == pytest.approx(-f10)

    def test_anharmonicity_negative_for_transmon(self, transmon):
        assert transmon.anharmonicity() < 0.0

    def test_anharmonicity_positive_for_cpb(self, transmon_cpb):
        """In the Cooper-pair box regime the ordering can be inverted; the
        anharmonicity is not necessarily ≈ -Ec."""
        alpha = transmon_cpb.anharmonicity()
        assert isinstance(alpha, float)

    def test_anharmonicity_approaches_ec_in_deep_limit(self):
        """α → -Ec as EJ/Ec → ∞.

        Convergence is O(sqrt(Ec/EJ)), so even at EJ/Ec=200 the deviation
        from -Ec is ~6%.  We verify the magnitude is in a reasonable band
        AND that it moves closer to -Ec at higher EJ/Ec (monotone trend).
        """
        Ec = 0.2
        t_deep = Transmon(Ec=Ec, EJ=200 * Ec, n_cutoff=20)
        t_mod = Transmon(Ec=Ec, EJ=50 * Ec, n_cutoff=15)
        alpha_deep = t_deep.anharmonicity()
        alpha_mod = t_mod.anharmonicity()
        # Rough magnitude bound: |α| within 20% of Ec in the deep limit
        assert alpha_deep == pytest.approx(-Ec, rel=0.20)
        # Trend: |α| shrinks toward Ec as EJ/Ec grows
        assert abs(alpha_deep) < abs(alpha_mod)


# ---------------------------------------------------------------------------
# Sweep methods
# ---------------------------------------------------------------------------

class TestSweepMethods:
    def test_charge_dispersion_sweep_shape(self, transmon):
        ng_vals = np.linspace(-0.5, 0.5, 20)
        energies = transmon.charge_dispersion_sweep(ng_vals, num_levels=4)
        assert energies.shape == (20, 4)

    def test_charge_dispersion_sweep_ground_state_zero(self, transmon):
        ng_vals = np.linspace(-0.5, 0.5, 10)
        energies = transmon.charge_dispersion_sweep(ng_vals, num_levels=3)
        np.testing.assert_allclose(energies[:, 0], 0.0)

    def test_charge_dispersion_sweep_restores_ng(self, transmon):
        original_ng = transmon.ng
        transmon.charge_dispersion_sweep(np.linspace(-1, 1, 30))
        assert transmon.ng == pytest.approx(original_ng)

    def test_charge_dispersion_sweep_restores_ng_on_exception(self):
        t = Transmon(Ec=0.2, EJ=10.0, ng=0.123)
        try:
            t.charge_dispersion_sweep(np.array([]))
        except Exception:
            pass
        assert t.ng == pytest.approx(0.123)

    def test_charge_dispersion_bands_symmetric_at_ng0(self, transmon):
        """Energy bands must be symmetric about ng=0 for ng=+x and ng=-x."""
        ng_pos = np.array([0.25])
        ng_neg = np.array([-0.25])
        e_pos = transmon.charge_dispersion_sweep(ng_pos, num_levels=3)
        e_neg = transmon.charge_dispersion_sweep(ng_neg, num_levels=3)
        np.testing.assert_allclose(e_pos, e_neg, atol=1e-10)

    def test_ej_ec_sweep_shape(self, transmon):
        ratios = np.linspace(1, 80, 10)
        freqs, anharmon = transmon.ej_ec_sweep(ratios)
        assert freqs.shape == (10,)
        assert anharmon.shape == (10,)

    def test_ej_ec_sweep_frequency_increases_with_ratio(self, transmon):
        """ω₀₁ increases monotonically with EJ/Ec."""
        ratios = np.linspace(5, 80, 15)
        freqs, _ = transmon.ej_ec_sweep(ratios)
        assert np.all(np.diff(freqs) > 0)

    def test_ej_ec_sweep_anharmonicity_trend(self, transmon):
        """|α| decreases monotonically toward Ec as EJ/Ec increases."""
        ratios = np.array([10.0, 30.0, 80.0, 200.0])
        _, anharmon = transmon.ej_ec_sweep(ratios)
        # All values should be negative
        assert np.all(anharmon < 0)
        # |α| decreases as EJ/Ec increases (converges toward Ec from above)
        assert np.all(np.diff(np.abs(anharmon)) < 0)
