"""Unit tests for the TransmonResonator class."""

import numpy as np
import pytest

from devices import Transmon, TransmonResonator


# ---------------------------------------------------------------------------
# Attrs integration
# ---------------------------------------------------------------------------

class TestAttrs:
    def test_fields_stored(self, transmon_resonator):
        assert transmon_resonator.Ec == 0.2
        assert transmon_resonator.EJ == 10.0
        assert transmon_resonator.omega_r == 6.0
        assert transmon_resonator.g == 0.1
        assert transmon_resonator.ng == 0.0
        assert transmon_resonator.n_cutoff == 15
        assert transmon_resonator.n_fock == 10

    def test_defaults(self):
        dev = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1)
        assert dev.ng == 0.0
        assert dev.n_cutoff == 15
        assert dev.n_fock == 10

    def test_repr_contains_field_names(self, transmon_resonator):
        r = repr(transmon_resonator)
        assert "Ec=0.2" in r
        assert "omega_r=6.0" in r
        assert "g=0.1" in r

    def test_repr_excludes_transmon_object(self, transmon_resonator):
        """_transmon is an internal derived field; repr=False keeps repr clean."""
        r = repr(transmon_resonator)
        assert "_transmon" not in r

    def test_eq_identical_instances(self):
        d1 = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1)
        d2 = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1)
        assert d1 == d2

    def test_eq_different_instances(self):
        d1 = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1)
        d2 = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=5.0, g=0.1)
        assert d1 != d2


# ---------------------------------------------------------------------------
# post_init — internal Transmon is created correctly
# ---------------------------------------------------------------------------

class TestPostInit:
    def test_transmon_property_returns_transmon(self, transmon_resonator):
        assert isinstance(transmon_resonator.transmon, Transmon)

    def test_transmon_ec_matches(self, transmon_resonator):
        assert transmon_resonator.transmon.Ec == transmon_resonator.Ec

    def test_transmon_ej_matches(self, transmon_resonator):
        assert transmon_resonator.transmon.EJ == transmon_resonator.EJ

    def test_transmon_ng_matches(self, transmon_resonator):
        assert transmon_resonator.transmon.ng == transmon_resonator.ng

    def test_transmon_n_cutoff_matches(self, transmon_resonator):
        assert transmon_resonator.transmon.n_cutoff == transmon_resonator.n_cutoff


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_dim(self, transmon_resonator):
        expected = transmon_resonator.transmon.dim * transmon_resonator.n_fock
        assert transmon_resonator.dim == expected

    def test_dim_varies_with_n_fock(self):
        d1 = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1, n_fock=5)
        d2 = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1, n_fock=15)
        assert d1.dim < d2.dim

    def test_dispersive_shift_matches_formula(self, transmon_resonator):
        omega_01 = transmon_resonator.transmon.transition_frequency(0, 1)
        alpha = transmon_resonator.transmon.anharmonicity()
        Delta = omega_01 - transmon_resonator.omega_r
        expected = (transmon_resonator.g ** 2 * alpha) / (Delta * (Delta + alpha))
        assert transmon_resonator.dispersive_shift == pytest.approx(expected)

    def test_dispersive_shift_negative_when_qubit_below_resonator(self, transmon_resonator):
        """ω₀₁ < ω_r → Δ < 0 → χ < 0 for α < 0."""
        omega_01 = transmon_resonator.transmon.transition_frequency(0, 1)
        assert omega_01 < transmon_resonator.omega_r
        assert transmon_resonator.dispersive_shift < 0.0

    def test_dispersive_shift_negative_regardless_of_delta_sign(self):
        """For a transmon (α < 0) in the dispersive limit (|Δ| >> g),
        χ = g²α / (Δ(Δ+α)) is negative whether Δ > 0 or Δ < 0,
        because α is much smaller in magnitude than |Δ|, so Δ+α
        retains the sign of Δ."""
        # qubit well below resonator (Δ < 0)
        d_below = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1)
        assert d_below.dispersive_shift < 0.0
        # qubit well above resonator (Δ > 0, |Δ| >> |α|)
        d_above = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=2.0, g=0.1)
        omega_01 = d_above.transmon.transition_frequency(0, 1)
        assert omega_01 > d_above.omega_r          # confirm qubit is above
        assert d_above.dispersive_shift < 0.0      # still negative

    def test_dispersive_shift_scales_with_g_squared(self):
        d1 = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1)
        d2 = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.2)
        assert d2.dispersive_shift == pytest.approx(4.0 * d1.dispersive_shift, rel=1e-6)


# ---------------------------------------------------------------------------
# Hamiltonian and eigenspectrum
# ---------------------------------------------------------------------------

class TestHamiltonianAndSpectrum:
    def test_hamiltonian_is_qobj(self, transmon_resonator):
        import qutip as qt
        H = transmon_resonator.build_hamiltonian()
        assert isinstance(H, qt.Qobj)

    def test_hamiltonian_dimension(self, transmon_resonator):
        H = transmon_resonator.build_hamiltonian()
        n = transmon_resonator.dim
        assert H.shape == (n, n)

    def test_hamiltonian_is_hermitian(self, transmon_resonator):
        H = transmon_resonator.build_hamiltonian().full()
        np.testing.assert_allclose(H, H.conj().T, atol=1e-10)

    def test_eigenspectrum_ground_state_zero(self, transmon_resonator):
        vals, _ = transmon_resonator.get_eigenspectrum(4)
        assert vals[0] == pytest.approx(0.0)

    def test_eigenspectrum_ordered(self, transmon_resonator):
        vals, _ = transmon_resonator.get_eigenspectrum(6)
        assert np.all(np.diff(vals) >= 0)

    def test_eigenspectrum_num_levels(self, transmon_resonator):
        for k in [3, 6, 9]:
            vals, vecs = transmon_resonator.get_eigenspectrum(k)
            assert len(vals) == k
            assert len(vecs) == k


# ---------------------------------------------------------------------------
# Resonator frequency sweep
# ---------------------------------------------------------------------------

class TestResonatorFrequencySweep:
    def test_shape(self, transmon_resonator):
        omega_r_vals = np.linspace(3.0, 7.0, 20)
        energies = transmon_resonator.resonator_frequency_sweep(omega_r_vals, num_levels=5)
        assert energies.shape == (20, 5)

    def test_ground_state_column_is_zero(self, transmon_resonator):
        omega_r_vals = np.linspace(3.0, 7.0, 10)
        energies = transmon_resonator.resonator_frequency_sweep(omega_r_vals, num_levels=4)
        np.testing.assert_allclose(energies[:, 0], 0.0)

    def test_omega_r_restored_after_sweep(self, transmon_resonator):
        original = transmon_resonator.omega_r
        transmon_resonator.resonator_frequency_sweep(np.linspace(3.0, 7.0, 15))
        assert transmon_resonator.omega_r == pytest.approx(original)

    def test_omega_r_restored_on_exception(self):
        dev = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=4.567, g=0.1)
        try:
            dev.resonator_frequency_sweep(np.array([]))
        except Exception:
            pass
        assert dev.omega_r == pytest.approx(4.567)

    def test_vacuum_rabi_avoided_crossing_exists(self, transmon_resonator):
        """Near ω_r = ω₀₁ the first two excited levels should have a gap."""
        omega_01 = transmon_resonator.transmon.transition_frequency(0, 1)
        omega_r_near = np.linspace(omega_01 - 0.3, omega_01 + 0.3, 60)
        energies = transmon_resonator.resonator_frequency_sweep(omega_r_near, num_levels=3)
        gap = energies[:, 2] - energies[:, 1]
        assert np.min(gap) > 0.0
