"""Unit tests for the TunableCouplerSystem class."""

import numpy as np
import pytest

from superqsim import Transmon, TunableCouplerSystem


# ---------------------------------------------------------------------------
# Attrs integration
# ---------------------------------------------------------------------------

class TestAttrs:
    def test_fields_stored(self, tunable_coupler):
        assert tunable_coupler.Ec_A == 0.20
        assert tunable_coupler.EJ_A == 10.0
        assert tunable_coupler.ng_A == 0.0
        assert tunable_coupler.Ec_B == 0.20
        assert tunable_coupler.EJ_B == 9.5
        assert tunable_coupler.ng_B == 0.0
        assert tunable_coupler.Ec_C == 0.30
        assert tunable_coupler.EJ_max_C == 18.0
        assert tunable_coupler.ng_C == 0.0
        assert tunable_coupler.g_AC == 0.10
        assert tunable_coupler.g_BC == 0.10
        assert tunable_coupler.n_cutoff == 15
        assert tunable_coupler.n_levels_A == 3
        assert tunable_coupler.n_levels_B == 3
        assert tunable_coupler.n_levels_C == 3

    def test_defaults(self):
        sys = TunableCouplerSystem(
            Ec_A=0.2, EJ_A=10.0,
            Ec_B=0.2, EJ_B=9.5,
            Ec_C=0.3, EJ_max_C=18.0,
        )
        assert sys.ng_A == 0.0
        assert sys.ng_B == 0.0
        assert sys.ng_C == 0.0
        assert sys.g_AC == 0.1
        assert sys.g_BC == 0.1
        assert sys.n_cutoff == 15
        assert sys.n_levels_A == 3
        assert sys.n_levels_B == 3
        assert sys.n_levels_C == 3

    def test_keyword_only_rejects_positional_args(self):
        with pytest.raises(TypeError):
            TunableCouplerSystem(0.2, 10.0, 0.2, 9.5, 0.3, 18.0)

    def test_repr_contains_field_names(self, tunable_coupler):
        r = repr(tunable_coupler)
        assert "Ec_A=0.2" in r
        assert "EJ_max_C=18.0" in r
        assert "g_AC=0.1" in r

    def test_repr_excludes_derived_transmons(self, tunable_coupler):
        """transmon_A and transmon_B are init=False, repr=False."""
        r = repr(tunable_coupler)
        assert "transmon_A" not in r
        assert "transmon_B" not in r

    def test_eq_identical_instances(self):
        kwargs = dict(Ec_A=0.2, EJ_A=10.0, Ec_B=0.2, EJ_B=9.5,
                      Ec_C=0.3, EJ_max_C=18.0)
        assert TunableCouplerSystem(**kwargs) == TunableCouplerSystem(**kwargs)

    def test_eq_different_instances(self):
        s1 = TunableCouplerSystem(Ec_A=0.2, EJ_A=10.0, Ec_B=0.2, EJ_B=9.5,
                                   Ec_C=0.3, EJ_max_C=18.0)
        s2 = TunableCouplerSystem(Ec_A=0.2, EJ_A=10.0, Ec_B=0.2, EJ_B=9.5,
                                   Ec_C=0.3, EJ_max_C=15.0)
        assert s1 != s2


# ---------------------------------------------------------------------------
# post_init — transmon_A and transmon_B are created correctly
# ---------------------------------------------------------------------------

class TestPostInit:
    def test_transmon_a_is_transmon(self, tunable_coupler):
        assert isinstance(tunable_coupler.transmon_A, Transmon)

    def test_transmon_b_is_transmon(self, tunable_coupler):
        assert isinstance(tunable_coupler.transmon_B, Transmon)

    def test_transmon_a_ec(self, tunable_coupler):
        assert tunable_coupler.transmon_A.Ec == tunable_coupler.Ec_A

    def test_transmon_a_ej(self, tunable_coupler):
        assert tunable_coupler.transmon_A.EJ == tunable_coupler.EJ_A

    def test_transmon_a_ng(self, tunable_coupler):
        assert tunable_coupler.transmon_A.ng == tunable_coupler.ng_A

    def test_transmon_a_n_cutoff(self, tunable_coupler):
        assert tunable_coupler.transmon_A.n_cutoff == tunable_coupler.n_cutoff

    def test_transmon_b_ec(self, tunable_coupler):
        assert tunable_coupler.transmon_B.Ec == tunable_coupler.Ec_B

    def test_transmon_b_ej(self, tunable_coupler):
        assert tunable_coupler.transmon_B.EJ == tunable_coupler.EJ_B

    def test_transmon_a_b_are_independent(self, tunable_coupler):
        assert tunable_coupler.transmon_A is not tunable_coupler.transmon_B


# ---------------------------------------------------------------------------
# coupler_EJ
# ---------------------------------------------------------------------------

class TestCouplerEJ:
    def test_at_zero_flux(self, tunable_coupler):
        assert tunable_coupler.coupler_EJ(0.0) == pytest.approx(tunable_coupler.EJ_max_C)

    def test_at_half_flux(self, tunable_coupler):
        assert tunable_coupler.coupler_EJ(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_at_quarter_flux(self, tunable_coupler):
        expected = tunable_coupler.EJ_max_C * abs(np.cos(np.pi * 0.25))
        assert tunable_coupler.coupler_EJ(0.25) == pytest.approx(expected)

    def test_monotonically_decreasing_from_0_to_half(self, tunable_coupler):
        flux_vals = np.linspace(0.0, 0.499, 50)
        ej_vals = np.array([tunable_coupler.coupler_EJ(f) for f in flux_vals])
        assert np.all(np.diff(ej_vals) <= 0)

    def test_non_negative(self, tunable_coupler):
        flux_vals = np.linspace(0.0, 1.0, 100)
        ej_vals = np.array([tunable_coupler.coupler_EJ(f) for f in flux_vals])
        assert np.all(ej_vals >= 0.0)

    def test_period_one(self, tunable_coupler):
        """EJ_coupler is periodic with period 1 in Φ/Φ₀."""
        assert tunable_coupler.coupler_EJ(0.0) == pytest.approx(
            tunable_coupler.coupler_EJ(1.0), abs=1e-10
        )


# ---------------------------------------------------------------------------
# Hamiltonian and eigenspectrum
# ---------------------------------------------------------------------------

class TestHamiltonianAndSpectrum:
    def test_hamiltonian_is_qobj(self, tunable_coupler):
        import qutip as qt
        H = tunable_coupler.build_hamiltonian(flux=0.0)
        assert isinstance(H, qt.Qobj)

    def test_hamiltonian_dimension(self, tunable_coupler):
        H = tunable_coupler.build_hamiltonian(flux=0.0)
        expected_dim = (
            tunable_coupler.n_levels_A
            * tunable_coupler.n_levels_B
            * tunable_coupler.n_levels_C
        )
        assert H.shape == (expected_dim, expected_dim)

    def test_hamiltonian_is_hermitian(self, tunable_coupler):
        H = tunable_coupler.build_hamiltonian(flux=0.0).full()
        np.testing.assert_allclose(H, H.conj().T, atol=1e-10)

    def test_hamiltonian_changes_with_flux(self, tunable_coupler):
        H0 = tunable_coupler.build_hamiltonian(flux=0.0).full()
        H1 = tunable_coupler.build_hamiltonian(flux=0.3).full()
        assert not np.allclose(H0, H1)

    def test_eigenspectrum_ground_state_zero(self, tunable_coupler):
        vals, _ = tunable_coupler.get_eigenspectrum(flux=0.0, num_levels=4)
        assert vals[0] == pytest.approx(0.0)

    def test_eigenspectrum_ordered(self, tunable_coupler):
        vals, _ = tunable_coupler.get_eigenspectrum(flux=0.0, num_levels=6)
        assert np.all(np.diff(vals) >= 0)

    def test_eigenspectrum_num_levels(self, tunable_coupler):
        for k in [3, 6]:
            vals, vecs = tunable_coupler.get_eigenspectrum(flux=0.0, num_levels=k)
            assert len(vals) == k
            assert len(vecs) == k

    def test_eigenspectrum_default_flux_zero(self, tunable_coupler):
        vals_explicit, _ = tunable_coupler.get_eigenspectrum(flux=0.0, num_levels=4)
        vals_default, _ = tunable_coupler.get_eigenspectrum(num_levels=4)
        np.testing.assert_allclose(vals_explicit, vals_default)


# ---------------------------------------------------------------------------
# Effective coupling (Schrieffer–Wolff)
# ---------------------------------------------------------------------------

class TestEffectiveCoupling:
    def test_output_shape(self, tunable_coupler):
        flux_vals = np.linspace(0.0, 0.4, 25)
        g_eff = tunable_coupler.effective_coupling(flux_vals)
        assert g_eff.shape == (25,)

    def test_finite_far_from_resonance(self, tunable_coupler):
        """At zero flux the coupler is far above the qubits; SW should be finite."""
        g_eff = tunable_coupler.effective_coupling(np.array([0.0]))
        assert np.isfinite(g_eff[0])

    def test_matches_sw_formula_at_zero_flux(self, tunable_coupler):
        flux = 0.0
        omega_A = tunable_coupler.transmon_A.transition_frequency(0, 1)
        omega_B = tunable_coupler.transmon_B.transition_frequency(0, 1)
        omega_C = tunable_coupler._make_coupler(flux).transition_frequency(0, 1)
        D_AC = omega_A - omega_C
        D_BC = omega_B - omega_C
        expected = (tunable_coupler.g_AC * tunable_coupler.g_BC / 2) * (1 / D_AC + 1 / D_BC)
        result = tunable_coupler.effective_coupling(np.array([flux]))
        np.testing.assert_allclose(result[0], expected)

    def test_nan_near_coupler_resonance(self, tunable_coupler):
        """The guard condition sets g_eff = NaN when |Δ_AC| or |Δ_BC| ≤ 1e-4.
        We binary-search for the flux where ω_C = ω_A, then do a very fine
        scan (step ~ 1e-7) around that point to guarantee at least one NaN."""
        omega_A = tunable_coupler.transmon_A.transition_frequency(0, 1)
        lo, hi = 0.0, 0.499
        for _ in range(60):
            mid = (lo + hi) / 2
            omega_C = tunable_coupler._make_coupler(mid).transition_frequency(0, 1)
            if omega_C > omega_A:
                lo = mid
            else:
                hi = mid
        crossing = (lo + hi) / 2
        flux_fine = np.linspace(crossing - 1e-4, crossing + 1e-4, 5000)
        g_eff = tunable_coupler.effective_coupling(flux_fine)
        assert np.any(np.isnan(g_eff))

    def test_g_eff_changes_sign(self, tunable_coupler):
        """g_eff must change sign as the coupler sweeps through qubit frequencies."""
        flux_vals = np.linspace(0.0, 0.499, 300)
        g_eff = tunable_coupler.effective_coupling(flux_vals)
        finite = g_eff[np.isfinite(g_eff)]
        assert np.any(finite > 0) and np.any(finite < 0)


# ---------------------------------------------------------------------------
# Flux sweep spectrum
# ---------------------------------------------------------------------------

class TestFluxSweepSpectrum:
    def test_shape(self, tunable_coupler):
        flux_vals = np.linspace(0.0, 0.4, 15)
        energies = tunable_coupler.flux_sweep_spectrum(flux_vals, num_levels=4)
        assert energies.shape == (15, 4)

    def test_ground_state_column_is_zero(self, tunable_coupler):
        flux_vals = np.linspace(0.0, 0.4, 10)
        energies = tunable_coupler.flux_sweep_spectrum(flux_vals, num_levels=3)
        np.testing.assert_allclose(energies[:, 0], 0.0)

    def test_energies_finite(self, tunable_coupler):
        flux_vals = np.linspace(0.0, 0.4, 10)
        energies = tunable_coupler.flux_sweep_spectrum(flux_vals, num_levels=4)
        assert np.all(np.isfinite(energies))

    def test_spectrum_changes_with_flux(self, tunable_coupler):
        flux_vals = np.array([0.0, 0.3])
        energies = tunable_coupler.flux_sweep_spectrum(flux_vals, num_levels=4)
        assert not np.allclose(energies[0], energies[1])
