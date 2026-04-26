"""
devices.py

Superconducting circuit device classes for quantum simulation.
All energies in GHz with ħ = 1.

Classes
-------
Transmon
    Single transmon qubit in the charge basis.  Charge-dispersion and
    EJ/Ec sweep functionality live as methods on this class.

TransmonResonator
    Transmon capacitively coupled to a microwave resonator.  Provides
    the full composite Hamiltonian, the dispersive-shift property, and
    a resonator-frequency sweep for mapping the vacuum-Rabi avoided crossing.

TunableCouplerSystem
    Two transmons coupled via a flux-tunable SQUID coupler (two Josephson
    junctions in a loop).  The composite Hamiltonian is assembled in the
    dressed single-qubit eigenbasis.  Provides the Schrieffer–Wolff
    effective coupling estimate and a full numerical flux-sweep.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import qutip as qt
from attrs import define, field


# ---------------------------------------------------------------------------
# Transmon
# ---------------------------------------------------------------------------

@define
class Transmon:
    """Single transmon qubit in the charge basis.

    Hamiltonian
    -----------
        H = 4 Ec (n̂ - ng)² - EJ cos φ̂

    In the charge basis {|n⟩, n = -N … N} this becomes the tridiagonal matrix

        H_mn = 4 Ec (m - ng)² δ_mn  -  (EJ/2)(δ_{m,n+1} + δ_{m,n-1})

    Parameters
    ----------
    Ec : float
        Charging energy  Ec = e²/(2C)  in GHz.
    EJ : float
        Josephson energy in GHz.
    ng : float
        Dimensionless offset charge.
    n_cutoff : int
        Charge-basis truncation; Hilbert-space dimension = 2·n_cutoff + 1.
    """

    Ec: float
    EJ: float
    ng: float = 0.0
    n_cutoff: int = 15

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Hilbert-space dimension."""
        return 2 * self.n_cutoff + 1

    @property
    def EJ_over_Ec(self) -> float:
        """EJ / Ec ratio."""
        return self.EJ / self.Ec

    @property
    def charge_states(self) -> np.ndarray:
        """Cooper-pair numbers n = -N … N."""
        return np.arange(-self.n_cutoff, self.n_cutoff + 1, dtype=float)

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def charge_operator(self) -> qt.Qobj:
        """Cooper-pair number operator n̂ in the charge basis."""
        return qt.Qobj(np.diag(self.charge_states), dims=[[self.dim], [self.dim]])

    def build_hamiltonian(self) -> qt.Qobj:
        """Assemble the tridiagonal transmon Hamiltonian as a Qobj."""
        n = self.charge_states
        H = np.diag(4.0 * self.Ec * (n - self.ng) ** 2)
        tunnel = np.full(self.dim - 1, -self.EJ / 2.0)
        H += np.diag(tunnel, 1) + np.diag(tunnel, -1)
        return qt.Qobj(H, dims=[[self.dim], [self.dim]])

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def get_eigenspectrum(
        self,
        num_levels: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[qt.Qobj]]:
        """Diagonalise H; eigenvalues shifted so E₀ = 0.

        Returns
        -------
        eigenvalues : ndarray, shape (k,)
        eigenstates : list of Qobj kets, length k
        """
        H = self.build_hamiltonian()
        vals, vecs = H.eigenstates()
        vals = vals - vals[0]
        k = num_levels if num_levels is not None else len(vals)
        return vals[:k], list(vecs[:k])

    def transition_frequency(self, i: int = 0, j: int = 1) -> float:
        """E_j - E_i.  Defaults to the qubit frequency ω₀₁."""
        vals, _ = self.get_eigenspectrum(max(i, j) + 1)
        return float(vals[j] - vals[i])

    def anharmonicity(self) -> float:
        """α = ω₁₂ - ω₀₁.  Negative for a transmon (≈ -Ec in deep limit)."""
        vals, _ = self.get_eigenspectrum(3)
        return float((vals[2] - vals[1]) - (vals[1] - vals[0]))

    # ------------------------------------------------------------------
    # Sweep methods
    # ------------------------------------------------------------------

    def charge_dispersion_sweep(
        self,
        ng_values: np.ndarray,
        num_levels: int = 5,
    ) -> np.ndarray:
        """Sweep the offset charge ng and return the energy spectrum.

        The original ng is restored after the sweep.

        Parameters
        ----------
        ng_values : ndarray
            Offset-charge values to sweep.
        num_levels : int
            Number of energy levels to track.

        Returns
        -------
        energies : ndarray, shape (len(ng_values), num_levels)
            Ground-state-referenced eigenenergies at each ng.
        """
        energies = np.zeros((len(ng_values), num_levels))
        saved_ng = self.ng
        try:
            for i, ng in enumerate(ng_values):
                self.ng = ng
                vals, _ = self.get_eigenspectrum(num_levels)
                energies[i] = vals
        finally:
            self.ng = saved_ng
        return energies

    def ej_ec_sweep(
        self,
        ratio_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sweep EJ/Ec at fixed Ec and ng.

        Covers the Cooper-pair-box (EJ/Ec ~ 1) → transmon (EJ/Ec >> 1) crossover.

        Parameters
        ----------
        ratio_values : ndarray
            EJ/Ec values to evaluate.

        Returns
        -------
        qubit_freqs : ndarray
            ω₀₁ at each EJ/Ec value.
        anharmonicities : ndarray
            α = ω₁₂ - ω₀₁ at each EJ/Ec value.
        """
        freqs = np.zeros(len(ratio_values))
        anharmon = np.zeros(len(ratio_values))
        for i, ratio in enumerate(ratio_values):
            t = Transmon(self.Ec, ratio * self.Ec, self.ng, self.n_cutoff)
            freqs[i] = t.transition_frequency(0, 1)
            anharmon[i] = t.anharmonicity()
        return freqs, anharmon


# ---------------------------------------------------------------------------
# TransmonResonator
# ---------------------------------------------------------------------------

@define
class TransmonResonator:
    """Transmon qubit capacitively coupled to a microwave resonator.

    Hamiltonian (transmon ⊗ resonator Hilbert space)
    -------------------------------------------------
        H = H_t ⊗ I_r  +  I_t ⊗ ω_r a†a  +  g n̂_t ⊗ (a + a†)

    where H_t is the bare transmon Hamiltonian, n̂_t the charge operator,
    and a is the resonator annihilation operator.

    Parameters
    ----------
    Ec, EJ, ng, n_cutoff : transmon parameters (same as Transmon).
    omega_r : float
        Resonator frequency in GHz.
    g : float
        Capacitive coupling strength in GHz.
    n_fock : int
        Fock-space truncation for the resonator mode.
    """

    Ec: float
    EJ: float
    omega_r: float
    g: float
    ng: float = 0.0
    n_cutoff: int = 15
    n_fock: int = 10
    _transmon: Transmon = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self._transmon = Transmon(Ec=self.Ec, EJ=self.EJ, ng=self.ng, n_cutoff=self.n_cutoff)

    @property
    def transmon(self) -> Transmon:
        """Bare transmon subsystem."""
        return self._transmon

    @property
    def dim(self) -> int:
        """Total Hilbert-space dimension."""
        return self._transmon.dim * self.n_fock

    @property
    def dispersive_shift(self) -> float:
        """Approximate dispersive shift χ ≈ g² α / (Δ (Δ + α)).

        Δ = ω₀₁ - ω_r,  α = anharmonicity (negative for a transmon).
        Valid in the dispersive limit |Δ| >> g.
        """
        omega_01 = self._transmon.transition_frequency(0, 1)
        alpha = self._transmon.anharmonicity()
        Delta = omega_01 - self.omega_r
        return (self.g ** 2 * alpha) / (Delta * (Delta + alpha))

    def build_hamiltonian(self) -> qt.Qobj:
        """Full composite Hamiltonian in the tensor-product Hilbert space."""
        d_t = self._transmon.dim
        H_t = qt.tensor(self._transmon.build_hamiltonian(), qt.qeye(self.n_fock))
        H_r = qt.tensor(qt.qeye(d_t), self.omega_r * qt.num(self.n_fock))
        n_hat = qt.tensor(self._transmon.charge_operator(), qt.qeye(self.n_fock))
        a = qt.tensor(qt.qeye(d_t), qt.destroy(self.n_fock))
        return H_t + H_r + self.g * n_hat * (a + a.dag())

    def get_eigenspectrum(
        self,
        num_levels: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[qt.Qobj]]:
        """Eigenspectrum of the composite system; E₀ = 0."""
        H = self.build_hamiltonian()
        vals, vecs = H.eigenstates()
        vals = vals - vals[0]
        k = num_levels if num_levels is not None else len(vals)
        return vals[:k], list(vecs[:k])

    def resonator_frequency_sweep(
        self,
        omega_r_values: np.ndarray,
        num_levels: int = 6,
    ) -> np.ndarray:
        """Sweep the resonator frequency and return the eigenspectrum.

        The original omega_r is restored after the sweep.

        Parameters
        ----------
        omega_r_values : ndarray
            Resonator frequency values to sweep.
        num_levels : int
            Number of composite eigenstates to track.

        Returns
        -------
        energies : ndarray, shape (len(omega_r_values), num_levels)
            Ground-state-referenced eigenenergies at each omega_r.
        """
        energies = np.zeros((len(omega_r_values), num_levels))
        saved = self.omega_r
        try:
            for i, wr in enumerate(omega_r_values):
                self.omega_r = wr
                vals, _ = self.get_eigenspectrum(num_levels)
                energies[i] = vals
        finally:
            self.omega_r = saved
        return energies


# ---------------------------------------------------------------------------
# TunableCouplerSystem
# ---------------------------------------------------------------------------

@define(kw_only=True)
class TunableCouplerSystem:
    """Two transmons coupled via a flux-tunable SQUID coupler.

    The coupler consists of two Josephson junctions forming a SQUID loop.
    An external flux Φ through the loop tunes the effective Josephson energy:

        EJ_coupler(Φ) = EJ_max × |cos(π Φ/Φ₀)|

    The full 3-body Hamiltonian is assembled in the dressed single-transmon
    eigenbasis to keep the composite Hilbert space tractable:

        H = H_A ⊗ I_B ⊗ I_C  +  I_A ⊗ H_B ⊗ I_C  +  I_A ⊗ I_B ⊗ H_C(Φ)
          + g_AC  n_A ⊗ I_B ⊗ n_C
          + g_BC  I_A ⊗ n_B ⊗ n_C

    where n_X are the charge-operator matrices projected into each subsystem's
    eigenbasis and H_X are diagonal in those bases.

    Parameters
    ----------
    Ec_A, EJ_A, ng_A : transmon A parameters.
    Ec_B, EJ_B, ng_B : transmon B parameters.
    Ec_C, EJ_max_C, ng_C : coupler SQUID parameters.
        EJ_max_C is the zero-flux Josephson energy; ng_C is the coupler
        offset charge.
    g_AC, g_BC : float
        Capacitive coupling strengths (GHz) between each qubit and the coupler.
    n_cutoff : int
        Charge-basis truncation for individually diagonalising each subsystem.
    n_levels_A, n_levels_B, n_levels_C : int
        Number of eigenstates retained per subsystem in the composite Hamiltonian.
        Composite Hilbert-space dimension = n_levels_A × n_levels_B × n_levels_C.
    """

    Ec_A: float
    EJ_A: float
    ng_A: float = 0.0
    Ec_B: float
    EJ_B: float
    ng_B: float = 0.0
    Ec_C: float
    EJ_max_C: float
    ng_C: float = 0.0
    g_AC: float = 0.1
    g_BC: float = 0.1
    n_cutoff: int = 15
    n_levels_A: int = 3
    n_levels_B: int = 3
    n_levels_C: int = 3
    transmon_A: Transmon = field(init=False, repr=False)
    transmon_B: Transmon = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self.transmon_A = Transmon(self.Ec_A, self.EJ_A, self.ng_A, self.n_cutoff)
        self.transmon_B = Transmon(self.Ec_B, self.EJ_B, self.ng_B, self.n_cutoff)

    def coupler_EJ(self, flux: float) -> float:
        """Effective Josephson energy at dimensionless flux Φ/Φ₀."""
        return self.EJ_max_C * abs(np.cos(np.pi * flux))

    def _make_coupler(self, flux: float) -> Transmon:
        return Transmon(self.Ec_C, self.coupler_EJ(flux), self.ng_C, self.n_cutoff)

    @staticmethod
    def _dressed_charge_op(
        n_hat: qt.Qobj,
        vecs: List[qt.Qobj],
        n_levels: int,
    ) -> qt.Qobj:
        """Project the charge operator into the dressed eigenstate basis."""
        n_mat = n_hat.full()
        v_mat = np.column_stack([v.full().flatten() for v in vecs])  # (dim, n_levels)
        result = np.real(v_mat.conj().T @ n_mat @ v_mat)
        return qt.Qobj(result, dims=[[n_levels], [n_levels]])

    def build_hamiltonian(self, flux: float = 0.0) -> qt.Qobj:
        """Full 3-body Hamiltonian at dimensionless flux Φ/Φ₀."""
        coupler = self._make_coupler(flux)

        vals_A, vecs_A = self.transmon_A.get_eigenspectrum(self.n_levels_A)
        vals_B, vecs_B = self.transmon_B.get_eigenspectrum(self.n_levels_B)
        vals_C, vecs_C = coupler.get_eigenspectrum(self.n_levels_C)

        H_A = qt.Qobj(np.diag(vals_A), dims=[[self.n_levels_A], [self.n_levels_A]])
        H_B = qt.Qobj(np.diag(vals_B), dims=[[self.n_levels_B], [self.n_levels_B]])
        H_C = qt.Qobj(np.diag(vals_C), dims=[[self.n_levels_C], [self.n_levels_C]])

        n_A = self._dressed_charge_op(self.transmon_A.charge_operator(), vecs_A, self.n_levels_A)
        n_B = self._dressed_charge_op(self.transmon_B.charge_operator(), vecs_B, self.n_levels_B)
        n_C = self._dressed_charge_op(coupler.charge_operator(), vecs_C, self.n_levels_C)

        I_A = qt.qeye(self.n_levels_A)
        I_B = qt.qeye(self.n_levels_B)
        I_C = qt.qeye(self.n_levels_C)

        return (
            qt.tensor(H_A, I_B, I_C)
            + qt.tensor(I_A, H_B, I_C)
            + qt.tensor(I_A, I_B, H_C)
            + self.g_AC * qt.tensor(n_A, I_B, n_C)
            + self.g_BC * qt.tensor(I_A, n_B, n_C)
        )

    def get_eigenspectrum(
        self,
        flux: float = 0.0,
        num_levels: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[qt.Qobj]]:
        """Eigenspectrum of the full 3-body system at given flux; E₀ = 0."""
        H = self.build_hamiltonian(flux)
        vals, vecs = H.eigenstates()
        vals = vals - vals[0]
        k = num_levels if num_levels is not None else len(vals)
        return vals[:k], list(vecs[:k])

    def effective_coupling(self, flux_values: np.ndarray) -> np.ndarray:
        """Schrieffer–Wolff estimate of the effective A–B coupling vs flux.

            g_eff(Φ) = (g_AC · g_BC / 2) × (1/Δ_AC + 1/Δ_BC)

        where Δ_iC = ω_i - ω_C(Φ).

        Near coupler resonance (Δ_iC ≈ 0) the perturbative approximation
        breaks down; those points are returned as NaN.

        Returns
        -------
        g_eff : ndarray, shape (len(flux_values),)
        """
        omega_A = self.transmon_A.transition_frequency(0, 1)
        omega_B = self.transmon_B.transition_frequency(0, 1)
        g_eff = np.full(len(flux_values), np.nan)
        for i, flux in enumerate(flux_values):
            omega_C = self._make_coupler(flux).transition_frequency(0, 1)
            D_AC = omega_A - omega_C
            D_BC = omega_B - omega_C
            if abs(D_AC) > 1e-4 and abs(D_BC) > 1e-4:
                g_eff[i] = (self.g_AC * self.g_BC / 2.0) * (1.0 / D_AC + 1.0 / D_BC)
        return g_eff

    def flux_sweep_spectrum(
        self,
        flux_values: np.ndarray,
        num_levels: int = 6,
    ) -> np.ndarray:
        """Full numerical eigenspectrum of the 3-body system vs flux.

        Returns
        -------
        energies : ndarray, shape (len(flux_values), num_levels)
        """
        energies = np.zeros((len(flux_values), num_levels))
        for i, flux in enumerate(flux_values):
            vals, _ = self.get_eigenspectrum(flux, num_levels)
            energies[i] = vals
        return energies
