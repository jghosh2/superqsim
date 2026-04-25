# superqsim

A Python library for simulating superconducting quantum circuits. Implements three device classes — `Transmon`, `TransmonResonator`, and `TunableCouplerSystem` — built on [QuTiP](https://qutip.org/) and NumPy. All energies are in **GHz** with ħ = 1.

---

## Physics Background

### The Transmon Qubit

A transmon is a superconducting qubit based on a Josephson junction shunted by a large capacitance. Its Hamiltonian in the Cooper-pair charge basis is:

```
H = 4 Ec (n̂ - ng)² - EJ cos φ̂
```

- **n̂** — Cooper-pair number operator (integer eigenvalues)
- **φ̂** — superconducting phase across the junction (conjugate to n̂)
- **Ec = e²/(2C)** — charging energy: the electrostatic energy cost of adding one Cooper pair
- **EJ** — Josephson energy: the energy scale of Cooper-pair tunneling
- **ng** — dimensionless offset charge (gate-tunable, or set by stray electric fields)

Expanding the cosine potential in the charge basis {|n⟩, n = −N … N} yields a tridiagonal matrix:

```
H_mn = 4 Ec (m − ng)² δ_mn  −  (EJ/2)(δ_{m,n+1} + δ_{m,n-1})
```

The diagonal gives the electrostatic energy of each charge state; the off-diagonal tunneling terms hybridize neighboring charge states into energy bands.

#### The EJ/Ec Ratio and the Transmon Regime

The character of the qubit is controlled by the ratio EJ/Ec:

- **EJ/Ec ~ 1 (Cooper-pair box)**: charge states are weakly hybridized. Energy levels are sensitive to ng — a small stray charge shifts qubit frequency significantly. This "charge noise" is the dominant dephasing mechanism.
- **EJ/Ec >> 1 (transmon)**: deep in the cosine well, wavefunctions spread over many charge states. The energy bands flatten exponentially in √(EJ/Ec), suppressing charge dispersion. Typical devices use EJ/Ec ≈ 50.

The cost of this suppression is **reduced anharmonicity**: the spacing between levels becomes more uniform. In the transmon limit the anharmonicity saturates at:

```
α = ω₁₂ − ω₀₁  ≈  −Ec
```

For Ec = 0.2 GHz this gives α ≈ −200 MHz — large enough that individual transitions can be addressed with ~10 ns microwave pulses, but small enough that the qubit frequency (ω₀₁ ≈ few GHz) is well into the easily-controlled microwave range.

#### Charge Dispersion

Charge dispersion measures how much the qubit frequency varies over a full period of ng:

```
εm = max_{ng} Em − min_{ng} Em
```

It decreases exponentially with √(EJ/Ec), which is why increasing EJ/Ec is so effective at reducing charge noise. The notebook's `charge_dispersion_sweep` demo illustrates this directly by sweeping ng from −1 to 1 and plotting the resulting energy bands.

---

### The Transmon–Resonator System

Coupling a transmon to a superconducting microwave resonator (a transmission-line or lumped-element LC circuit) enables qubit control and readout via the **circuit QED** architecture. The full Hamiltonian is:

```
H = H_t ⊗ Ir  +  It ⊗ ωr a†a  +  g n̂t ⊗ (a + a†)
```

- **a, a†** — resonator ladder operators
- **ωr** — bare resonator frequency
- **g** — capacitive coupling strength (typically 10–200 MHz)

The coupling term `g n̂t (a + a†)` reflects the fact that the transmon charge operator (electric dipole) couples to the resonator voltage (electric field).

#### Resonant Regime (|Δ| ~ g, Δ = ω₀₁ − ωr)

When the transmon and resonator are near resonance, they hybridize into dressed polariton modes. The two lowest excited states repel each other, opening a gap:

```
vacuum-Rabi splitting ≈ 2g |⟨0|n̂|1⟩|
```

This avoided crossing is directly observable in the resonator frequency sweep demo.

#### Dispersive Regime (|Δ| >> g)

Far off resonance, the transmon and resonator remain largely separate but develop a qubit-state-dependent resonator frequency shift. Perturbation theory gives the **dispersive shift**:

```
χ ≈  g² α / (Δ (Δ + α))
```

where α is the transmon anharmonicity. The resonator frequency effectively becomes:

```
ωr → ωr + χ σz/2
```

so measuring the resonator transmission reveals the qubit state without directly driving the qubit — the foundation of **dispersive readout**. The sign of χ depends on the sign of Δ (qubit above or below the resonator).

---

### The Tunable Coupler System

Two transmon qubits (A, B) interact via a SQUID coupler (C) — a loop of two Josephson junctions threaded by an external magnetic flux Φ. The flux tunes the effective Josephson energy of the loop:

```
EJ_C(Φ) = EJ_max |cos(π Φ/Φ₀)|
```

where Φ₀ = h/(2e) is the superconducting flux quantum. As Φ approaches Φ₀/2, EJ_C → 0 and the coupler frequency drops from its maximum (at Φ = 0) down toward the qubit frequencies. This tunability is the key tool for controlling qubit–qubit coupling.

The full three-body Hamiltonian is assembled in the dressed single-qubit eigenbasis to keep the composite Hilbert space tractable:

```
H = HA ⊗ IB ⊗ IC  +  IA ⊗ HB ⊗ IC  +  IA ⊗ IB ⊗ HC(Φ)
  + g_AC  nA ⊗ IB ⊗ nC
  + g_BC  IA ⊗ nB ⊗ nC
```

There is no direct A–B coupling term; interaction is mediated entirely through the coupler.

#### Schrieffer–Wolff Effective Coupling

When the coupler is far detuned from both qubits (|ΔiC| >> g_iC, where ΔiC = ωi − ωC), it can be adiabatically eliminated. The Schrieffer–Wolff (SW) perturbative transformation yields an effective direct A–B coupling:

```
g_eff(Φ) = (g_AC · g_BC / 2) × (1/Δ_AC + 1/Δ_BC)
```

This formula has a striking consequence: **g_eff changes sign** as ωC sweeps through the qubit frequencies. In a real device there is typically a residual direct capacitive coupling g_AB between the qubits; by tuning Φ so that the coupler-mediated term exactly cancels g_AB, the total effective coupling can be set to zero — a "parking" point for idle qubits. This is the operating principle of tunable-coupler architectures used in state-of-the-art superconducting processors.

Near the degeneracy points (ΔiC → 0) the perturbative approximation breaks down and the full numerical diagonalization is required; the `flux_sweep_spectrum` method provides this.

---

## Installation

```bash
pip install -r requirements.txt
```

The main dependencies are `qutip`, `numpy`, `scipy`, and `matplotlib`. The full `requirements.txt` pins all transitive dependencies for reproducibility.

---

## Quick Start

```python
from devices import Transmon, TransmonResonator, TunableCouplerSystem
import numpy as np

# Single transmon (EJ/Ec = 50, deep transmon regime)
t = Transmon(Ec=0.2, EJ=10.0, ng=0.0, n_cutoff=15)
print(t.transition_frequency(0, 1))   # ω₀₁ ≈ 3.79 GHz
print(t.anharmonicity() * 1e3)        # α ≈ -230 MHz

# Charge dispersion sweep
ng_vals = np.linspace(-1, 1, 300)
energies = t.charge_dispersion_sweep(ng_vals, num_levels=5)

# EJ/Ec crossover
ratios = np.linspace(1, 80, 200)
freqs, anharmon = t.ej_ec_sweep(ratios)

# Transmon coupled to a resonator (dispersive regime)
dev = TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1)
print(dev.dispersive_shift * 1e3)     # χ ≈ -0.43 MHz

# Two qubits with a flux-tunable coupler
sys = TunableCouplerSystem(
    Ec_A=0.20, EJ_A=10.0,
    Ec_B=0.20, EJ_B=9.5,
    Ec_C=0.30, EJ_max_C=18.0,
    g_AC=0.10, g_BC=0.10,
)
flux_vals = np.linspace(0, 0.45, 100)
g_eff = sys.effective_coupling(flux_vals)   # SW estimate vs flux
spec   = sys.flux_sweep_spectrum(flux_vals, num_levels=6)  # full numerics
```

See `transmon_demo.ipynb` for annotated plots of all three systems.

---

## API Reference

### `Transmon(Ec, EJ, ng=0.0, n_cutoff=15)`

Single transmon qubit in the charge basis.

| Method / Property | Description |
|---|---|
| `build_hamiltonian()` | Tridiagonal Hamiltonian as a QuTiP `Qobj` |
| `get_eigenspectrum(num_levels)` | Eigenvalues (E₀ = 0) and eigenstates |
| `transition_frequency(i, j)` | Ej − Ei; default is ω₀₁ |
| `anharmonicity()` | α = ω₁₂ − ω₀₁ |
| `charge_dispersion_sweep(ng_values, num_levels)` | Energy bands vs ng |
| `ej_ec_sweep(ratio_values)` | ω₀₁ and α vs EJ/Ec |
| `EJ_over_Ec` | EJ/Ec ratio |
| `charge_states` | Cooper-pair numbers −N … N |

### `TransmonResonator(Ec, EJ, omega_r, g, ng=0.0, n_cutoff=15, n_fock=10)`

Transmon capacitively coupled to a single-mode resonator.

| Method / Property | Description |
|---|---|
| `build_hamiltonian()` | Full composite Hamiltonian |
| `get_eigenspectrum(num_levels)` | Composite eigensystem; E₀ = 0 |
| `dispersive_shift` | χ ≈ g²α / (Δ(Δ+α)) |
| `resonator_frequency_sweep(omega_r_values, num_levels)` | Spectrum vs ωr (vacuum-Rabi crossing) |
| `transmon` | Access to the bare `Transmon` subsystem |

### `TunableCouplerSystem(Ec_A, EJ_A, ..., Ec_C, EJ_max_C, g_AC, g_BC, ...)`

Two transmons coupled via a flux-tunable SQUID coupler.

| Method / Property | Description |
|---|---|
| `build_hamiltonian(flux)` | Full 3-body Hamiltonian at dimensionless flux Φ/Φ₀ |
| `get_eigenspectrum(flux, num_levels)` | 3-body eigensystem; E₀ = 0 |
| `coupler_EJ(flux)` | EJ_max |cos(π Φ/Φ₀)| |
| `effective_coupling(flux_values)` | Schrieffer–Wolff g_eff vs flux |
| `flux_sweep_spectrum(flux_values, num_levels)` | Full numerical spectrum vs flux |

---

## Further Reading

- Koch et al., *Charge-insensitive qubit design derived from the Cooper pair box*, PRA 76, 042319 (2007) — original transmon proposal
- Blais et al., *Circuit quantum electrodynamics*, Rev. Mod. Phys. 93, 025005 (2021) — comprehensive review of circuit QED
- Yan et al., *Tunable coupling scheme for implementing high-fidelity two-qubit gates*, PRA 93, 022309 (2016) — tunable coupler physics
