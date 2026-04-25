# superqsim

A Python library for simulating superconducting quantum circuits. Implements three device classes — `Transmon`, `TransmonResonator`, and `TunableCouplerSystem` — built on [QuTiP](https://qutip.org/) and NumPy. All energies are in **GHz** with $\hbar = 1$.

---

## Physics Background

### The Transmon Qubit

A transmon is a superconducting qubit based on a Josephson junction shunted by a large capacitance. Its Hamiltonian in the Cooper-pair charge basis is:

$$H = 4E_C\!\left(\hat{n} - n_g\right)^2 - E_J \cos\hat{\varphi}$$

- $\hat{n}$ — Cooper-pair number operator (integer eigenvalues)
- $\hat{\varphi}$ — superconducting phase across the junction (conjugate to $\hat{n}$)
- $E_C = e^2/(2C)$ — charging energy: the electrostatic energy cost of adding one Cooper pair
- $E_J$ — Josephson energy: the energy scale of Cooper-pair tunneling
- $n_g$ — dimensionless offset charge (gate-tunable, or set by stray electric fields)

Expanding the cosine potential in the charge basis $\{|n\rangle,\; n = -N \ldots N\}$ yields a tridiagonal matrix:

$$H_{mn} = 4E_C(m - n_g)^2\,\delta_{mn} - \frac{E_J}{2}\!\left(\delta_{m,n+1} + \delta_{m,n-1}\right)$$

The diagonal gives the electrostatic energy of each charge state; the off-diagonal tunneling terms hybridize neighboring charge states into energy bands.

#### The $E_J/E_C$ Ratio and the Transmon Regime

The character of the qubit is controlled by the ratio $E_J/E_C$:

- **$E_J/E_C \sim 1$ (Cooper-pair box)**: charge states are weakly hybridized. Energy levels are sensitive to $n_g$ — a small stray charge shifts qubit frequency significantly. This "charge noise" is the dominant dephasing mechanism.
- **$E_J/E_C \gg 1$ (transmon)**: deep in the cosine well, wavefunctions spread over many charge states. The energy bands flatten exponentially in $\sqrt{E_J/E_C}$, suppressing charge dispersion. Typical devices use $E_J/E_C \approx 50$.

The cost of this suppression is **reduced anharmonicity**: the spacing between levels becomes more uniform. In the transmon limit the anharmonicity saturates at:

$$\alpha = \omega_{12} - \omega_{01} \approx -E_C$$

For $E_C = 0.2\ \text{GHz}$ this gives $\alpha \approx -200\ \text{MHz}$ — large enough that individual transitions can be addressed with ~10 ns microwave pulses, but small enough that the qubit frequency ($\omega_{01} \approx$ few GHz) is well into the easily-controlled microwave range.

#### Charge Dispersion

Charge dispersion measures how much the qubit frequency varies over a full period of $n_g$:

$$\varepsilon_m = \max_{n_g} E_m - \min_{n_g} E_m$$

It decreases exponentially with $\sqrt{E_J/E_C}$, which is why increasing $E_J/E_C$ is so effective at reducing charge noise. The notebook's `charge_dispersion_sweep` demo illustrates this directly by sweeping $n_g$ from $-1$ to $1$ and plotting the resulting energy bands.

---

### The Transmon–Resonator System

Coupling a transmon to a superconducting microwave resonator (a transmission-line or lumped-element LC circuit) enables qubit control and readout via the **circuit QED** architecture. The full Hamiltonian is:

$$H = H_t \otimes \mathbb{I}_r + \mathbb{I}_t \otimes \omega_r a^\dagger a + g\,\hat{n}_t \otimes \!\left(a + a^\dagger\right)$$

- $a,\,a^\dagger$ — resonator ladder operators
- $\omega_r$ — bare resonator frequency
- $g$ — capacitive coupling strength (typically 10–200 MHz)

The coupling term $g\,\hat{n}_t(a + a^\dagger)$ reflects the fact that the transmon charge operator (electric dipole) couples to the resonator voltage (electric field).

#### Resonant Regime ($|\Delta| \sim g$, $\Delta = \omega_{01} - \omega_r$)

When the transmon and resonator are near resonance, they hybridize into dressed polariton modes. The two lowest excited states repel each other, opening a gap:

$$\text{vacuum-Rabi splitting} \approx 2g\,\left|\langle 0|\hat{n}|1\rangle\right|$$

This avoided crossing is directly observable in the resonator frequency sweep demo.

#### Dispersive Regime ($|\Delta| \gg g$)

Far off resonance, the transmon and resonator remain largely separate but develop a qubit-state-dependent resonator frequency shift. Perturbation theory gives the **dispersive shift**:

$$\chi \approx \frac{g^2\,\alpha}{\Delta(\Delta + \alpha)}$$

where $\alpha$ is the transmon anharmonicity. The resonator frequency effectively becomes:

$$\omega_r \to \omega_r + \chi\,\frac{\sigma_z}{2}$$

so measuring the resonator transmission reveals the qubit state without directly driving the qubit — the foundation of **dispersive readout**. The sign of $\chi$ depends on the sign of $\Delta$ (qubit above or below the resonator).

---

### The Tunable Coupler System

Two transmon qubits (A, B) interact via a SQUID coupler (C) — a loop of two Josephson junctions threaded by an external magnetic flux $\Phi$. The flux tunes the effective Josephson energy of the loop:

$$E_{J,C}(\Phi) = E_{J,\max}\left|\cos\!\left(\pi\,\frac{\Phi}{\Phi_0}\right)\right|$$

where $\Phi_0 = h/(2e)$ is the superconducting flux quantum. As $\Phi$ approaches $\Phi_0/2$, $E_{J,C} \to 0$ and the coupler frequency drops from its maximum (at $\Phi = 0$) down toward the qubit frequencies. This tunability is the key tool for controlling qubit–qubit coupling.

The full three-body Hamiltonian is assembled in the dressed single-qubit eigenbasis to keep the composite Hilbert space tractable:

$$H = H_A \otimes \mathbb{I}_B \otimes \mathbb{I}_C + \mathbb{I}_A \otimes H_B \otimes \mathbb{I}_C + \mathbb{I}_A \otimes \mathbb{I}_B \otimes H_C(\Phi) + g_{AC}\,\hat{n}_A \otimes \mathbb{I}_B \otimes \hat{n}_C + g_{BC}\,\mathbb{I}_A \otimes \hat{n}_B \otimes \hat{n}_C$$

There is no direct A–B coupling term; interaction is mediated entirely through the coupler.

#### Schrieffer–Wolff Effective Coupling

When the coupler is far detuned from both qubits ($|\Delta_{iC}| \gg g_{iC}$, where $\Delta_{iC} = \omega_i - \omega_C$), it can be adiabatically eliminated. The Schrieffer–Wolff (SW) perturbative transformation yields an effective direct A–B coupling:

$$g_\text{eff}(\Phi) = \frac{g_{AC} \cdot g_{BC}}{2}\left(\frac{1}{\Delta_{AC}} + \frac{1}{\Delta_{BC}}\right)$$

This formula has a striking consequence: **$g_\text{eff}$ changes sign** as $\omega_C$ sweeps through the qubit frequencies. In a real device there is typically a residual direct capacitive coupling $g_{AB}$ between the qubits; by tuning $\Phi$ so that the coupler-mediated term exactly cancels $g_{AB}$, the total effective coupling can be set to zero — a "parking" point for idle qubits. This is the operating principle of tunable-coupler architectures used in state-of-the-art superconducting processors.

Near the degeneracy points ($\Delta_{iC} \to 0$) the perturbative approximation breaks down and the full numerical diagonalization is required; the `flux_sweep_spectrum` method provides this.

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
| `get_eigenspectrum(num_levels)` | Eigenvalues ($E_0 = 0$) and eigenstates |
| `transition_frequency(i, j)` | $E_j - E_i$; default is $\omega_{01}$ |
| `anharmonicity()` | $\alpha = \omega_{12} - \omega_{01}$ |
| `charge_dispersion_sweep(ng_values, num_levels)` | Energy bands vs $n_g$ |
| `ej_ec_sweep(ratio_values)` | $\omega_{01}$ and $\alpha$ vs $E_J/E_C$ |
| `EJ_over_Ec` | $E_J/E_C$ ratio |
| `charge_states` | Cooper-pair numbers $-N \ldots N$ |

### `TransmonResonator(Ec, EJ, omega_r, g, ng=0.0, n_cutoff=15, n_fock=10)`

Transmon capacitively coupled to a single-mode resonator.

| Method / Property | Description |
|---|---|
| `build_hamiltonian()` | Full composite Hamiltonian |
| `get_eigenspectrum(num_levels)` | Composite eigensystem; $E_0 = 0$ |
| `dispersive_shift` | $\chi \approx g^2\alpha / (\Delta(\Delta+\alpha))$ |
| `resonator_frequency_sweep(omega_r_values, num_levels)` | Spectrum vs $\omega_r$ (vacuum-Rabi crossing) |
| `transmon` | Access to the bare `Transmon` subsystem |

### `TunableCouplerSystem(Ec_A, EJ_A, ..., Ec_C, EJ_max_C, g_AC, g_BC, ...)`

Two transmons coupled via a flux-tunable SQUID coupler.

| Method / Property | Description |
|---|---|
| `build_hamiltonian(flux)` | Full 3-body Hamiltonian at dimensionless flux $\Phi/\Phi_0$ |
| `get_eigenspectrum(flux, num_levels)` | 3-body eigensystem; $E_0 = 0$ |
| `coupler_EJ(flux)` | $E_{J,\max}\|\cos(\pi\Phi/\Phi_0)\|$ |
| `effective_coupling(flux_values)` | Schrieffer–Wolff $g_\text{eff}$ vs flux |
| `flux_sweep_spectrum(flux_values, num_levels)` | Full numerical spectrum vs flux |

---

## Further Reading

- Koch et al., *Charge-insensitive qubit design derived from the Cooper pair box*, PRA **76**, 042319 (2007) — original transmon proposal
- Blais et al., *Circuit quantum electrodynamics*, Rev. Mod. Phys. **93**, 025005 (2021) — comprehensive review of circuit QED
- Yan et al., *Tunable coupling scheme for implementing high-fidelity two-qubit gates*, PRA **93**, 022309 (2016) — tunable coupler physics
