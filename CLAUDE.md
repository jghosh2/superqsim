# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses a local virtual environment at `.venv/` (Python 3.12).

```bash
source .venv/bin/activate
pip install -r requirements.txt   # first-time setup
```

## Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_transmon.py -v

# Run a single test
python -m pytest tests/test_transmon.py::TestOperators::test_hamiltonian_diagonal -v

# Launch the demo notebook
jupyter lab transmon_demo.ipynb
```

There is no linter configured. `requirements.txt` pins all transitive dependencies for exact reproducibility — add new top-level packages there.

## Architecture

All simulation code lives in a single module: **`devices.py`**. All energies are in **GHz** with **ℏ = 1**.

### Class hierarchy

```
Transmon                         — single qubit, charge basis
TransmonResonator                — Transmon ⊗ resonator Fock space
TunableCouplerSystem             — Transmon_A ⊗ Transmon_B ⊗ Coupler (dressed basis)
```

All three classes use `@define` from **attrs** (v26+). `__init__` parameters are declared as annotated class fields; derived objects are built in `__attrs_post_init__`.

### Transmon

Represents $H = 4E_C(\hat n - n_g)^2 - E_J\cos\hat\varphi$ as a $(2N+1)\times(2N+1)$ tridiagonal `qutip.Qobj`, where $N$ is `n_cutoff`. The charge states array runs from $-N$ to $N$.

**Sweep methods** (`charge_dispersion_sweep`, `ej_ec_sweep`) temporarily mutate `self.ng` / spin up fresh `Transmon` instances inside a `try/finally` to restore state.

### TransmonResonator

`Ec`, `EJ`, `ng`, `n_cutoff` are stored as fields **and** forwarded into an embedded `Transmon` created in `__attrs_post_init__` (accessible via the `.transmon` property). The composite Hamiltonian is built in the tensor-product space (transmon charge basis ⊗ resonator Fock space). `resonator_frequency_sweep` mutates `self.omega_r` and restores it with `try/finally`.

### TunableCouplerSystem

Uses `@define(kw_only=True)` — all constructor arguments must be passed as keywords.

`transmon_A` and `transmon_B` are `field(init=False, repr=False)` instances built in `__attrs_post_init__`. The **coupler** is not stored; it is re-instantiated on every call via `_make_coupler(flux)`, which creates a fresh `Transmon` with `EJ = EJ_max * |cos(π·flux)|`.

The 3-body Hamiltonian is assembled in the **dressed single-transmon eigenbasis** (not the raw charge basis), keeping the composite Hilbert space to `n_levels_A × n_levels_B × n_levels_C` dimensions. `_dressed_charge_op` projects the charge operator into this truncated eigenbasis.

### Hamiltonians

All `build_hamiltonian` methods return `qutip.Qobj` objects. Eigenspectra are always shifted so $E_0 = 0$.

### Tests

`tests/conftest.py` adds the project root to `sys.path` and defines shared fixtures (`transmon`, `transmon_cpb`, `transmon_offset`, `transmon_resonator`, `tunable_coupler`). One test file per class, organised into inner classes by method/property group.
