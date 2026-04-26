# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses a local virtual environment at `.venv/` (Python 3.12).

```bash
source .venv/bin/activate
pip install -e .                  # install package in editable mode (first-time or after pyproject.toml changes)
pip install -e ".[notebook]"      # also installs matplotlib and jupyterlab
```

`requirements.txt` pins all transitive dependencies for exact reproducibility — add new top-level packages to `pyproject.toml` `dependencies` and re-pin there.

## Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_transmon.py -v

# Run a single test
python -m pytest tests/test_transmon.py::TestOperators::test_hamiltonian_diagonal -v

# Launch the demo notebook
jupyter lab notebooks/transmon_demo.ipynb
```

## Project structure

```
src/superqsim/      # installable package
    __init__.py     # re-exports Transmon, TransmonResonator, TunableCouplerSystem
    devices.py      # all simulation classes
tests/              # pytest suite, one file per class
    conftest.py     # shared fixtures
notebooks/          # Jupyter demos (transmon_demo.ipynb)
pyproject.toml      # package metadata and pytest config
requirements.txt    # pinned full dependency tree
```

## Architecture

All simulation code is in **`src/superqsim/devices.py`**. All energies are in **GHz** with **ℏ = 1**.

### Class hierarchy

```
Transmon                         — single qubit in the charge basis
TransmonResonator                — Transmon ⊗ resonator Fock space
TunableCouplerSystem             — Transmon_A ⊗ Transmon_B ⊗ Coupler (dressed basis)
```

All three use `@define` from **attrs** (v26+). `__init__` parameters are declared as annotated class fields; derived objects are built in `__attrs_post_init__`.

### Transmon

Represents $H = 4E_C(\hat n - n_g)^2 - E_J\cos\hat\varphi$ as a $(2N+1)\times(2N+1)$ tridiagonal `qutip.Qobj` where $N$ = `n_cutoff`. Sweep methods (`charge_dispersion_sweep`, `ej_ec_sweep`) temporarily mutate `self.ng` or spin up fresh `Transmon` instances and always restore state in a `try/finally`.

### TransmonResonator

`Ec`, `EJ`, `ng`, `n_cutoff` are stored as fields and forwarded into an embedded `Transmon` created in `__attrs_post_init__` (accessible via the `.transmon` property). The composite Hamiltonian lives in the tensor-product space (transmon charge basis ⊗ resonator Fock space). `resonator_frequency_sweep` mutates `self.omega_r` and restores it with `try/finally`.

### TunableCouplerSystem

Decorated with `@define(kw_only=True)` — all constructor arguments must be keyword-only.

`transmon_A` and `transmon_B` are `field(init=False, repr=False)` instances built in `__attrs_post_init__`. The **coupler** is not stored; it is re-instantiated on every call via `_make_coupler(flux)`, which creates a fresh `Transmon` with `EJ = EJ_max * |cos(π·flux)|`.

The 3-body Hamiltonian is assembled in the **dressed single-transmon eigenbasis** (not the raw charge basis), keeping the composite Hilbert space to `n_levels_A × n_levels_B × n_levels_C` dimensions. `_dressed_charge_op` projects the charge operator into this truncated eigenbasis.

### Hamiltonians

All `build_hamiltonian` methods return `qutip.Qobj` objects. Eigenspectra are always shifted so $E_0 = 0$.

### Tests

`tests/conftest.py` defines shared fixtures (`transmon`, `transmon_cpb`, `transmon_offset`, `transmon_resonator`, `tunable_coupler`). One test file per class, organised into inner classes by method/property group. No `sys.path` manipulation is needed — the editable install makes `superqsim` importable directly.
