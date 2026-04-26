"""
Shared pytest fixtures for superqsim unit tests.
"""

import numpy as np
import pytest

from superqsim import Transmon, TransmonResonator, TunableCouplerSystem


@pytest.fixture
def transmon():
    """Standard transmon in the deep transmon regime (EJ/Ec = 50)."""
    return Transmon(Ec=0.2, EJ=10.0, ng=0.0, n_cutoff=15)


@pytest.fixture
def transmon_cpb():
    """Cooper-pair box (EJ/Ec ~ 1); charge-sensitive regime."""
    return Transmon(Ec=0.2, EJ=0.2, ng=0.0, n_cutoff=15)


@pytest.fixture
def transmon_offset():
    """Transmon with a non-zero offset charge."""
    return Transmon(Ec=0.2, EJ=10.0, ng=0.5, n_cutoff=15)


@pytest.fixture
def transmon_resonator():
    """Transmon below the resonator — qubit at ~3.8 GHz, resonator at 6.0 GHz."""
    return TransmonResonator(Ec=0.2, EJ=10.0, omega_r=6.0, g=0.1)


@pytest.fixture
def tunable_coupler():
    """Two transmons with a flux-tunable SQUID coupler."""
    return TunableCouplerSystem(
        Ec_A=0.20, EJ_A=10.0,
        Ec_B=0.20, EJ_B=9.5,
        Ec_C=0.30, EJ_max_C=18.0,
        g_AC=0.10, g_BC=0.10,
    )
