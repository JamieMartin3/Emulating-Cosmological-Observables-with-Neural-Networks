#!/usr/bin/env python3
"""Batch generation of CLASS and CAMB observables for LHS-sampled cosmologies.

This script combines the sampling logic from ``LHC_Gen_YAML.py`` with the
observable calculation pipeline from ``Cl_CLASS_CAMB_Test.py``. It builds
100 batches of 2 cosmological parameter sets (configurable via CLI), computes
observables with both CLASS and CAMB for every successful draw, and stores all
intermediate and aggregated products on disk.
"""

import argparse
import json
import math
import os
from collections import OrderedDict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
from astropy.cosmology import Planck15 as Planck

import classy
import camb
from camb import CAMBparams, model
import pyDOE


# --------------------------------------------------------------------------------------
# Shared cosmology constants (mirrors Cl_CLASS_CAMB_Test.py)
# --------------------------------------------------------------------------------------
h_planck = Planck.h


class LCDM:
    """Convenience container for Planck15 baseline parameters."""

    h = h_planck
    H0 = Planck.H0.value
    omega_b = Planck.Ob0 * h**2
    omega_cdm = (Planck.Om0 - Planck.Ob0) * h**2
    omega_k = Planck.Ok0
    Neff = Planck.Neff
    Tcmb = Planck.Tcmb0.value
    A_s = 2.097e-9
    tau_reio = 0.0540
    n_s = 0.9652
    m_ncdm = 0.06


# --------------------------------------------------------------------------------------
# Sampling configuration mirrors LHC_Gen_YAML.py
# --------------------------------------------------------------------------------------
COSMO_PARAMS = [
    'logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio',
    'm_ncdm', 'N_eff', 'r', 'log10T_heat_hmcode',
    'w0_fld', 'wa_fld', 'Omega_k'
]

COSMO_PARAMS_CLASS = [
    'ln10^{10}A_s', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'tau_reio',
    'm_ncdm', 'N_eff', 'r', 'log10T_heat_hmcode',
    'w0_fld', 'wa_fld', 'Omega_k'
]

DEFAULT_BATCH_SIZE = 1
DEFAULT_OUTPUT = Path("./batch_outputs")

Z_PK = 2.5
K_VALS = np.logspace(-4, 1, 200)
Z_BG = np.linspace(0, 5, 5_000)
L_MAX = 10_000
ERROR_LOG_FILENAME = "error_log.jsonl"

class SamplerConfig(object):
    """Container for execution parameters."""

    def __init__(self, yaml_file, n_batches, batch_size, output_dir, seed, chunk_size):
        self.yaml_file = yaml_file
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.seed = seed
        self.chunk_size = chunk_size


class ParamSpec(object):
    """Bounds and CLASS mapping for a sampled parameter."""

    def __init__(self, bounds, class_name):
        self.bounds = bounds
        self.class_name = class_name


# --------------------------------------------------------------------------------------
# YAML parsing and LHS sampling utilities
# --------------------------------------------------------------------------------------


def load_yaml_config(yaml_path: Path) -> Dict:
    with yaml_path.open("r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def extract_param_specs(config: Dict) -> OrderedDict[str, ParamSpec]:
    """Extract uniform parameter bounds from YAML configuration."""
    params_section = config.get("params", {})
    specs: OrderedDict[str, ParamSpec] = OrderedDict()

    for name, entry in params_section.items():
        prior = entry.get("prior")
        if not prior:
            continue

        # Only accept parameters that define explicit uniform bounds
        if "min" in prior and "max" in prior:
            bounds = {
                "min": float(prior["min"]),
                "max": float(prior["max"]),
            }

            # Optional: only include parameters of interest
            if name not in COSMO_PARAMS:
                continue

            idx = COSMO_PARAMS.index(name)
            specs[name] = ParamSpec(bounds=bounds, class_name=COSMO_PARAMS_CLASS[idx])

    return specs


def build_candidate_arrays(specs: OrderedDict[str, ParamSpec], n_samples: int) -> np.ndarray:
    arrays = []
    for spec in specs.values():
        bounds = spec.bounds
        arrays.append(np.linspace(bounds['min'], bounds['max'], n_samples))
    return np.vstack(arrays)


def generate_lhs_samples(
    specs: OrderedDict[str, ParamSpec],
    n_samples: int,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    if n_samples <= 0 or not specs:
        return []

    n_params = len(specs)
    param_arrays = build_candidate_arrays(specs, n_samples)

    if seed is not None:
        np.random.seed(seed)

    design = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
    indices = (design * n_samples).astype(int)

    samples: List[Dict[str, float]] = []
    keys = list(specs.keys())
    for sample_idx in range(n_samples):
        values = {}
        for p_idx in range(n_params):
            values[keys[p_idx]] = param_arrays[p_idx][indices[sample_idx, p_idx]]

        if 'N_eff' in values and 'm_ncdm' in values:
            # Constants
            N_ncdm_const = 1
            deg_ncdm = 3

            # Sampled arrays
            neff = values['N_eff']

            # (Optional) store convenience keys
            values['N_ncdm'] = int(N_ncdm_const)
            values['deg_ncdm'] = int(deg_ncdm)

        if values['N_eff'] > 3.03959:
            values['N_ur'] = values['N_eff'] - 3.03959
            values['T_ncdm'] = 0.71611
        else:
            values['N_ur'] = 0.0
            values['T_ncdm'] = 0.71611 * (values['N_eff'] / 3.03959)**0.25
        

        samples.append(values)

    return samples


class LHSSamplePool:
    """Sample provider that replenishes LHS draws in configurable chunks."""

    def __init__(self, specs: OrderedDict[str, ParamSpec], rng: np.random.Generator, chunk_size: int) -> None:
        self._specs = specs
        self._rng = rng
        self._chunk_size = max(chunk_size, 1)
        self._buffer: Deque[Dict[str, float]] = deque()

    def _refill(self, n_samples: Optional[int] = None) -> None:
        target = n_samples or self._chunk_size
        seed = int(self._rng.integers(0, 2**32 - 1))
        new_samples = generate_lhs_samples(self._specs, target, seed=seed)
        self._buffer.extend(new_samples)

    def next(self) -> Dict[str, float]:
        if not self._buffer:
            self._refill()
        return self._buffer.popleft().copy()


# --------------------------------------------------------------------------------------
# Parameter conversion and bookkeeping
# --------------------------------------------------------------------------------------


def logA_to_As(logA: float) -> float:
    """Convert log(10^10 A_s) to A_s."""
    return math.exp(logA) / 1e10


def build_full_parameter_dict(sample: Dict[str, float]) -> Dict[str, float]:
    """Combine sampled values with Planck baselines for downstream use."""
    full = {
        'logA': sample.get('logA', math.log(1e10 * LCDM.A_s)),
        'n_s': sample.get('n_s', LCDM.n_s),
        'H0': sample.get('H0', LCDM.H0),
        'omega_b': sample.get('omega_b', LCDM.omega_b),
        'omega_cdm': sample.get('omega_cdm', LCDM.omega_cdm),
        'tau_reio': sample.get('tau_reio', LCDM.tau_reio),
        'm_ncdm': sample.get('m_ncdm', LCDM.m_ncdm) / (sample.get('N_ncdm', 1) * sample.get('deg_ncdm', 1)),
        'sigma_m': sample.get('m_ncdm', LCDM.m_ncdm),
        'N_eff': sample.get('N_eff', LCDM.Neff),
        'r': sample.get('r', 0.0),
        'log10T_heat_hmcode': sample.get('log10T_heat_hmcode', np.nan),
        'w0_fld': sample.get('w0_fld', -1.0),
        'wa_fld': sample.get('wa_fld', 0.0),
        'Omega_k': sample.get('Omega_k', LCDM.omega_k),
        'N_ncdm': sample.get('N_ncdm', 3),
        'N_ur': sample.get('N_ur', max(LCDM.Neff - 3, 0.0)),
        'deg_ncdm': sample.get('deg_ncdm', 3),
        'T_ncdm': sample.get('T_ncdm', 0.71611 / ((4.0/11.0)**(1.0/3.0))),
    }
    full['A_s'] = logA_to_As(full['logA'])
    return full


def build_camb_class_input(full_params: Dict[str, float]) -> Dict[str, float]:
    """Translate parameter dictionary into the structure expected by CAMB/CLASS."""
    return {
        'H0': float(full_params['H0']),
        'ombh2': float(full_params['omega_b']),
        'omch2': float(full_params['omega_cdm']),
        'tau': float(full_params['tau_reio']),
        'ns': float(full_params['n_s']),
        'As': float(full_params['A_s']),
        'N_eff': float(full_params['N_eff']),
        'sigma_m': float(full_params['sigma_m']),
        'm_ncdm': float(full_params['m_ncdm']),
        'r': float(full_params['r']),
        'log10T_heat_hmcode': float(full_params['log10T_heat_hmcode']) if not math.isnan(full_params['log10T_heat_hmcode']) else None,
        'w0_fld': float(full_params['w0_fld']),
        'wa_fld': float(full_params['wa_fld']),
        'Omega_k': float(full_params['Omega_k']),
        'zpk': Z_PK,
        'N_ncdm': int(full_params['N_ncdm']),
        'N_ur': full_params['N_ur'],
        'T_ncdm': full_params['T_ncdm'],
        'deg_ncdm': int(full_params['deg_ncdm']),
        'm_ncdm_str': ','.join([f'{full_params["m_ncdm"]:.32f}'] * int(full_params["deg_ncdm"])),
        'T_ncdm_str': ','.join([f'{full_params["T_ncdm"]:.32f}'] * int(full_params["deg_ncdm"])),
    }


# --------------------------------------------------------------------------------------
# CAMB / CLASS configuration (copied + lightly parameterised from Cl_CLASS_CAMB_Test.py)
# --------------------------------------------------------------------------------------


def configure_camb_params(p: Dict[str, float], nonlinear: bool = True, lmax: int = L_MAX, BB: bool = False) -> CAMBparams:
    params = CAMBparams()

    mnu_value = p.get('m_ncdm', LCDM.m_ncdm)
    if mnu_value < 0:
        raise ValueError("m_ncdm must be non-negative for CAMB")

    params.set_cosmology(
        H0=p['H0'],
        ombh2=p['ombh2'],
        omch2=p['omch2'],
        tau=p['tau'],
        mnu=p['sigma_m'],
        num_massive_neutrinos=p.get('N_ncdm', 1) * p.get('deg_ncdm', 1),
        nnu=p.get('N_eff', LCDM.Neff),
        omk=p.get('Omega_k', LCDM.omega_k),
    )

    if BB:
        params.WantTensors = True
        params.InitPower.set_params(
            ns=p['ns'],
            As=p['As'],
            r=p.get('r', 0.05),
            nt=None,
            pivot_scalar=0.05,
            pivot_tensor=0.05,
        )
    else:
        params.InitPower.set_params(ns=p['ns'], As=p['As'])

    params.set_accuracy(lAccuracyBoost=2.2, AccuracyBoost=2.0, lSampleBoost=2.0)
    params.set_for_lmax(lmax=lmax, lens_potential_accuracy=15, lens_margin=2050)

    if BB:
        params.max_l_tensor = lmax
        acc = params.Accuracy
        acc.k_eta_max_tensor = max(getattr(acc, 'k_eta_max_tensor', 0.0), 3.5 * lmax)
        acc.k_eta_max_scalar = max(getattr(acc, 'k_eta_max_scalar', 3.0 * lmax), 3.0 * lmax)

    params.NonLinear = model.NonLinear_both if nonlinear else model.NonLinear_none
    params.set_matter_power(redshifts=[p.get('zpk', Z_PK)], kmax=10, k_per_logint=130)

    params.AccurateMassiveNeutrinoTransfers = True
    params.DoLateRadTruncation = False
    params.recombination_model = "HyRec"
    params.halofit_version = "mead2020"
    params.TCMB = Planck.Tcmb0.value
    '''
    log10T_heat = p.get('log10T_heat_hmcode')   
    A_b = p.get('A_b', 3.13)
    eta_b = p.get('eta_b', 0.603)
    if any(x is not None for x in [log10T_heat, A_b, eta_b]):
        try:
            kwargs = {}
            if log10T_heat is not None and not math.isnan(log10T_heat):
                kwargs['log10T_heat'] = log10T_heat
            if A_b is not None and not math.isnan(A_b):
                kwargs['A_b'] = A_b
            if eta_b is not None and not math.isnan(eta_b):
                kwargs['eta_b'] = eta_b
            params.NonLinearModel.set_params(**kwargs)
        except AttributeError:
            pass
    
    if 'w0_fld' in p or 'wa_fld' in p:
        params.set_dark_energy(w=p.get('w0_fld', -1.0), wa=p.get('wa_fld', 0.0))
    '''
    return params


def configure_class(p: Dict[str, float], lmax: int = L_MAX, nonlinear: bool = True, BB: bool = False) -> Dict[str, float]:
    class_params = {
        # Outputs
        'output': 'lCl,tCl,pCl,mPk',
        'modes': 's',

        # Cosmology
        'h': p['H0'] / 100.0,
        'Omega_b': p['ombh2'] / (p['H0'] / 100.0) ** 2,
        'Omega_cdm': p['omch2'] / (p['H0'] / 100.0) ** 2,
        'Omega_k': p.get('Omega_k', 0.0),
        'tau_reio': p['tau'],
        'A_s': p['As'],
        'n_s': p['ns'],
        'T_cmb': Planck.Tcmb0.value,
        'YHe': 'BBN',
        'z_max_pk': max(p.get('zpk', Z_PK), 0.0) + 0.5,
        'N_ncdm': p.get('N_ncdm', 1) * p.get('deg_ncdm', 1),
        'm_ncdm': p['m_ncdm_str'],
        'N_ur': p.get('N_ur', 0.0),
        'T_ncdm': p.get('T_ncdm_str'),

        # Accuracy controls
        'P_k_max_h/Mpc': 100.0,
        'l_max_scalars': lmax,
        'delta_l_max': 1800,
        'l_logstep': 1.025,
        'l_linstep': 20,
        'perturbations_sampling_stepsize': 0.05,
        'l_switch_limber': 30.0,
        'hyper_sampling_flat': 32.0,
        'l_max_g': 40,
        'l_max_ur': 35,
        'l_max_pol_g': 60,
        'l_max_ncdm': 30,
        'ur_fluid_approximation': 2,
        'ur_fluid_trigger_tau_over_tau_k': 130.0,
        'radiation_streaming_approximation': 2,
        'radiation_streaming_trigger_tau_over_tau_k': 240.0,
        'hyper_flat_approximation_nu': 7000.0,
        'transfer_neglect_delta_k_S_t0': 0.17,
        'transfer_neglect_delta_k_S_t1': 0.05,
        'transfer_neglect_delta_k_S_t2': 0.17,
        'transfer_neglect_delta_k_S_e': 0.17,
        'accurate_lensing': 1,
        'start_small_k_at_tau_c_over_tau_h': 0.0004,
        'start_large_k_at_tau_h_over_tau_k': 0.05,
        'tight_coupling_trigger_tau_c_over_tau_h': 0.005,
        'tight_coupling_trigger_tau_c_over_tau_k': 0.008,
        'start_sources_at_tau_c_over_tau_h': 0.006,
        'tol_ncdm_synchronous': 1e-6,
        'recombination': 'HyRec',
        'sBBN file': '/external/bbn/sBBN_2021_copy.dat',
        'non_linear': 'hmcode',
        'lensing': 'yes',
    }
    '''
    if 'w0_fld' in p or 'wa_fld' in p:
        class_params['fluid_equation_of_state'] = 'CLP'
        class_params['Omega_fld'] = 1.0 - (class_params['Omega_b'] + class_params['Omega_cdm'] + class_params.get('Omega_k', 0.0))
        class_params['w0_fld'] = p.get('w0_fld', -1.0)
        class_params['wa_fld'] = p.get('wa_fld', 0.0)
    
    log10T_heat = p.get('log10T_heat_hmcode')
    A_b = p.get('A_b', 3.13)
    eta_b = p.get('eta_b', 0.603)

    if log10T_heat is not None and not math.isnan(log10T_heat):
        class_params['log10T_heat'] = log10T_heat
    if A_b is not None and not math.isnan(A_b):
        class_params['A_b'] = A_b
    if eta_b is not None and not math.isnan(eta_b):
        class_params['eta_b'] = eta_b    
    '''
    if BB:
        class_params.update({
            'modes': 's,t',
            'r': p.get('r', 0.05),
            'k_pivot': 0.05,
            'reio_parametrization': 'reio_camb',
            #'l_max_tensors': lmax,
        })

    return class_params


# --------------------------------------------------------------------------------------
# Observable computation helpers
# --------------------------------------------------------------------------------------


def compute_class_observables(p: Dict[str, float]) -> Dict[str, np.ndarray]:
    class_params_nl = configure_class(p, lmax=L_MAX, nonlinear=True, BB=False)
    cl_nl = classy.Class()

    cl_nl.set(class_params_nl)
    cl_nl.compute()

    cls_l = cl_nl.lensed_cl()
    ell_lensed = cls_l['ell']
    ell_factor = ell_lensed * (ell_lensed + 1) / (2 * np.pi)

    cls_raw = cl_nl.raw_cl(lmax=L_MAX)
    ell_unlensed = cls_raw['ell']
    ell_factor_unlensed = ell_unlensed * (ell_unlensed + 1) / (2 * np.pi)

    pk_lin = np.array([cl_nl.pk_lin(k, p.get('zpk', Z_PK)) for k in K_VALS])
    pk_nonlin = np.array([cl_nl.pk(k, p.get('zpk', Z_PK)) for k in K_VALS])

    pk_nonlin_cb = np.array([cl_nl.pk_cb(k, p.get('zpk', Z_PK)) for k in K_VALS])
    pk_lin_cb = np.array([cl_nl.pk_cb_lin(k, p.get('zpk', Z_PK)) for k in K_VALS])

    H_class = np.array([cl_nl.Hubble(z) for z in Z_BG]) * 299792.458
    DA_class = np.array([cl_nl.angular_distance(z) for z in Z_BG])

    derived = cl_nl.get_current_derived_parameters(
        ['sigma8', 'z_reio', 'z_d', 'rs_d', 'z_rec', 'rs_rec', '100*theta_s', 'YHe']
    )

    cl_nl.struct_cleanup()
    cl_nl.empty()

    class_params_bb = configure_class(p, lmax=L_MAX, nonlinear=True, BB=True)
    cl_bb = classy.Class()
    cl_bb.set(class_params_bb)
    cl_bb.compute()

    cls_bb = cl_bb.raw_cl(lmax=L_MAX)
    ell_unlensed_bb = cls_bb['ell']
    ell_factor_bb = ell_unlensed_bb * (ell_unlensed_bb + 1) / (2 * np.pi)
    cl_bb_unlensed = cls_bb['bb'] * ell_factor_bb

    cl_bb.struct_cleanup()
    cl_bb.empty()

    return {
        'ell_lensed': ell_lensed,
        'cl_tt_lensed': cls_l['tt'] * ell_factor,
        'cl_ee_lensed': cls_l['ee'] * ell_factor,
        'cl_bb_lensed': cls_l['bb'] * ell_factor,
        'cl_te_lensed': cls_l['te'] * ell_factor,
        'ell_phi_phi': ell_lensed,
        'cl_phi_phi': cls_l['pp'],
        'ell_unlensed': ell_unlensed,
        'cl_tt_unlensed': cls_raw['tt'] * ell_factor_unlensed,
        'cl_ee_unlensed': cls_raw['ee'] * ell_factor_unlensed,
        'cl_te_unlensed': cls_raw['te'] * ell_factor_unlensed,
        'ell_bb_unlensed': ell_unlensed_bb[:500],
        'cl_bb_unlensed': cl_bb_unlensed[:500],
        'pk_k': K_VALS,
        'pk_linear': pk_lin,
        'pk_nonlinear': pk_nonlin,
        'pk_linear_cb': pk_lin_cb,
        'pk_nonlinear_cb': pk_nonlin_cb,
        'background_z': Z_BG,
        'background_Hz': H_class,
        'background_DAz': DA_class,
        'derived_theta_star': derived['100*theta_s'],
        'derived_sigma8': derived['sigma8'],
        'derived_YHe': derived['YHe'],
        'derived_z_reio': derived['z_reio'],
        'derived_z_star': derived['z_rec'],
        'derived_r_star': derived['rs_rec'],
        'derived_z_drag': derived['z_d'],
        'derived_r_drag': derived['rs_d'],
    }


def compute_camb_observables(p: Dict[str, float]) -> Dict[str, np.ndarray]:
    camb_params_nl = configure_camb_params(p, nonlinear=True, lmax=L_MAX, BB=False)
    camb_results_nl = camb.get_results(camb_params_nl)
    
    lensed_cls = camb_results_nl.get_lensed_scalar_cls(lmax=L_MAX)
    unlensed_cls = camb_results_nl.get_unlensed_scalar_cls(lmax=L_MAX)
    lp_camb = camb_results_nl.get_lens_potential_cls(lmax=L_MAX, raw_cl=True)
    
    pk_nonlin = camb.get_matter_power_interpolator(
        camb_params_nl, nonlinear=True, hubble_units=False, k_hunit=False, kmax=10, zmax=3
    ).P(p.get('zpk', Z_PK), K_VALS)
    
    pk_nonlin_cb = camb.get_matter_power_interpolator(camb_params_nl, nonlinear=True, hubble_units=False, k_hunit=False, kmax=10, zmax=3, var1='delta_nonu', var2='delta_nonu').P(p.get('zpk', Z_PK), K_VALS)

    H_camb = np.array([camb_results_nl.hubble_parameter(z) for z in Z_BG])
    DA_camb = np.array([camb_results_nl.angular_diameter_distance(z) for z in Z_BG])
    
    param_dict_z0 = dict(p)
    param_dict_z0['zpk'] = 0.0
    camb_params_nl_zero = configure_camb_params(param_dict_z0, nonlinear=True, lmax=L_MAX)
    camb_results_nl_zero = camb.get_results(camb_params_nl_zero)

    derived = camb_results_nl_zero.get_derived_params()
    derived_dict = {
        'theta_star': derived['thetastar'],
        'sigma8': camb_results_nl_zero.get_sigma8_0(),
        'YHe': camb_params_nl_zero.YHe,
        'z_reio': camb_params_nl_zero.Reion.get_zre(camb_params_nl_zero),
        'z_star': derived['zstar'],
        'r_star': derived['rstar'],
        'z_drag': derived['zdrag'],
        'r_drag': derived['rdrag'],
    }

    camb_params_lin = configure_camb_params(p, nonlinear=False, lmax=L_MAX)

    pk_lin = camb.get_matter_power_interpolator(
        camb_params_lin, nonlinear=False, hubble_units=False, k_hunit=False, kmax=10, zmax=3
    ).P(p.get('zpk', Z_PK), K_VALS)
    
    pk_lin_cb = camb.get_matter_power_interpolator(camb_params_lin, nonlinear=False, hubble_units=False, k_hunit=False, kmax=10, zmax=3, var1='delta_nonu', var2='delta_nonu').P(p.get('zpk', Z_PK), K_VALS)

    camb_params_bb = configure_camb_params(p, nonlinear=True, lmax=L_MAX, BB=True)
    camb_results_bb = camb.get_results(camb_params_bb)
    bb_unlensed = camb_results_bb.get_unlensed_total_cls(lmax=L_MAX)
    
    return {
        'ell_lensed': np.arange(lensed_cls.shape[0]),
        'cl_tt_lensed': lensed_cls.T[0],
        'cl_ee_lensed': lensed_cls.T[1],
        'cl_bb_lensed': lensed_cls.T[2],
        'cl_te_lensed': lensed_cls.T[3],
        'ell_phi_phi': np.arange(lp_camb.shape[0]),
        'cl_phi_phi': lp_camb[:, 0],
        'ell_unlensed': np.arange(unlensed_cls.shape[0]),
        'cl_tt_unlensed': unlensed_cls.T[0],
        'cl_ee_unlensed': unlensed_cls.T[1],
        'cl_te_unlensed': unlensed_cls.T[3],
        'ell_bb_unlensed': np.arange(len(bb_unlensed.T[2][:500])),
        'cl_bb_unlensed': bb_unlensed.T[2][:500],
        'pk_k': K_VALS,
        'pk_linear': pk_lin,
        'pk_nonlinear': pk_nonlin,
        'pk_linear_cb': pk_lin_cb,
        'pk_nonlinear_cb': pk_nonlin_cb,
        'background_z': Z_BG,
        'background_Hz': H_camb,
        'background_DAz': DA_camb,
        'derived_theta_star': derived_dict['theta_star'],
        'derived_sigma8': derived_dict['sigma8'],
        'derived_YHe': derived_dict['YHe'],
        'derived_z_reio': derived_dict['z_reio'],
        'derived_z_star': derived_dict['z_star'],
        'derived_r_star': derived_dict['r_star'],
        'derived_z_drag': derived_dict['z_drag'],
        'derived_r_drag': derived_dict['r_drag'],
    }


# --------------------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------------------


def save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
    np.savez_compressed(path, **arrays)


def save_parameters(path: Path, params: Dict[str, float]) -> None:
    arrays = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
    np.savez_compressed(path, **arrays)


def append_error_log(path: Path, entry: Dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------------------
# Main execution loop
# --------------------------------------------------------------------------------------


def run_single_batch(config: SamplerConfig, batch_idx: int) -> None:
    """
    Generate one batch of CLASS and CAMB outputs with pre-validation.
    Quickly skips invalid cosmologies before expensive computations.
    """
    yaml_config = load_yaml_config(config.yaml_file)
    param_specs = extract_param_specs(yaml_config)
    print(f"Extracted parameter specs: {param_specs}")

    rng = np.random.default_rng(config.seed + batch_idx if config.seed is not None else None)
    pool = LHSSamplePool(param_specs, rng=rng, chunk_size=config.chunk_size)
    print(f"Sampling from {len(param_specs)} parameters: {list(param_specs.keys())}")

    batch_dir = config.output_dir / f"batch_{batch_idx:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    error_log_path = config.output_dir / ERROR_LOG_FILENAME
    if error_log_path.exists() and batch_idx == 1:
        error_log_path.unlink()

    set_counter = 0
    
    while set_counter < config.batch_size:
        sample = pool.next()
        full_params = build_full_parameter_dict(sample)
        camb_class_input = build_camb_class_input(full_params)
        print(f"[Batch {batch_idx}, Set {set_counter+1}] Params: {full_params}")
        
        try:
            class_data = compute_class_observables(camb_class_input)
            print(f"[Batch {batch_idx}, Set {set_counter+1}] CLASS complete")
        except Exception as exc:
            append_error_log(error_log_path, {
                'batch': batch_idx,
                'set_index': set_counter + 1,
                'stage': 'CLASS',
                'parameters': full_params,
                'error': repr(exc),
            })
            print(f"[Batch {batch_idx}, Set {set_counter+1}] CLASS failed: {repr(exc)}")
            continue

        try:
            camb_data = compute_camb_observables(camb_class_input)
            print(f"[Batch {batch_idx}, Set {set_counter+1}] CAMB complete")
        except Exception as exc:
            append_error_log(error_log_path, {
                'batch': batch_idx,
                'set_index': set_counter + 1,
                'stage': 'CAMB',
                'parameters': full_params,
                'error': repr(exc),
            })
            print(f"[Batch {batch_idx}, Set {set_counter+1}] CAMB failed: {repr(exc)}")
            continue

        set_idx = set_counter + 1
        save_parameters(batch_dir / f"set_{set_idx:02d}_params", full_params)
        save_npz(batch_dir / f"set_{set_idx:02d}_class", class_data)
        save_npz(batch_dir / f"set_{set_idx:02d}_camb", camb_data)
        print(f"[Batch {batch_idx}, Set {set_idx}] Saved successfully.")
        set_counter += 1
        

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args() -> SamplerConfig:
    parser = argparse.ArgumentParser(description="Batch CLASS/CAMB data generator")
    parser.add_argument(
        '--yaml-file',
        type=Path,
        default='/home/jam249/rds/rds-dirac-dp002/jam249/Neural-Net-Emulator-for-Cosmological-Observables/Data-Generation/Parallel-Data-Gen/parameter-ranges copy.yaml',
        help='Cobaya YAML file used to define cosmological parameter priors.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='/home/jam249/rds/rds-dirac-dp002/jam249/Neural-Net-Emulator-for-Cosmological-Observables/Data-Generation/Parallel-Data-Gen/Slurm_Batch/sh_batch_outputs',
        help='Directory where batch outputs will be written.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Number of parameter sets per batch (default: 2).'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional random seed for reproducibility.'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=256,
        help='Number of LHS samples to draw whenever the pool is replenished.'
    )
    parser.add_argument(
        '--batch-idx',
        type=int,
        required=False,
        default=1,
        help='Index of this SLURM batch (1-based).'
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SamplerConfig(
        yaml_file=args.yaml_file.resolve(),
        n_batches=1,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    run_single_batch(config, args.batch_idx)


if __name__ == '__main__':
    main()
