#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Ricardo Carreon
Created on: 2025-05-26
Last updated: 2025-05-26

Description:
    This is a model extracted from Romanelli et all. (2021).
    The so called AFGA model to calculate the cross sections. Premise:
    'The neutron cross section of a complex molecule ≈ the sum of the neutron cross sections of its functional groups'

Python version: 3.x
"""
# ====== Required Imports ======
import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import itertools


# ====== Parameters and Constants ======
SIGMOID_PARAMS = { #Parameters described in Romanelli, M., et al. (2021).
    # "A new model for the calculation of neutron scattering cross sections of organic molecules." 
    # Journal of Applied Crystallography, 54(2), 123-135.
    "CH_ali": {"sf": 19.14, "sb": 73.11, "c": 37.46, "d": 1.03},
    "CH_aro": {"sf": 17.51, "sb": 86.72, "c": 28.62, "d": 0.84},
    "CH2":    {"sf": 16.20, "sb": 105.1, "c": 23.40, "d": 0.71},
    "CH3":    {"sf": 13.60, "sb": 174.8, "c": 27.14, "d": 0.55},
    "NH3":    {"sf": 18.82, "sb": 74.92, "c": 32.90, "d": 0.94},
    "NH2":    {"sf": 17.54, "sb": 101.5, "c": 28.75, "d": 0.75},
    "NH":     {"sf": 19.10, "sb": 78.09, "c": 36.39, "d": 0.95},
    "OH":     {"sf": 19.27, "sb": 71.29, "c": 44.57, "d": 1.09},
    "SH":     {"sf": 19.24, "sb": 107.3, "c": 60.64, "d": 0.85}
}

FREE_CS = { # Free scattering cross sections (barns) - from Sears 1992
    "H": 82.03,
    "C": 5.55,
    "N": 11.53,
    "O": 4.23,
}

E_eV = [ # Energy values in eV for calculations and plotting. SAme as in the Supplementary section of Romanelli et al. (2021).
    0.0010000000, 0.0012589254, 0.0015848932, 0.0019952623, 0.0025118864,
    0.0031622777, 0.0039810717, 0.0050118723, 0.0063095734, 0.0079432823,
    0.0100000000, 0.0125892541, 0.0158489319, 0.0199526231, 0.0251188643,
    0.0316227766, 0.0398107171, 0.0501187234, 0.0630957344, 0.0794328235,
    0.1000000000, 0.1258925412, 0.1584893192, 0.1995262315, 0.2511886432,
    0.3162277660, 0.3981071706, 0.5011872336, 0.6309573445, 0.7943282347,
    1.0000000000, 2,3,4,5,6,7,8,9,10.000
]


# ====== Calculation Functions ======
def sigmoid_CS(E_meV, sf, sb, c, d):
    """Sigmoidal function for neutron cross section."""
    return sf + sb / (1 + c * E_meV**d)

def calc_CS (E_meV, group_counts, non_H_atoms):
    """
    Calculate AFGA hydrogen cross section and total cross section including free atom terms.

    Parameters:
    - E_meV: array-like, neutron energy in meV
    - group_counts: dict, counts of H-containing functional groups (e.g., {"CH3": 2})
    - non_H_atoms: dict, counts of non-H atoms (e.g., {"C": 8})

    Returns:
    - sigma_H(E): array, energy-dependent H cross section from AFGA
    - sigma_total(E): array, total cross section including fixed non-H contributions
    """
    FREE_CS = {
        "C": 5.55,
        "N": 11.53,
        "O": 4.23,}

    E = np.atleast_1d(E_meV)
    sigma_H = np.zeros_like(E, dtype=float)

    for group, n_groups in group_counts.items():
        if group not in SIGMOID_PARAMS:
            raise ValueError(f"Unknown functional group '{group}'")
        sf, sb, c, d = SIGMOID_PARAMS[group].values()
        sigma_H += n_groups * sigmoid_CS(E, sf, sb, c, d)

    sigma_total = sigma_H.copy()
    for element, count in non_H_atoms.items():
        if element not in FREE_CS:
            raise ValueError(f"Unknown atom type '{element}'")
        sigma_total += count * FREE_CS[element]

    return sigma_H if sigma_H.size > 1 else sigma_H[0], sigma_total if sigma_total.size > 1 else sigma_total[0]


# ====== Conversion Functions ======
def wvl_to_energy(lambda_A):
    return 81.804 / lambda_A**2  # Returns energy in meV

def energy_to_wvl(E_meV):
    return np.sqrt(81.804 / E_meV)  # Returns wavelength in Å


# ====== Plotting Functions ======
def plot_dual_axes(compound_name, group_counts, non_H_atoms, E_eV, sigma_H, sigma_T, H_mod = 15):
    """
    Plots a two-panel figure for a single compound showing hydrogen cross section 
    as a function of energy and wavelength, using dual x-axes.

    Panel A (left):
        - Bottom x-axis: Neutron energy [eV] (log scale)
        - Top x-axis: Neutron wavelength [Å] (log scale)
        - Y-axis: Hydrogen cross section [barn]
        - Plot: H-only contribution (normalized if desired)

    Panel B (right):
        - Bottom x-axis: Neutron wavelength [Å]
        - Top x-axis: Neutron energy [meV]
        - Y-axis: Total and H-only cross section [barn]
        - Plot: Total + H-only contributions

    Parameters:
    -----------
    compound_name : str
        Name of the compound (used as title in both panels).
    
    group_counts : dict
        Dictionary of hydrogen-containing groups with their counts (e.g., {"CH3": 1, "CH_aro": 5}).
    
    non_H_atoms : dict
        Dictionary of non-hydrogen atoms and their counts (e.g., {"C": 7, "O": 1}).
    
    E_eV : list or np.ndarray
        Neutron energy values in electronvolts [eV].
    
    wvl_A : list or np.ndarray
        Corresponding neutron wavelengths in angstroms [Å], precomputed as sqrt(81.804 / E_meV).
    
    sigma_H : sigma_H resulting from the `calc_CS` function.

    sigma_T : sigma_T resulting from the `calc_CS` function.

    H_mod : float, correction with number of hydrogen atoms.

    Returns:
    --------
    None
        The function directly displays the two-panel plot using matplotlib.
    """
    # Conversion to meV
    E_meV = [e*1000 for e in E_eV]

    # Conversion to Angstroms
    wvl_A = energy_to_wvl(np.array(E_meV))


    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Energy (bottom) + Wavelength (top)
    ax_energy = axs[0]
    ax_wave_top = ax_energy.twiny()

    if isinstance(sigma_T, (list, np.ndarray)):
        ax_energy.plot(E_eV, sigma_T, label="Total")
    ax_energy.plot(E_eV, sigma_H/H_mod, "*-", label="H only")
    ax_energy.set_xscale("log")
    ax_energy.set_xlabel("Energy [eV]")
    ax_energy.set_ylabel("Cross Section [barn]")
    ax_energy.set_title(f"{compound_name}")
    ax_energy.grid(True, which="both")
    ax_energy.legend()

    ax_wave_top.set_xscale("log")
    ax_wave_top.set_xlim(ax_energy.get_xlim())
    ax_wave_top.set_xticks(E_eV[::4])
    ax_wave_top.set_xticklabels([f"{w:.2f}" for w in wvl_A[::4]])
    ax_wave_top.set_xlabel("Wavelength [Å]")

    # Panel B: Wavelength (bottom) + Energy (top)
    ax_wave = axs[1]
    ax_energy_top = ax_wave.twiny()

    if isinstance(sigma_T, (list, np.ndarray)):
        ax_wave.plot(wvl_A, sigma_T, label="Total")
    ax_wave.plot(wvl_A, sigma_H/H_mod, "*-", label="H only")
    ax_wave.set_xlabel("Wavelength [Å]")
    ax_wave.set_title(f"{compound_name}")
    ax_wave.grid(True, which="both")
    ax_wave.legend()

    tick_wavelengths = [w for w in ax_wave.get_xticks() if w > 0.0]
    tick_energies = [wvl_to_energy(w) for w in tick_wavelengths]
    ax_energy_top.set_xlim(ax_wave.get_xlim())
    ax_energy_top.set_xticks(tick_wavelengths)
    ax_energy_top.set_xticklabels([f"{e:.2f}" for e in tick_energies])
    ax_energy_top.set_xlabel("Energy [meV]")

    plt.tight_layout()
    plt.show()

def plot_all_CS(full_dict, E_eV, calc_CS, plot_H=True, plot_T=True, energy_limits=None, wvl_limits=None):
    """
    Plots σ_H (normalized per H atom) and/or σ_T for all compounds in `full_dict`
    on a 1×2 subplot layout with dual x-axes.

    Parameters:
    -----------
    full_dict : dict
        Dictionary of compounds: {compound_name: (group_dict, atom_dict)}
    E_eV : list or array
        Neutron energy values [eV]
    calc_CS : function
        Function returning (sigma_H, sigma_T) given E_eV, group_dict, atom_dict
    plot_H : bool
        Whether to plot σ_H / H_mod
    plot_T : bool
        Whether to plot total cross section σ_T
    energy_limits : list of two tuples, optional
        [(x_min, x_max), (y_min, y_max)] for the Energy vs σ plot (Panel 1)
    wvl_limits : list of two tuples, optional
        [(x_min, x_max), (y_min, y_max)] for the Wavelength vs σ plot (Panel 2)
    """
    
    # Conversion to meV
    E_meV = [e*1000 for e in E_eV]

    # Conversion to Angstroms
    wvl_A = energy_to_wvl(np.array(E_meV))

    plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 18,
    "figure.titlesize": 18})
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 16))
    ax_energy = axs[0]
    ax_wave = axs[1]

    ax_wave_top = ax_energy.twiny()
    ax_energy_top = ax_wave.twiny()

    line_styles = itertools.cycle(["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5))])

    for name, (group_counts, non_H_atoms) in full_dict.items():
        sigma_H, sigma_T = calc_CS(E_eV, group_counts, non_H_atoms)
        H_mod = sum(group_counts.values()) or 1  # Avoid division by zero
        style = next(line_styles)
        
        if plot_H:
            ax_energy.plot(E_eV, sigma_H / H_mod, linestyle=style, label=f"{name} (H)")
            ax_wave.plot(wvl_A, sigma_H / H_mod, linestyle=style, label=f"{name} (H)")
        if plot_T:
            ax_energy.plot(E_eV, sigma_T, linestyle=style, label=f"{name} (Total)")
            ax_wave.plot(wvl_A, sigma_T, linestyle=style, label=f"{name} (Total)")

    # Apply axis limits from energy_limits = [(x1, x2), (y1, y2)]
    if energy_limits:
        ax_energy.set_xlim(energy_limits[0])
        ax_energy.set_ylim(energy_limits[1])
        ax_wave_top.set_xlim(energy_limits[0])
    else:
        ax_energy.set_xscale("log")
        ax_wave_top.set_xscale("log")
        ax_wave_top.set_xlim(ax_energy.get_xlim())
    
    # Apply axis limits from wvl_limits = [(x1, x2), (y1, y2)]
    if wvl_limits:
        ax_wave.set_xlim(wvl_limits[0])
        ax_wave.set_ylim(wvl_limits[1])
        ax_energy_top.set_xlim(wvl_limits[0])
    else:
        #ax_wave.set_xscale("log")
        ax_energy_top.set_xlim(ax_wave.get_xlim())

    # Panel 1: Energy bottom / Wavelength top
    ax_energy.set_xscale("log")
    ax_energy.set_xlabel("Energy [eV]")
    ax_energy.set_ylabel("Cross Section [barn]")
    ax_energy.set_title("AFGA Simulated Cross Sections vs Energy", pad=15) # to increase the sitance with upper axis
    ax_energy.grid(True, which="both")
    ax_energy.legend(fontsize="small")

    ax_wave_top.set_xscale("log")
    ax_wave_top.set_xlim(ax_energy.get_xlim())
    ax_wave_top.set_xticks(E_eV[::4])
    ax_wave_top.set_xticklabels([f"{w:.2f}" for w in wvl_A[::4]])
    ax_wave_top.set_xlabel("Wavelength [Å]")

    # Panel 2: Wavelength bottom / Energy top
    ax_wave.set_xlabel("Wavelength [Å]")
    ax_wave.set_ylabel("Cross Section [barn]")
    ax_wave.set_title("AFGA Simulated Cross Sections vs Wavelength", pad=15) # to increase the sitance with upper axis)
    ax_wave.grid(True, which="both")
    ax_wave.legend(fontsize="small")

    tick_wavelengths = [w for w in ax_wave.get_xticks() if w > 0.0]
    tick_energies = [wvl_to_energy(w) for w in tick_wavelengths]
    ax_energy_top.set_xlim(ax_wave.get_xlim())
    ax_energy_top.set_xticks(tick_wavelengths)
    ax_energy_top.set_xticklabels([f"{e:.2f}" for e in tick_energies])
    ax_energy_top.set_xlabel("Energy [meV]")

    plt.tight_layout()
    plt.show()