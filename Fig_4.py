
"""
Decoherence trajectories in the two-qubit budget geometry.

Simulates:
  - Three local channels: dephasing, depolarising, amplitude damping
  - Plus two correlated 2-qubit channels:
      * correlated_phase_flip
      * correlated_amplitude_damping
  - Local channels can be applied on: A only, B only, or both AB
  - For five initial states:
      * Bell state
      * Pure separable state
      * Mixed entangled state
      * Mixed separable state
      * CHSH-violating Werner-like state

At each noise strength p, we compute (X(p), Y(p)) and plot the trajectory
on top of the analytic budget-geometry boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from dataclasses import dataclass
from scipy.interpolate import interp1d
from collections import OrderedDict, defaultdict
from numpy.linalg import eigvalsh, norm
import os

# =============================
#  Font configuration
# =============================
font_path = "./times.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_family = "Times New Roman"
else:
    font_family = "DejaVu Serif"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [font_family],
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
})
    # ---------------------------------------------------------------------
# Basic linear algebra helpers
# ---------------------------------------------------------------------

def dagger(M):
    return np.conjugate(M).T

def tensor(a, b):
    return np.kron(a, b)

# Pauli & identity
sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0],
                    [0, -1]], dtype=complex)
identity_2 = np.eye(2, dtype=complex)

paulis = [sigma_x, sigma_y, sigma_z]


# ---------------------------------------------------------------------
# Random state generators
# ---------------------------------------------------------------------

def random_complex_vector(d):
    v = np.random.normal(size=(d,)) + 1j * np.random.normal(size=(d,))
    v /= np.linalg.norm(v)
    return v

def random_pure_state_2qubit():
    v = random_complex_vector(4)
    return np.outer(v, np.conjugate(v))

def random_mixed_state_2qubit():
    G = np.random.normal(size=(4, 4)) + 1j * np.random.normal(size=(4, 4))
    M = G @ dagger(G)
    return M / np.trace(M)

def random_single_qubit_pure():
    v = random_complex_vector(2)
    return np.outer(v, np.conjugate(v))

def random_product_pure_state():
    rhoA = random_single_qubit_pure()
    rhoB = random_single_qubit_pure()
    return tensor(rhoA, rhoB)

def random_XY_state(X, Y, ent=0, max_tries=10_000):
    """
    Generate a random physical two-qubit state
    with exact (X,Y), without optimising magic.
    """
    R = X**2 + Y**2
    P = 1.0 / (4.0 - 3.0 * R)
    BL = 3 * P * X**2
    BNL = 3 * P * Y**2

    if (X * X <= 2 / 3) and (R <= 1):
        for _ in range(max_tries):

            # --- random split of local budget ---
            a = np.random.rand()
            r3 = np.sqrt(a * BL)
            s3 = np.sqrt((1 - a) * BL)
            if np.random.rand() < 0.5:
                s3 *= -1

            # --- random split of correlation budget ---
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            t1, t2, t3 = np.sqrt(BNL) * u

            # --- positivity check ---
            Dp = np.sqrt((r3 + s3)**2 + (t1 - t2)**2)
            Dm = np.sqrt((r3 - s3)**2 + (t1 + t2)**2)

            if (1 + t3 >= Dp) and (1 - t3 >= Dm):

                # build rho
                sx = np.array([[0, 1], [1, 0]], complex)
                sy = np.array([[0, -1j], [1j, 0]], complex)
                sz = np.array([[1, 0], [0, -1]], complex)
                I = np.eye(2, dtype=complex)

                rho = (
                    np.kron(I, I)
                    + r3 * np.kron(sz, I)
                    + s3 * np.kron(I, sz)
                    + t1 * np.kron(sx, sx)
                    + t2 * np.kron(sy, sy)
                    + t3 * np.kron(sz, sz)
                ) / 4.0
                
                if ent == 0:

                    return (rho + rho.conj().T) / 2
                    
                else:
                
                    pt = rho.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
                    
                    if (np.min(eigvalsh(pt)) >= -1e-10) and (ent < 0) :
                    
                        return (rho + rho.conj().T) / 2
                        
                    if (np.min(eigvalsh(pt)) < -1e-10) and (ent > 0) :
                    
                        return (rho + rho.conj().T) / 2

        raise RuntimeError("Failed to sample physical state at given (X,Y) ting")

    raise RuntimeError("Failed to sample physical state at given (X,Y)")

# ============================================================
# Diagnostics
# ============================================================

def print_fano(rho):
    r = np.zeros(3); s = np.zeros(3); T = np.zeros((3,3))
    for i in range(3):
        r[i] = np.real(np.trace(rho @ np.kron(pauli[i],I2)))
        s[i] = np.real(np.trace(rho @ np.kron(I2,pauli[i])))
        for j in range(3):
            T[i,j] = np.real(np.trace(rho @ np.kron(pauli[i],pauli[j])))
    print("Fano coefficients:")
    print("r =", np.round(r,6))
    print("s =", np.round(s,6))
    print("T =")
    for row in T:
        print(" ", np.round(row,6))

def print_eigenvalues(rho):
    ev = eigvalsh(rho)
    print("Eigenvalues:", np.round(ev,10), " sum =", np.sum(ev))

def ppt_status(rho, tol=1e-10):
    pt = rho.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
    if np.min(eigvalsh(pt)) >= -tol:
        print("PPT: SEPARABLE")
    else:
        print("PPT: ENTANGLED")
# ---------------------------------------------------------------------
# PPT / negativity
# ---------------------------------------------------------------------

def partial_transpose(rho, sys='A'):
    """
    Partial transpose of 4x4 two-qubit rho.
    Basis: |00>,|01>,|10>,|11>.
    """
    rho = np.asarray(rho, dtype=complex).reshape(4, 4)
    rho_pt = np.zeros_like(rho, dtype=complex)

    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    i = 2*a + b
                    j = 2*c + d
                    if sys == 'A':
                        i_pt = 2*c + b
                        j_pt = 2*a + d
                    else:
                        i_pt = 2*a + d
                        j_pt = 2*c + b
                    rho_pt[i_pt, j_pt] = rho[i, j]
    return rho_pt

def negativity(rho):
    rho_pt = partial_transpose(rho, sys='A')
    evals = np.linalg.eigvalsh(rho_pt)
    neg_evals = evals[evals < 0]
    return float(-np.sum(neg_evals))

def random_mixed_entangled_state(max_tries=5000):
    for _ in range(max_tries):
        rho = random_mixed_state_2qubit()
        if negativity(rho) > .1:  
            return rho
    raise RuntimeError("Could not find entangled mixed state")

def random_mixed_separable_state(max_tries=100000):
    for _ in range(max_tries):
        rho = random_mixed_state_2qubit()
        neg = negativity(rho)
        P = np.real_if_close(np.trace(rho @ rho))
        if neg < 1e-8 and P < 0.29:
            return rho
    raise RuntimeError("Could not find mixed separable state")


# ---------------------------------------------------------------------
# Simple named initial states
# ---------------------------------------------------------------------
def bell_state_phi_minus():
    # (|00> - |11>)/sqrt(2)
    v = np.zeros(4, dtype=complex)
    v[0] = 1/np.sqrt(2)
    v[3] = -1/np.sqrt(2)
    return np.outer(v, np.conjugate(v))

def bell_state_psi_plus():
    # (|01> + |10>)/sqrt(2)
    v = np.zeros(4, dtype=complex)
    v[1] = 1/np.sqrt(2)
    v[2] = 1/np.sqrt(2)
    return np.outer(v, np.conjugate(v))

def bell_state_psi_minus():
    # (|01> - |10>)/sqrt(2)
    v = np.zeros(4, dtype=complex)
    v[1] = 1/np.sqrt(2)
    v[2] = -1/np.sqrt(2)
    return np.outer(v, np.conjugate(v))

def bell_state_phi_plus():
    v = np.zeros(4, dtype=complex)
    v[0] = 1/np.sqrt(2)
    v[3] = 1/np.sqrt(2)
    return np.outer(v, np.conjugate(v))

def pure_separable_state():
    # |00>
    v = np.zeros(4, dtype=complex)
    v[0] = 1.0
    return np.outer(v, np.conjugate(v))

def chsh_strong_werner_state(p=0.9):
    """
    Strongly CHSH-violating Werner-like state:
      rho = p |Phi+><Phi+| + (1-p) I/4, with large p.
    """
    rho_bell = bell_state_phi_plus()
    rho_mix = np.eye(4, dtype=complex)/4.0
    return p * rho_bell + (1-p) * rho_mix


# ---------------------------------------------------------------------
# Fano (Bloch) decomposition
# ---------------------------------------------------------------------

def rho_to_fano(rho):
    """
    For 4x4 two-qubit rho, compute:
      r_i = tr[rho (sigma_i ⊗ I)]
      s_i = tr[rho (I ⊗ sigma_i)]
      t_ij = tr[rho (sigma_i ⊗ sigma_j)]
    """
    r = np.zeros(3, dtype=float)
    s = np.zeros(3, dtype=float)
    t = np.zeros((3, 3), dtype=float)

    for i in range(3):
        op_r = tensor(paulis[i], identity_2)
        r[i] = np.real(np.trace(rho @ op_r))

        op_s = tensor(identity_2, paulis[i])
        s[i] = np.real(np.trace(rho @ op_s))

        for j in range(3):
            op_t = tensor(paulis[i], paulis[j])
            t[i, j] = np.real(np.trace(rho @ op_t))

    return r, s, t


@dataclass
class BudgetPoint:
    X: float
    Y: float
    R: float
    p: float   # noise parameter
    label: str # for trajectory label

def compute_budget_from_r_s_t(r, s, t, p_value, label=""):
    r = np.asarray(r, dtype=float).reshape(3)
    s = np.asarray(s, dtype=float).reshape(3)
    t = np.asarray(t, dtype=float).reshape(3, 3)

    BL  = float(np.dot(r, r) + np.dot(s, s))
    BNL = float(np.sum(t**2))
    B   = BL + BNL
    P   = (B + 1.0)/4.0

    eps = 1e-12
    P_safe = max(P, eps)

    X_sq = BL  / (3.0 * P_safe)
    Y_sq = BNL / (3.0 * P_safe)

    X_sq = max(X_sq, 0.0)
    Y_sq = max(Y_sq, 0.0)

    X = float(np.sqrt(X_sq))
    Y = float(np.sqrt(Y_sq))
    R = X_sq + Y_sq

    return BudgetPoint(X=X, Y=Y, R=R, p=p_value, label=label)


# ---------------------------------------------------------------------
# Noise channels (Kraus form)
# ---------------------------------------------------------------------

def dephasing_channel_kraus(p):
    """
    Single-qubit pure dephasing channel with dephasing strength p ∈ [0,1]:
      p = 0   : identity
      p = 1   : full dephasing
    """
    p = float(np.clip(p, 0.0, 1.0))
    q = 0.5 * p  # effective phase-flip probability

    K0 = np.sqrt(1 - q) * identity_2
    K1 = np.sqrt(q)     * sigma_z
    return [K0, K1]

def depolarizing_channel_kraus(p):
    """
    Single-qubit depolarizing channel with parameter p ∈ [0,1]:
      rho -> (1 - p) * rho + p * I/2
    """
    p = float(np.clip(p, 0.0, 1.0))
    p_eff = 0.75 * p

    K0 = np.sqrt(1 - p_eff) * identity_2
    Kx = np.sqrt(p_eff / 3.0) * sigma_x
    Ky = np.sqrt(p_eff / 3.0) * sigma_y
    Kz = np.sqrt(p_eff / 3.0) * sigma_z
    return [K0, Kx, Ky, Kz]

def amplitude_damping_channel_kraus(p):
    """
    Single-qubit amplitude damping with probability p.
    """
    K0 = np.array([[1, 0],
                   [0, np.sqrt(1-p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)],
                   [0, 0]], dtype=complex)
    return [K0, K1]

def apply_single_qubit_kraus_channel(rho, kraus_ops, which='A'):
    """
    Apply single-qubit channel to a 2-qubit state rho.
    which ∈ {'A','B','AB'}
    """
    rho = np.asarray(rho, dtype=complex).reshape(4, 4)

    if which in ('A', 'B'):
        new_rho = np.zeros_like(rho, dtype=complex)
        for K in kraus_ops:
            if which == 'A':
                K_full = tensor(K, identity_2)
            else:
                K_full = tensor(identity_2, K)
            new_rho += K_full @ rho @ dagger(K_full)
        return new_rho

    elif which == 'AB':
        new_rho = np.zeros_like(rho, dtype=complex)
        for K_a in kraus_ops:
            for K_b in kraus_ops:
                K_full = tensor(K_a, K_b)
                new_rho += K_full @ rho @ dagger(K_full)
        return new_rho

    else:
        raise ValueError("which must be 'A', 'B', or 'AB'")


# ---------------------------------------------------------------------
# Genuine two-qubit (correlated) channels
# ---------------------------------------------------------------------

def apply_two_qubit_kraus_channel(rho, kraus_ops):
    """
    Apply genuine 2-qubit channel with 4x4 Kraus ops K:
      rho -> sum_K K rho K^\dagger
    """
    rho = np.asarray(rho, dtype=complex).reshape(4, 4)
    new_rho = np.zeros_like(rho, dtype=complex)
    for K in kraus_ops:
        new_rho += K @ rho @ dagger(K)
    return new_rho

def correlated_phase_flip_channel_kraus(p):
    """
    Two-qubit correlated phase-flip:
      with prob (1-p): do nothing
      with prob p    : apply Z ⊗ Z
    """
    I4 = np.eye(4, dtype=complex)
    ZZ = tensor(sigma_z, sigma_z)
    K0 = np.sqrt(1 - p) * I4
    K1 = np.sqrt(p)     * ZZ
    return [K0, K1]

def correlated_amplitude_damping_channel_kraus(p):
    """
    Simple correlated amplitude damping:
      |11> decays to |00> with probability p.
    """
    K0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, np.sqrt(1-p)]], dtype=complex)

    K1 = np.zeros((4, 4), dtype=complex)
    K1[0, 3] = np.sqrt(p)  # |00><11|

    return [K0, K1]




# ---------------------------------------------------------------------
# Trajectory simulation
# ---------------------------------------------------------------------

def generate_trajectory(rho0, channel_name, which, p_values):
    """
    For a given initial state rho0, channel, and which-qubit target,
    compute (X(p),Y(p)) trajectory.
    """
    traj = []

    for p in p_values:
        if channel_name == 'dephasing':
            Ks = dephasing_channel_kraus(p)
            rho_t = apply_single_qubit_kraus_channel(rho0, Ks, which=which)

        elif channel_name == 'depolarizing':
            Ks = depolarizing_channel_kraus(p)
            rho_t = apply_single_qubit_kraus_channel(rho0, Ks, which=which)

        elif channel_name == 'amplitude_damping':
            Ks = amplitude_damping_channel_kraus(p)
            rho_t = apply_single_qubit_kraus_channel(rho0, Ks, which=which)

        elif channel_name == 'correlated_phase_flip':
            Ks = correlated_phase_flip_channel_kraus(p)
            rho_t = apply_two_qubit_kraus_channel(rho0, Ks)

        elif channel_name == 'correlated_amplitude_damping':
            Ks = correlated_amplitude_damping_channel_kraus(p)
            rho_t = apply_two_qubit_kraus_channel(rho0, Ks)

        else:
            raise ValueError("Unknown channel_name")

        r, s, t = rho_to_fano(rho_t)
        label = f"{channel_name}_{which}"
        bp = compute_budget_from_r_s_t(r, s, t, p_value=p, label=label)
        traj.append(bp)

    return traj



def plot_negativity_vs_p(rho0, channel_name, which='A', 
                         p_values=None, title=None, show=True):
    """
    Plot negativity as a function of noise strength p for any of your channels.

    channel_name ∈ {
        'dephasing', 'depolarizing', 'amplitude_damping',
        'correlated_phase_flip', 'correlated_amplitude_damping'
    }
    which ∈ {'A','B','AB','corr'} depending on channel type.
    """

    if p_values is None:
        p_values = np.linspace(0, 1, 201)

    negs = []

    for p in p_values:
        # --- Apply the appropriate channel ---
        if channel_name == 'dephasing':
            Ks = dephasing_channel_kraus(p)
            rho_t = apply_single_qubit_kraus_channel(rho0, Ks, which)

        elif channel_name == 'depolarizing':
            Ks = depolarizing_channel_kraus(p)
            rho_t = apply_single_qubit_kraus_channel(rho0, Ks, which)

        elif channel_name == 'amplitude_damping':
            Ks = amplitude_damping_channel_kraus(p)
            rho_t = apply_single_qubit_kraus_channel(rho0, Ks, which)

        elif channel_name == 'correlated_phase_flip':
            Ks = correlated_phase_flip_channel_kraus(p)
            rho_t = apply_two_qubit_kraus_channel(rho0, Ks)

        elif channel_name == 'correlated_amplitude_damping':
            Ks = correlated_amplitude_damping_channel_kraus(p)
            rho_t = apply_two_qubit_kraus_channel(rho0, Ks)

        else:
            raise ValueError("Unknown channel name.")

        # --- compute negativity ---
        negs.append(negativity(rho_t))

    # Convert to NumPy array
    negs = np.array(negs, dtype=float)

    # --- Plot ---
    plt.figure(figsize=(6,4))
    plt.plot(p_values, negs, lw=2)
    plt.axhline(0, color='k', linestyle='--')

    plt.xlabel("Noise strength p", fontsize=14)
    plt.ylabel("Negativity", fontsize=14)
    plt.title(title)
    plt.grid(alpha=0.3)


    plt.tight_layout()
    plt.savefig(f"negativity_vs_p_{channel_name}_{which}.pdf", dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return p_values, negs


def plot_geometry_with_trajectories(trajectories_by_state,
                                    show=True,
                                    save_prefix="Budget-XY-trajectories",
                                    persist_samples=5,
                                    min_sep=0.03,
                                    exclude_right_region=True,
                                    right_x_cut=np.sqrt(2/3)+0.001,
                                    right_y_max=np.sqrt(1/3)+0.001,
                                    start_color_map=None):
    """
    Plot trajectories on the budget-geometry with distinct line styles and
    distinct endpoint markers PER CHANNEL & PER TARGET (A, B, AB, corr).

    New: start_color_map controls the color of the start marker and can be:
      - None: use defaults based on state category (bell, pure product, mixed entangled, mixed separable)
      - dict mapping state_label -> color string (e.g. {"MixedSep":"#A6D96A", "BellPhi+":"#FFD700"})
    """

    import re

    fig, ax = plt.subplots(figsize=(7, 7))

    # ----------------------------
    # Color map (exact keys) and fallback tab20
    # ----------------------------
    custom_colors = {
        "dephasing_A":  "#FF9100",
        "dephasing_B":  "#FF9100",
        "dephasing_AB": "#FF9100",

        "depolarizing_A":  "#FF3C00",
        "depolarizing_B":  "#FF3C00",
        "depolarizing_AB": "#FF3C00",

        "amplitude_damping_A":  "#E61089",
        "amplitude_damping_B":  "#E61089",
        "amplitude_damping_AB": "#E61089",

        # correlated channels (only one target: 'corr')
        "correlated_phase_flip_corr":        "#03AD3F",
        "correlated_amplitude_damping_corr": "#0257E0",
    }

    legend_name_map = {
        "dephasing_B": "Dephasing (one-sided)",
        "dephasing_AB": "Dephasing",
        "dephasing_A": "Dephasing (one-sided)",

        "depolarizing_A": "Depolarising (one-sided)",
        "depolarizing_B": "Depolarising (one-sided)",
        "depolarizing_AB": "Depolarising",

        "amplitude_damping_A": "Amplitude damping (one-sided)",
        "amplitude_damping_B": "Amplitude damping (one-sided)",
        "amplitude_damping_AB": "Amplitude damping",

        "correlated_phase_flip_corr": "Correlated phase flip",
        "correlated_amplitude_damping_corr": "Correlated amplitude damping",
    }

    # ----------------------------
    # Explicit per-traj-key style maps (most-specific)
    # ----------------------------
    styles_exact = {

 		"dephasing_A":  {"dashes": [2, 1]},
        "dephasing_B":  {"dashes": [2, 1]},
        "dephasing_AB": {"dashes": [1, 0]},

        "depolarizing_A":  {"dashes": [2, 1]},
        "depolarizing_B":  {"dashes": [2, 1]},
        "depolarizing_AB": {"dashes": [1, 0]},

        "amplitude_damping_A":  {"dashes": [2, 1]},
        "amplitude_damping_B":  {"dashes": [2, 1]},
        "amplitude_damping_AB": {"dashes": [1, 0]},

        "correlated_phase_flip_corr":        {"dashes": [1, 0]},
        "correlated_amplitude_damping_corr": {"dashes": [1, 0]},
    }

    end_marker_exact = {
        "dephasing_A": "D",
        "dephasing_B": "s",
        "dephasing_AB": "P",

        "depolarizing_A": "^",
        "depolarizing_B": "v",
        "depolarizing_AB": "X",

        "amplitude_damping_A": ">",
        "amplitude_damping_B": "<",
        "amplitude_damping_AB": "*",
        "correlated_phase_flip_corr": "o",
        "correlated_amplitude_damping_corr": "h",
    }

    styles_prefix = {
        "dephasing": ( [1,0] ),
        "depolarizing": ( [5,1] ),
        "amplitude_damping": ( [2,1] ),
        "correlated_phase_flip": ( [3,1,1,1] ),
        "correlated_amplitude_damping": ( [2,2] ),
    }
    marker_prefix = {"A": "o", "B": "s", "AB": "X", "corr": "d"}

    legend_marker_size = 9

    # ============= analytic background ============
    X = np.linspace(0, 1, 2000)
    Y_R1 = np.sqrt(np.maximum(0, 1 - X**2))
    Y_R13 = np.sqrt(np.maximum(0, 1/3 - X**2))

    x_max_cc = np.sqrt(2/3)
    mask_cc = X <= x_max_cc
    X_cc = X[mask_cc]
    Y_cc = np.sqrt(np.maximum(0, 2/3 - 0.5 * X_cc**2))
    Y_cc_interp = interp1d(X_cc, Y_cc, bounds_error=False, fill_value=0)(X)

    Y_chsh_raw = np.sqrt(np.maximum(0, (4 - 3*X**2)/5))
    Y_chsh = np.minimum(Y_chsh_raw, Y_R1)

    Y_arr = np.linspace(0, 1, 2000)
    X_max = np.zeros_like(Y_arr)
    mask_lowY = Y_arr**2 < 1/3
    mask_highY = ~mask_lowY
    X_max[mask_lowY] = np.sqrt(2/3)
    X_max[mask_highY] = np.sqrt(np.maximum(0, 1 - Y_arr[mask_highY]**2))
    X_R1 = np.sqrt(np.maximum(0, 1 - Y_arr**2))

    col_R1 = "#5AA9E6"; col_CHSH_below = "#7FC8F8"; col_CC = "#FFEBEB"; ppt_color = "#FF6392"
    ax.fill_between(X, 0, Y_R1, color=col_R1, alpha=0.4, zorder=1)
    ax.fill_between(X, 0, Y_chsh, color=col_CHSH_below, alpha=0.45, zorder=2)
    ax.fill_between(X, 0, Y_cc_interp, color=col_CC, alpha=1, zorder=3)
    ax.fill_between(X, 0, Y_R13, color="#FFEE99", alpha=0.65, zorder=8)

    ax.plot(X, Y_chsh, "-.", color="#274C77", linewidth=2, zorder=6)
    ax.plot(X_cc, Y_cc, color="#975AE6", linewidth=2.5, zorder=7)
    ax.plot([x_max_cc, x_max_cc], [0, Y_cc[-1]], color="#975AE6", linewidth=2.5, zorder=7)
    ax.plot(X, Y_R13, "--", color=ppt_color, linewidth=2.5, zorder=9)

    epsilon = 0.004
    x0 = np.sqrt(1/3)
    ax.plot(np.linspace(x0, 1, 400), epsilon*np.ones(400), "--", color=ppt_color, linewidth=2.5, zorder=9)
    ax.fill_betweenx(Y_arr, X_max, X_R1, where=(X_R1 > np.sqrt(2.0/3)), color="lightgray", zorder=20)
    ax.plot(X_max, Y_arr, color="black", linewidth=2.5, zorder=21)

    mask_purity = X <= x_max_cc
    ax.plot(X[mask_purity], Y_R1[mask_purity], color="#274C77", linewidth=2.5, zorder=22)
    mask_purity_gray = X > x_max_cc
    ax.plot(X[mask_purity_gray], Y_R1[mask_purity_gray], "--", color="#49616E", linewidth=2.5, zorder=22)

    # boundary functions
    def Y_CHSH_fn(Xv):
        Xv = np.asarray(Xv, dtype=float)
        Y_raw = np.sqrt(np.maximum(0, (4 - 3*Xv**2)/5))
        Y_purity = np.sqrt(np.maximum(0, 1 - Xv**2))
        return np.minimum(Y_raw, Y_purity)

    def Y_CC_fn(Xv):
        Xv = np.asarray(Xv, dtype=float)
        out = np.zeros_like(Xv, dtype=float)
        maskv = Xv <= np.sqrt(2/3)
        out[maskv] = np.sqrt(np.maximum(0, 2/3 - 0.5 * Xv[maskv]**2))
        return out

    def Y_PPT_fn(Xv):
        Xv = np.asarray(Xv, dtype=float)
        return np.sqrt(np.maximum(0, 1/3 - Xv**2))

    # detection tuning
    tol_on = 1e-10
    tol_sign = 1e-6
    hug_threshold = 0.85

    def region_labels(Xs, Ys):
        Xs = np.asarray(Xs, dtype=float); Ys = np.asarray(Ys, dtype=float)
        n = len(Xs)
        labels = np.zeros(n, dtype=int)
        Ych = Y_CHSH_fn(Xs); Ycc = Y_CC_fn(Xs); Ypt = Y_PPT_fn(Xs)
        labels[Ys > Ypt + tol_sign] = 1
        labels[Ys > Ycc + tol_sign] = 2
        labels[Ys > Ych + tol_sign] = 3
        for ii in range(1, n):
            if (abs(Ys[ii] - Ych[ii]) < tol_sign) or (abs(Ys[ii] - Ycc[ii]) < tol_sign) or (abs(Ys[ii] - Ypt[ii]) < tol_sign):
                labels[ii] = labels[ii-1]
        return labels

    placed_label_boxes = []
    def label_overlaps_box(x, y, w=0.06, h=0.03):
        for (xx, yy, ww, hh) in placed_label_boxes:
            if (abs(x - xx) < (w + ww)/2) and (abs(y - yy) < (h + hh)/2):
                return True
        return False
    def register_label_box(x, y, w=0.06, h=0.03):
        placed_label_boxes.append((x, y, w, h))

    seen_p_values = defaultdict(set)
    reported_p_list = defaultdict(list)

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_index = 0
    legend_order = OrderedDict()
    traj_plotted_count = 0
    reported_summary = defaultdict(list)
    suppressed_crossings = []

    # ----------------------------
    # Default start-color mapping by state category (can be overridden by providing start_color_map)
    # ----------------------------
    default_category_colors = {
        "bell": "#99CAF0",          # light blue
        "pure_product": "#FF0000",      # red
        "mixed_entangled": "#5AA9E6", # CHSH
        "mixed_separable": "#FFEE99", # yellow
        "unknown": "#CCCCCC",       # fallback gray
    }

    def infer_state_category(state_label):
        s = str(state_label).lower()
        # bell
        if ("bell" in s) or ("phi" in s) or ("psi" in s and "bell" in s):
            return "bell"
        # pure product / separable pure
        if ("pure" in s) or ("product" in s) or (s.startswith("puresep")) or ("puresep" in s) or ("pure_sep" in s) or ("product" in s):
            return "pure_product"
        # mixed entangled
        if ("mixedent" in s) or ("mixed_ent" in s) or ("entangled" in s) or ("mixedentangled" in s):
            return "mixed_entangled"
        # mixed separable
        if ("mixedsep" in s) or ("mixed_sep" in s) or ("separable" in s) or ("mixed separable" in s):
            return "mixed_separable"
        # fallback checks: if label contains 'sep' assume separable
        if "sep" in s and "mix" in s:
            return "mixed_separable"
        # last resort
        return "unknown"

    # normalize provided start_color_map
    if start_color_map is None:
        start_color_map_resolved = {}
    else:
        # shallow copy so we don't mutate caller dict
        start_color_map_resolved = dict(start_color_map)

    # Now iterate all trajectories
    for state_label, traj_dict in trajectories_by_state.items():
        if not isinstance(traj_dict, dict):
            continue

        # determine start color for this state_label
        if state_label in start_color_map_resolved:
            start_color_for_state = start_color_map_resolved[state_label]
        else:
            cat = infer_state_category(state_label)
            start_color_for_state = start_color_map_resolved.get(cat, default_category_colors.get(cat, default_category_colors["unknown"]))

        for traj_label, traj in traj_dict.items():
            if not traj:
                continue

            Xs_all = np.asarray([bp.X for bp in traj], dtype=float)
            Ys_all = np.asarray([bp.Y for bp in traj], dtype=float)
            ps_all = np.asarray([bp.p for bp in traj], dtype=float)
            mask_all = np.isfinite(Xs_all) & np.isfinite(Ys_all) & np.isfinite(ps_all)
            if not np.any(mask_all):
                continue

            valid_indices = np.nonzero(mask_all)[0]
            first_idx_orig = int(valid_indices[0])
            last_idx_orig  = int(valid_indices[-1])

            Xs = Xs_all[mask_all]; Ys = Ys_all[mask_all]; ps = ps_all[mask_all]
            N = len(Xs)
            if N == 0:
                continue

            traj_key = f"{traj_label}"

            if traj_key in custom_colors:
                color = custom_colors[traj_key]
            else:
                color = colors[color_index % len(colors)]
                color_index += 1

            if traj_key in styles_exact:
                dashes = styles_exact[traj_key]["dashes"]
            else:
                prefix = traj_label.split('_')[0]
                if prefix == "correlated" and len(traj_label.split('_')) >= 2:
                    prefix = f"correlated_{traj_label.split('_')[1]}"
                dashes = styles_prefix.get(prefix, [1, 0])

            if traj_key in end_marker_exact:
                end_marker = end_marker_exact[traj_key]
            else:
                suffix = traj_label.split('_')[-1]
                end_marker = marker_prefix.get(suffix, "X")

            # plot line
            line_handle, = ax.plot(Xs, Ys, linewidth=2, alpha=0.6, color=color, zorder=80)
            try:
                line_handle.set_dashes(dashes)
            except Exception:
                line_handle.set_dashes([1, 0])

            # start marker uses the color chosen per state
            x_start = float(Xs_all[first_idx_orig]); y_start = float(Ys_all[first_idx_orig])
            def start_on_any_boundary(x0, y0):
                return (abs(y0 - Y_CHSH_fn(x0)) < tol_on or
                        abs(y0 - Y_CC_fn(x0))   < tol_on or
                        abs(y0 - Y_PPT_fn(x0))  < tol_on)

            #if not start_on_any_boundary(x_start, y_start):
            ax.scatter(x_start, y_start, s=60, marker='o',
                        facecolor=start_color_for_state,
                        edgecolor="black", linewidths=0.9,
                        zorder=200, clip_on=False)

            # end marker
            x_end = float(Xs_all[last_idx_orig]); y_end = float(Ys_all[last_idx_orig])
            ax.scatter(x_end, y_end, s=80, marker=end_marker, facecolors=color,
                       edgecolors='black', linewidths=0.9, zorder=300, clip_on=False)

            # legend proxy (line + endpoint marker)
            legend_label = legend_name_map.get(traj_label, traj_label)
            if legend_label not in legend_order:
                from matplotlib.lines import Line2D
                proxy = Line2D([0], [0],
                               color=color,
                               linewidth=0.001,
                               marker=end_marker,
                               markersize=legend_marker_size,
                               markeredgecolor='black')
                try:
                    proxy.set_dashes(dashes)
                except Exception:
                    proxy.set_dashes([1,0])
                legend_order[legend_label] = proxy

            traj_plotted_count += 1

            # region detection (same as previous code)
            labels = region_labels(Xs, Ys)
            if len(labels) < 2:
                continue

            Bch = Ys - Y_CHSH_fn(Xs); Bcc = Ys - Y_CC_fn(Xs); Bpt = Ys - Y_PPT_fn(Xs)
            ch_frac = np.mean(np.abs(Bch) < tol_sign)
            cc_frac = np.mean(np.abs(Bcc) < tol_sign)
            pt_frac = np.mean(np.abs(Bpt) < tol_sign)

            i = 0
            while i < len(labels) - 1:
                if labels[i+1] != labels[i]:
                    end_check = min(len(labels), i+1 + persist_samples)
                    if np.all(labels[i+1:end_check] == labels[i+1]):
                        old_label = labels[i]
                        new_label = labels[i+1]

                        if {old_label, new_label} == {0,1}:
                            bname, bfn, bcolor, bfrac = "PPT", Y_PPT_fn, "#ADFF2F", pt_frac
                        elif {old_label, new_label} == {1,2}:
                            bname, bfn, bcolor, bfrac = "CC", Y_CC_fn, "#FFA500", cc_frac
                        elif {old_label, new_label} == {2,3}:
                            bname, bfn, bcolor, bfrac = "CHSH", Y_CHSH_fn, "#FFD700", ch_frac
                        else:
                            if new_label > old_label:
                                boundary_index = old_label + 1
                            else:
                                boundary_index = old_label - 1
                            if boundary_index == 1:
                                bname, bfn, bcolor, bfrac = "PPT", Y_PPT_fn, "#ADFF2F", pt_frac
                            elif boundary_index == 2:
                                bname, bfn, bcolor, bfrac = "CC", Y_CC_fn, "#FFA500", cc_frac
                            else:
                                bname, bfn, bcolor, bfrac = "CHSH", Y_CHSH_fn, "#FFD700", ch_frac

                        if bfrac > hug_threshold:
                            i += 1
                            continue

                        v1 = Ys[i] - bfn(Xs[i]); v2 = Ys[i+1] - bfn(Xs[i+1])
                        if not (abs(v1) > tol_sign or abs(v2) > tol_sign):
                            i += 1
                            continue
                        p1, p2 = float(ps[i]), float(ps[i+1])
                        if (v2 - v1) == 0:
                            i += 1
                            continue
                        p_cross = p1 - v1 * (p2 - p1) / (v2 - v1)
                        if p_cross <= float(ps[0]) + 1e-12 or p_cross < 0 or p_cross > 1:
                            i += 1
                            continue

                        Xc = float(np.interp(p_cross, ps, Xs))
                        Yc = float(np.interp(p_cross, ps, Ys))

                        if exclude_right_region and (Xc >= right_x_cut) and (Yc <= right_y_max):
                            p_round_tmp = round(float(p_cross), 2)
                            suppressed_crossings.append((state_label, traj_label, bname, p_round_tmp, Xc, Yc, "right_region"))
                            i = end_check
                            continue

                        p_round = round(float(p_cross), 2)
                        key = (state_label, traj_label, bname)
                        prevs = reported_p_list[(state_label, traj_label)]
                        if any(abs(p_round - pv) < min_sep for pv in prevs):
                            i += 1
                            continue
                        if p_round in seen_p_values[key]:
                            i += 1
                            continue

                        seen_p_values[key].add(p_round)
                        reported_p_list[(state_label, traj_label)].append(p_round)
                        reported_summary[(state_label, traj_label)].append((bname, p_round))

                        i = end_check
                        continue
                i += 1

    # finalize axes and legend
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xlabel("$X$", fontsize=20); ax.set_ylabel("$Y$", fontsize=20)
    ax.tick_params(labelsize=16); ax.set_aspect("equal", "box"); ax.grid(alpha=0.25)

    if len(legend_order) > 0:
        handles = list(legend_order.values())
        labels = list(legend_order.keys())
        ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=10, frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8); spine.set_zorder(50)

    plt.tight_layout()
    plt.savefig("Fig_4.pdf", dpi=300, bbox_inches="tight")
    if show:
        plt.show()


# In[22]:


def main():
    np.random.seed(42)  # For Pure Prod, Mixed Entangled States:1234, For mixed separable states: 42 

    # Noise strengths
    p_values = np.linspace(0.0, 1.0, 50)

    # Choose initial states to simulate

    # Initial states
    rho_bell_phi_plus  = bell_state_phi_plus()
    rho_bell_phi_minus = bell_state_phi_minus()
    rho_bell_psi_plus  = bell_state_psi_plus()
    rho_bell_psi_minus = bell_state_psi_minus()
    rho_pure_sep       = random_product_pure_state()
    rho_mixed_ent      = random_XY_state(0.2,0.83,+1) 
    rho_mixed_sep      = random_XY_state(0.4,0.3,-1)
    rho_chsh_guarantee = chsh_strong_werner_state(p=0.9)

    initial_states = {
        #  "BellPhi+":  rho_bell_phi_plus,
        # "BellPhi-":  rho_bell_phi_minus,
        # "BellPsi+":  rho_bell_psi_plus,
        # "BellPsi-":  rho_bell_psi_minus,
        # "PureSep":   rho_pure_sep, 
        #    "MixedEnt":  rho_mixed_ent,
         "MixedSep":  rho_mixed_sep,
        #  "CHSHWerner": rho_chsh_guarantee,
    }
    # --- per-channel target mapping ---
    # Each value may be a string ('A','B','AB') or a list of such strings.
    # Correlated (two-qubit) channels will be handled separately and should not appear here.
    local_channel_targets = {
          'dephasing': ['AB'],                 # single target
          'depolarizing': ['AB','A'],               # single target
        # Example: amplitude damping applied to BOTH A and AB:
          'amplitude_damping': ['AB','A'],
    }

    # local channels to run (must be a subset of local_channel_targets keys)
    local_channels = list(local_channel_targets.keys())

    # two-qubit correlated channels (handled with which='corr' in generate_trajectory)
    correlated_channels = ['correlated_amplitude_damping'] #'correlated_phase_flip', 'correlated_amplitude_damping'

    # helper to normalize target entry -> list of targets
    def _normalize_targets(t):
        if isinstance(t, (list, tuple)):
            return list(t)
        return [t]

    # validate targets early (fail fast with friendly error)
    allowed = {'A', 'B', 'AB'}
    for ch, targets in local_channel_targets.items():
        for tt in _normalize_targets(targets):
            if tt not in allowed:
                raise ValueError(f"Invalid target '{tt}' for local channel '{ch}'. "
                                 "Allowed: 'A','B','AB'")

    trajectories_by_state = {}

    for s_label, rho0 in initial_states.items():
        trajectories_by_state[s_label] = {}

        # Local channels: allow multiple targets per channel
        for ch in local_channels:
            targets = _normalize_targets(local_channel_targets.get(ch, 'B'))
            for t in targets:
                traj = generate_trajectory(rho0, ch, which=t, p_values=p_values)
                key = f"{ch}_{t}"
                # If you want guaranteed unique keys even for repeated names, you could enumerate:
                # key = f"{ch}_{t}"
                trajectories_by_state[s_label][key] = traj

        # Correlated channels (genuine 2-qubit) — call with which='corr'
        for ch_corr in correlated_channels:
            traj = generate_trajectory(rho0, ch_corr, which='corr', p_values=p_values)
            key  = f"{ch_corr}_corr"
            trajectories_by_state[s_label][key] = traj

    # Plot results
    plot_geometry_with_trajectories(trajectories_by_state)

if __name__ == "__main__":
    main()







