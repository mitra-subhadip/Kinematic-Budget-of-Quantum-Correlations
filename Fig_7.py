#!/usr/bin/env python3
"""
Decoherence trajectories in the Qubit-Qutrit (2x3) System.
Supports both XY Plane (Normalized) and BL-BNL Plane (Physical Budgets).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from dataclasses import dataclass
from collections import OrderedDict
import os

# =============================
#  0. Global Configuration
# =============================
PLOT_MODE = "BOTH"  # Options: "XY", "BLBNL", "BOTH"

# =============================
#  1. Font & Style Configuration
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
    "axes.linewidth": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# =============================
#  2. Math Helpers (2x3 System)
# =============================

def dagger(M):
    return np.conjugate(M).T

def tensor(a, b):
    return np.kron(a, b)

# Identities
I2 = np.eye(2, dtype=complex)
I3 = np.eye(3, dtype=complex)

def partial_trace_2x3(rho, keep='A'):
    rho_tensor = rho.reshape(2, 3, 2, 3)
    if keep == 'A':
        return np.einsum('ijkj->ik', rho_tensor)
    else:
        return np.einsum('ijil->jl', rho_tensor)

def partial_transpose_2x3(rho):
    rho_tensor = rho.reshape(2, 3, 2, 3)
    rho_pt = rho_tensor.transpose(0, 3, 2, 1)
    return rho_pt.reshape(6, 6)

def get_min_eigenvalue(rho):
    return np.min(np.linalg.eigvalsh(rho))

def get_negativity(rho):
    rho_pt = partial_transpose_2x3(rho)
    evals = np.linalg.eigvalsh(rho_pt)
    neg_sum = np.sum(np.abs(evals)) - 1.0
    return max(0.0, neg_sum / 2.0)

def is_ppt(rho):
    return get_min_eigenvalue(partial_transpose_2x3(rho)) >= -1e-9

# =============================
#  3. Coordinate Transformation
# =============================

def rho_to_coords_2x3(rho):
    rho = np.asarray(rho, dtype=complex)
    P = np.real(np.trace(rho @ rho))
    if P < 1e-9: P = 1e-9
    
    rhoA = partial_trace_2x3(rho, keep='A')
    rhoB = partial_trace_2x3(rho, keep='B')
    pA = np.real(np.trace(rhoA @ rhoA))
    pB = np.real(np.trace(rhoB @ rhoB))
    
    B_L = 2.0 * pA + 3.0 * pB - 2.0
    B_tot = 6.0 * P - 1.0
    B_NL = B_tot - B_L
    
    B_L = max(0.0, B_L)
    B_NL = max(0.0, B_NL)
    
    norm_factor = 5.0 * P
    X = np.sqrt(B_L / norm_factor)
    Y = np.sqrt(B_NL / norm_factor)
    
    return X, Y

def xy_to_blbnl(x, y):
    """
    Transforms normalized (X, Y) to physical (BL, BNL).
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    r2 = x**2 + y**2
    # Clamp r2 to slightly less than 1.2 to avoid singularity
    r2 = np.minimum(r2, 1.199)
    
    denom = 6.0 - 5.0 * r2
    P = 1.0 / denom
    
    BL = 5.0 * P * x**2
    BNL = 5.0 * P * y**2
    
    return BL, BNL

@dataclass
class BudgetPoint:
    X: float
    Y: float
    p: float
    label: str

# =============================
#  4. State Generators (2x3)
# =============================

def random_ket(d):
    v = np.random.normal(0, 1, size=d) + 1j * np.random.normal(0, 1, size=d)
    return v / np.linalg.norm(v)

def ginibre_mixed_state(dim=6):
    Gr = np.random.normal(0, 1, size=(dim, dim))
    Gi = np.random.normal(0, 1, size=(dim, dim))
    G = Gr + 1j * Gi
    rho = G @ dagger(G)
    rho /= np.trace(rho)
    return rho

def state_classical_dirichlet():
    p = np.random.dirichlet(np.ones(6)) 
    return np.diag(p).astype(complex)

def state_pure_entangled_max():
    v = np.zeros(6, dtype=complex)
    v[0] = 1.0 / np.sqrt(2)
    v[4] = 1.0 / np.sqrt(2)
    return np.outer(v, v.conj())

def state_pure_product():
    va = random_ket(2)
    vb = random_ket(3)
    v = tensor(va, vb)
    return np.outer(v, v.conj())

def state_mixed_general(target="PPT", min_Y=0.0):
    max_iter = 10000
    for _ in range(max_iter):
        if target == "NPT" and min_Y > 0.6:
            rho_rand = ginibre_mixed_state(6)
            rho_bell = state_pure_entangled_max()
            w = np.random.uniform(0.6, 0.95) 
            rho = w * rho_bell + (1-w) * rho_rand
            rho /= np.trace(rho)
        else:
            rho = ginibre_mixed_state(6)

        ppt_check = is_ppt(rho)
        _, Y = rho_to_coords_2x3(rho)
        
        if Y < min_Y: continue

        if target == "PPT" and ppt_check:
            if np.trace(rho@rho) < 0.9: return rho
        elif target == "NPT" and not ppt_check:
            if get_negativity(rho) > 1e-3: return rho
            
    return state_pure_entangled_max() if target=="NPT" else ginibre_mixed_state(6)

def state_mem_critical():
    """
    Maximally Mixed Entangled State (MEM) at the kinematic floor (w=2/3).
    Ansatz: w|Phi+><Phi+| + (1-w)sigma
    Filler sigma = I/2 (x) |2><2|
    """
    w = 2.0/3.0
    rho_bell = state_pure_entangled_max()
    
    # Filler state sigma: Qubit mixed (I/2), Qutrit pure |2><2|
    sigma_A = I2 / 2.0
    sigma_B = np.zeros((3,3), dtype=complex); sigma_B[2,2] = 1.0
    sigma = tensor(sigma_A, sigma_B)
    
    return w * rho_bell + (1-w) * sigma

def state_filler_sigma():
    """
    The filler state sigma = I/2 (qubit) (x) |2><2| (qutrit). 
    Populates indices 2 (|02>) and 5 (|12>) equally.
    """
    rho = np.zeros((6,6), dtype=complex)
    rho[2,2] = 0.5
    rho[5,5] = 0.5
    return rho

def state_mem_boundary(w=2/3):
    """
    Generates a state on the MEM boundary. 
    rho = w * |Phi+><Phi+| + (1-w) * sigma
    w=1 is Pure Bell; w=2/3 is Maximally Mixed Entangled.
    """
    rho_bell = state_pure_entangled_max()
    rho_sigma = state_filler_sigma()
    return w * rho_bell + (1-w) * rho_sigma

# =============================
#  5. Kraus Operators
# =============================

def kraus_qubit_depol(p):
    K0 = np.sqrt(1 - 0.75*p) * I2
    f = np.sqrt(p/4.0)
    X = np.array([[0,1],[1,0]], complex); Y = np.array([[0,-1j],[1j,0]], complex); Z = np.array([[1,0],[0,-1]], complex)
    return [K0, f*X, f*Y, f*Z]

def kraus_qubit_ad(p):
    E0 = np.array([[1, 0], [0, np.sqrt(1-p)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return [E0, E1]

def kraus_qubit_deph(p):
    K0 = np.sqrt(1-p) * I2
    K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
    return [K0, K1]

def kraus_qutrit_depol(p):
    coeff_0 = np.sqrt(max(0, 1.0 - (8.0 * p / 9.0)))
    ops = [coeff_0 * I3]
    w = np.exp(2j * np.pi / 3.0)
    Z = np.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]], dtype=complex)
    X = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    coeff_k = np.sqrt(p / 9.0)
    for n in range(3):
        for m in range(3):
            if n==0 and m==0: continue
            Op = np.linalg.matrix_power(X, n) @ np.linalg.matrix_power(Z, m)
            ops.append(coeff_k * Op)
    return ops

def kraus_qutrit_ad(p):
    E0 = np.diag([1, np.sqrt(1-p), np.sqrt(1-p)]).astype(complex)
    E1 = np.zeros((3,3), dtype=complex); E1[0,1] = np.sqrt(p)
    E2 = np.zeros((3,3), dtype=complex); E2[0,2] = np.sqrt(p)
    return [E0, E1, E2]

def kraus_qutrit_deph(p):
    K0 = np.sqrt(1-p) * I3
    P1 = np.diag([1, -1, 1]).astype(complex)
    P2 = np.diag([1, 1, -1]).astype(complex)
    K1 = np.sqrt(p/2.0) * P1
    K2 = np.sqrt(p/2.0) * P2
    return [K0, K1, K2]

def kraus_fcad_2x3(p):
    diag_elements = [1.0, 1.0, 1.0, 1.0, np.sqrt(1-p), np.sqrt(1-p)]
    A0 = np.diag(diag_elements).astype(complex)
    A1 = np.zeros((6, 6), dtype=complex); A1[0, 4] = np.sqrt(p)
    A2 = np.zeros((6, 6), dtype=complex); A2[0, 5] = np.sqrt(p)
    return [A0, A1, A2]

def apply_channel_2x3(rho, channel_name, target, p):
    rho = rho.reshape(6, 6)
    new_rho = np.zeros_like(rho, dtype=complex)

    if channel_name == 'correlated_amplitude_damping':
        ops = kraus_fcad_2x3(p)
        for K in ops: new_rho += K @ rho @ dagger(K)
    elif channel_name == 'mem_generation':
        # Map simulation p [0, 1] to physical mixing p_phys [0, 1/3]
        # p_phys = 0   => Pure Bell (w=1)
        # p_phys = 1/3 => Max Mixed Entangled (w=2/3)
        p_phys = p * (1.0 / 3.0)
        
        sigma = state_filler_sigma()
        new_rho = (1 - p_phys) * rho + p_phys * sigma
        return new_rho / np.trace(new_rho)
    else:
        if channel_name == 'depolarizing':
            ops_A = kraus_qubit_depol(p); ops_B = kraus_qutrit_depol(p)
        elif channel_name == 'amplitude_damping':
            ops_A = kraus_qubit_ad(p); ops_B = kraus_qutrit_ad(p)
        elif channel_name == 'dephasing':
            ops_A = kraus_qubit_deph(p); ops_B = kraus_qutrit_deph(p)
        else: raise ValueError(f"Unknown channel: {channel_name}")

        if target == 'A':
            for ka in ops_A:
                K = tensor(ka, I3); new_rho += K @ rho @ dagger(K)
        elif target == 'B':
            for kb in ops_B:
                K = tensor(I2, kb); new_rho += K @ rho @ dagger(K)
        elif target == 'AB':
            for ka in ops_A:
                for kb in ops_B:
                    K = tensor(ka, kb); new_rho += K @ rho @ dagger(K)
    return new_rho / np.trace(new_rho)

# =============================
#  6. Simulation & Geometry
# =============================

def generate_trajectory(rho0, channel_name, target, p_values):
    traj = []
    for p in p_values:
        rho_t = apply_channel_2x3(rho0, channel_name, target, p)
        X, Y = rho_to_coords_2x3(rho_t)
        traj.append(BudgetPoint(X, Y, p, f"{channel_name}_{target}"))
    return traj

# --- A. Original XY Plotter ---
def plot_geometry_xy(trajectories_by_state, save_prefix="QubitQutrit_XY"):
    fig, ax = plt.subplots(figsize=(8, 8))

    X_vals = np.linspace(0, 1.0, 1000)
    Y_R1 = np.sqrt(np.maximum(0, 1 - X_vals**2))
    
    # --- C. QC Envelope (Corrected Piecewise Projection) ---
    X_prod = np.sqrt(0.6)
    
    # Segment 1: Trine to Dimer (B_L from 0 to 0.5)
    BL_qc1 = np.linspace(0, 0.5, 500)
    BNL_qc1 = 1 + BL_qc1
    P_qc1 = (BL_qc1 + BNL_qc1 + 1) / 6.0
    X_qc1 = np.sqrt(BL_qc1 / (5 * P_qc1))
    Y_qc1 = np.sqrt(BNL_qc1 / (5 * P_qc1))

    # Segment 2: Dimer to Pure Product (B_L from 0.5 to 3.0)
    BL_qc2 = np.linspace(0.5, 3.0, 500)
    BNL_qc2 = 1.5 + 0.2 * (BL_qc2 - 0.5)
    P_qc2 = (BL_qc2 + BNL_qc2 + 1) / 6.0
    X_qc2 = np.sqrt(BL_qc2 / (5 * P_qc2))
    Y_qc2 = np.sqrt(BNL_qc2 / (5 * P_qc2))

    # Combine the true X and Y coordinates of the QC boundary
    X_qc_true = np.concatenate([X_qc1, X_qc2])
    Y_qc_true = np.concatenate([Y_qc1, Y_qc2])

    def get_qc_envelope(x_vals):
        # Interpolate onto our master X array using the properly curved coordinates
        y_vals = np.interp(x_vals, X_qc_true, Y_qc_true, left=np.nan, right=0)
        y_vals[x_vals > X_prod] = 0
        return y_vals

    Y_qc = get_qc_envelope(X_vals)
    
    w = np.linspace(2/3, 1, 200)
    w2 = w**2
    P_qq = (3*w2 - 2*w + 1) / 2.0
    BL_qq = 4.5*w2 - 6*w + 2
    BNL_qq = 4.5*w2
    X_qq = np.sqrt(BL_qq / (5 * P_qq))
    Y_qq = np.sqrt(BNL_qq / (5 * P_qq))
    def get_ent_wedge(x_grid):
        return np.interp(x_grid, X_qq, Y_qq, left=0, right=0)
    Y_ent_fill = get_ent_wedge(X_vals)
    
    R_ball = np.sqrt(0.2)
    X_ball = np.linspace(0, R_ball, 200)
    Y_ball_plot = np.sqrt(np.maximum(0, R_ball**2 - X_ball**2))

    col_Blue, col_Sep, col_CC, col_Gray,col_Gray_up = "#5AA9E6", "#FFEE99", "#FFEBEB", "lightgray","#B8A597"
    x_prod, x_max_ent = np.sqrt(0.6), np.sqrt(0.1)
    
    mask_left  = X_vals <= x_max_ent          
    mask_mid   = (X_vals > x_max_ent) & (X_vals <= x_prod) 
    mask_tail  = (X_vals > x_prod) 
    Y_qcurve_raw = np.sqrt(np.maximum(0, 1.6 - 2*X_vals**2))

    ax.fill_between(X_vals, 0, Y_qc, where=(X_vals <= x_prod), color=col_CC, zorder=3,alpha=1)
    ax.fill_between(X_vals, 0, Y_qcurve_raw, where=mask_tail, color=col_CC, zorder=3,alpha=1)
    ax.fill_between(X_ball, 0, Y_ball_plot, color=col_Sep, zorder=8,alpha=0.65)
    
    # --- C. Blue Region (Entangled) - MERGED ---
    # 1. Create a single 'Upper Boundary' array for the entire entangled region
    Y_upper_combined = np.where(X_vals <= x_max_ent, Y_ent_fill, Y_R1)

    # 2. Define the total horizontal range for entanglement
    mask_entangled_total = (X_vals <= X_prod)

    # 3. Fill the entire region in ONE call
    ax.fill_between(X_vals, Y_qc, Y_upper_combined, 
                    where=mask_entangled_total, 
                    alpha=0.45, 
                    color=col_Blue, 
                    edgecolor="none", 
                    linewidth=0,
                    zorder=2)

    ax.fill_between(X_vals, Y_ent_fill, Y_R1, where=mask_left, color=col_Gray_up, zorder=3)
    ax.fill_between(X_vals, Y_qcurve_raw, Y_R1, where=mask_tail, color=col_Gray, zorder=4)

    
    
    # --- D. NEW: Root Activation Curve Transformed to X-Y ---
    BL_root = np.linspace(0, 0.125, 300)
    BNL_root = 2/3 + (11 * np.sqrt(2) / 12) * np.sqrt(BL_root)
    P_root = (BL_root + BNL_root + 1) / 6.0
    X_root = np.sqrt(BL_root / (5 * P_root))
    Y_root = np.sqrt(BNL_root / (5 * P_root))
    # E. NEW: Root Activation Pocket (Deep Pink)
    # Create a dedicated X array for filling between 0 and the Kink point to avoid interpolation artifacts
    X_fill_root = np.linspace(0, X_root[-1], 300)
    Y_fill_root_bot = np.interp(X_fill_root, X_root, Y_root)
    Y_fill_root_top = np.interp(X_fill_root, X_qc_true, Y_qc_true)
    ax.fill_between(X_fill_root, Y_fill_root_bot, Y_fill_root_top, color="#F38989", alpha=0.6, zorder=4)
    
    # NEW: Root Activation Curve (Deep Pink)
    ax.plot(X_root, Y_root, color="deeppink", linewidth=2.5, linestyle='-', zorder=12, label="C boundary")

    line_purple = "#975AE6"
    mask_purity_left = X_vals < x_max_ent
    ax.plot(X_vals[mask_purity_left], Y_R1[mask_purity_left], color="#49616E", linewidth=2.5, linestyle='--', zorder=10)
    mask_purity_mid = (X_vals >= x_max_ent) & (X_vals <= x_prod)
    ax.plot(X_vals[mask_purity_mid], Y_R1[mask_purity_mid], color="#274C77", linewidth=2.5, linestyle='-', zorder=10)
    mask_purity_right = X_vals > x_prod
    ax.plot(X_vals[mask_purity_right], Y_R1[mask_purity_right], color="#49616E", linewidth=2.5, linestyle='--', zorder=10)
    ax.plot(X_vals[X_vals <= x_prod], Y_qc[X_vals <= x_prod], color=line_purple, linewidth=2.5, zorder=13)
    ax.plot(X_qq, Y_qq, color="black", linewidth=2.0, zorder=9)
    ax.plot(X_ball, Y_ball_plot, color="#D62728", linestyle="--", linewidth=2.5, zorder=15)
    ax.plot(X_vals[mask_tail], Y_qcurve_raw[mask_tail], color="black", linewidth=2.5, linestyle='-', zorder=8)
    X_limit = np.sqrt(0.8)
    ax.plot([0, X_limit], [0, 0], color="red", linewidth=1.5, linestyle='--', zorder=11)

    _plot_trajectories_on_ax(ax, trajectories_by_state, mode="XY")

    ax.set_xlabel(r"$X$", fontsize=20)
    ax.set_ylabel(r"$Y$", fontsize=20)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.pdf", dpi=300)
    plt.show()

# --- B. BL-BNL Plotter (Perfected Native Geometry) ---
def plot_geometry_blbnl(trajectories_by_state, save_prefix="QubitQutrit_BLBNL"):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Base Grids
    bl_grid = np.linspace(0, 3.0, 1000)
    
    # 2. Exact Piecewise Linear QC Envelope (Purple Line)
    bnl_qc_exact = np.zeros_like(bl_grid)
    mask1 = bl_grid <= 0.5
    bnl_qc_exact[mask1] = 1.0 + bl_grid[mask1]
    mask2 = bl_grid > 0.5
    bnl_qc_exact[mask2] = 1.5 + 0.2 * (bl_grid[mask2] - 0.5)
    
    # 3. Upper Entanglement Boundary (Wedge)
    w = np.linspace(2/3, 1, 300)
    w2 = w**2
    bl_wedge = 4.5*w2 - 6*w + 2   
    bnl_wedge = 4.5*w2            
    
    # 4. Purity Arc (Absolute Upper Limit)
    bl_pure = np.linspace(0.5, 3.0, 300)
    bnl_pure = 5.0 - bl_pure
    
    bl_top_pts = np.concatenate([bl_wedge, bl_pure])
    bnl_top_pts = np.concatenate([bnl_wedge, bnl_pure])
    sort_idx = np.argsort(bl_top_pts)
    bnl_top_interp = np.interp(bl_grid, bl_top_pts[sort_idx], bnl_top_pts[sort_idx])

    # 5. Kinematic Floor (The physical boundary Q=0 for B_NL)
    bnl_floor = np.maximum(0, 2.0 * bl_grid - 4.0)

    # 6. Sep Ball Line
    bnl_ball = np.maximum(0, 0.2 - bl_grid)

    # --- FILL REGIONS ---
    col_Blue, col_Sep, col_CC, col_Gray, col_Gray_up = "#5AA9E6", "#FFEE99", "#FFEBEB", "lightgray", "#B8A597"
    
    # Non-Physical tail (Q < 0) - Bounded between 0 and the floor for BL > 2.0
    ax.fill_between(bl_grid, 0, bnl_floor, 
                    where=(bl_grid > 2.0), 
                    color=col_Gray, zorder=3)

    # Classical (Pink) - Bounded perfectly between the physical floor and QC envelope
    ax.fill_between(bl_grid, bnl_floor, bnl_qc_exact, 
                    color=col_CC, zorder=3, alpha=1.0)
    
    # Entangled (Blue)
    ax.fill_between(bl_grid, bnl_qc_exact, bnl_top_interp, 
                    color=col_Blue, zorder=2, alpha=0.45, edgecolor="none", linewidth=0)
    
    # Upper Gray Pocket (Only present for BL < 0.5 above the wedge)
    bnl_purity_limit = 5.0 - bl_grid
    ax.fill_between(bl_grid, bnl_top_interp, bnl_purity_limit, 
                    where=(bl_grid < 0.5), color=col_Gray_up, zorder=2)

    # Sep Ball (Orange)
    ax.fill_between(bl_grid, 0, bnl_ball, 
                    where=(bl_grid <= 0.2), color=col_Sep, zorder=8, alpha=0.65)
    

    # --- E. NEW: Root Activation Curve ---
    bl_root = np.linspace(0, 0.125, 300)
    bnl_root = 2/3 + (11 * np.sqrt(2) / 12) * np.sqrt(bl_root)
    # Get the corresponding top boundary (QC Envelope) for the shaded region
    QC_BL_Exact = [0, 0.5, 3.0]
    QC_BNL_Exact = [1.0, 1.5, 2.0]
    def get_qc_exact_bnl(bl):
        return np.interp(bl, QC_BL_Exact, QC_BNL_Exact, left=np.nan, right=np.nan)
    bnl_root_top = get_qc_exact_bnl(bl_root)
    ax.fill_between(bl_root, bnl_root, bnl_root_top, color="#F38989", alpha=0.6, zorder=4)
    ax.plot(bl_root, bnl_root, color="deeppink", lw=2.5, ls='-', zorder=12, label="C boundary")

    # --- BOUNDARY LINES ---
    # Exact QC Line (Purple)
    ax.plot(bl_grid, bnl_qc_exact, color="#975AE6", lw=2.5, zorder=13)
    
    # Entanglement Boundary Wedge (Black)
    ax.plot(bl_wedge, bnl_wedge, color="black", lw=2.0, zorder=9)
    
    # Purity Arc Lines (Dashed for BL < 0.5, Solid for BL >= 0.5)
    bl_pur_left = np.linspace(0, 0.5, 100)
    ax.plot(bl_pur_left, 5.0 - bl_pur_left, color="#49616E", lw=2.5, ls='--', zorder=10)
    bl_pur_mid = np.linspace(0.5, 3.0, 100)
    ax.plot(bl_pur_mid, 5.0 - bl_pur_mid, color="#274C77", lw=2.5, ls='-', zorder=10)

    # Kinematic Floor Lines
    # Red dashed for BL from 0 to 2
    ax.plot([0, 2], [0, 0], color="red", lw=1.5, ls='--', zorder=11)
    # Black solid tail from BL 2 to 3
    bl_tail = np.linspace(2, 3, 100)
    ax.plot(bl_tail, 2.0 * bl_tail - 4.0, color="black", lw=2.5, ls='-', zorder=11)
    
    # Ball Boundary Line
    bl_ball_line = np.linspace(0, 0.2, 100)
    ax.plot(bl_ball_line, 0.2 - bl_ball_line, color="#D62728", ls='--', lw=2.5, zorder=15)

    # --- TRAJECTORIES ---
    _plot_trajectories_on_ax(ax, trajectories_by_state, mode="BLBNL")

    ax.set_xlabel(r"$B_\mathrm{L}$", fontsize=20)
    ax.set_ylabel(r"$B_\mathrm{NL}$", fontsize=20)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 5.0)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.pdf", dpi=300)
    plt.show()

def _plot_trajectories_on_ax(ax, trajectories_by_state, mode="XY"):
    channel_colors = {
        "amplitude_damping": "#E61089", "correlated_amplitude_damping": "#0257E0", 
        "bit_flip": "#03AD3F", "phase_flip": "#FF9100", "dephasing": "#FF9100", "depolarizing": "#FF3C00","mem_generation": "#5CD68F"
    }
    marker_map = {"A": "D", "B": "s", "AB": "X", "corr": "h"}
    start_colors = {"Bell": "#99CAF0", "Product": "#FF0000", "PPT": "#FFEE99", "NPT": "#5AA9E6", "Classical": "#BBBBBB","MEM": "#5CD68F"}
    legend_name_map = {
        "depolarizing_A": "Depolarising (Qubit)", "depolarizing_B": "Depolarising (Qutrit)", "depolarizing_AB": "Depolarising (Both)",
        "amplitude_damping_A": "Amplitude damping (Qubit)", "amplitude_damping_B": "Amplitude damping (Qutrit)", "amplitude_damping_AB": "Amplitude damping (Both)",
        "correlated_amplitude_damping_corr": "Correlated amplitude damping","dephasing_B": "Dephasing (Qutrit)",
        "mem_generation_AB": "Rank-2 MEMS ansatz"
    }
    
    legend_handles = OrderedDict()
    
    for state_label, traj_dict in trajectories_by_state.items():
        s_color = "#CCCCCC"
        for k, v in start_colors.items():
            if k == state_label:
                s_color = v
                break
            
        for key, traj in traj_dict.items():
            if not traj: continue
            parts = key.split('_')
            target = parts[-1]
            channel = "_".join(parts[:-1])
            
            xs_raw = np.array([p.X for p in traj])
            ys_raw = np.array([p.Y for p in traj])
            
            if mode == "BLBNL":
                Xs, Ys = xy_to_blbnl(xs_raw, ys_raw)
            else:
                Xs, Ys = xs_raw, ys_raw
            
            color = channel_colors.get(channel, "black")
            current_zorder = 50
            if channel == "depolarizing" and target == "AB": current_zorder = 200

            if target == 'A': dashes = [6, 3]
            elif target == 'B': dashes = [2, 2]
            elif target == 'AB': dashes = [1, 0]
            elif target == 'corr': dashes = [1, 0]

            end_marker = marker_map.get(target, "o")
            if channel == "amplitude_damping" and target == "AB": end_marker = "*"

            line, = ax.plot(Xs, Ys, color=color, linewidth=2.0, alpha=0.9, zorder=current_zorder)
            line.set_dashes(dashes)
            
            ax.scatter(Xs[0], Ys[0], s=60, marker='o', facecolor=s_color, 
                       edgecolor='black', zorder=current_zorder + 10, clip_on=False)
            ax.scatter(Xs[-1], Ys[-1], s=80, marker=end_marker, facecolor=color,
                       edgecolor='black', zorder=current_zorder + 10, clip_on=False)
            
            l_key = f"{channel}_{target}"
            l_name = legend_name_map.get(l_key, l_key)
            if l_name not in legend_handles:
                from matplotlib.lines import Line2D
                h = Line2D([0], [0], color=color, lw=2, marker=end_marker, 
                           markeredgecolor='black', markerfacecolor=color)
                try: h.set_dashes(dashes)
                except: pass
                legend_handles[l_name] = h

    ax.legend(legend_handles.values(), legend_handles.keys(), loc="upper right", frameon=False, fontsize=10)
    ax.grid(alpha=0.25)

# =============================
#  7. Main Execution
# =============================

def main():
        np.random.seed(0) # MixedEntangled, Sep: 41, PureProd: 1267, 0 for classical
        print("Generating states...")
        states = {
            #   "Bell": state_pure_entangled_max(),
                # "NPT": state_mixed_general(target="NPT", min_Y=0.8),
            #  "Product": state_pure_product(),
                # "PPT": state_mixed_general(target="PPT"),
               "Classical": state_classical_dirichlet(),
            #    "MEM": state_mem_critical()
        }

        configs = [
            ("depolarizing", ["AB", "A", "B"]),
            ("amplitude_damping", ["AB", "A", "B"]),
            ("correlated_amplitude_damping", ["corr"]), 
            #  ("dephasing", [ "B"]),
            #   ("mem_generation", ["AB"])
        ]

        p_steps = np.linspace(0, 1.0, 40)

        results = {}
        for name, rho in states.items():
            print(f"Simulating trajectories for: {name}")
            results[name] = {}
            for ch_name, targets in configs:
                for t in targets:
                    key = f"{ch_name}_{t}"
                    traj = generate_trajectory(rho, ch_name, t, p_steps)
                    results[name][key] = traj

        if PLOT_MODE in ["XY", "BOTH"]:
            print("Plotting XY Plane...")
            plot_geometry_xy(results)

        if PLOT_MODE in ["BLBNL", "BOTH"]:
            print("Plotting BL-BNL Plane...")
            plot_geometry_blbnl(results)

if __name__ == "__main__":
    main()