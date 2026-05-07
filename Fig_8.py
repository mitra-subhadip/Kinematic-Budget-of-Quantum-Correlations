#!/usr/bin/env python3
"""
Decoherence trajectories in two-qutrit geometry (XY or BL-BNL).

Updates:
  - "Biased Mixing" generator for NPT states (Mixes Bell + Noise instead of Product + Noise).
  - Ensures NPT states are strictly above the CC envelope.
  - Generates both XY and BL-BNL plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from dataclasses import dataclass
from collections import OrderedDict
import os

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
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# =============================
#  2. Math Helpers (Qutrit d=3)
# =============================

def dagger(M):
    return np.conjugate(M).T

def tensor(a, b):
    return np.kron(a, b)

# Identity
I3 = np.eye(3, dtype=complex)

def partial_transpose(rho, dims=(3, 3)):
    """Computes Partial Transpose (TB) for d=3."""
    rho = rho.reshape(3, 3, 3, 3)
    rho_pt = rho.transpose(0, 3, 2, 1) 
    return rho_pt.reshape(9, 9)

def get_min_eigenvalue(rho):
    return np.min(np.linalg.eigvalsh(rho))

def is_ppt(rho):
    """Returns True if state is Positive Partial Transpose."""
    pt = partial_transpose(rho)
    return get_min_eigenvalue(pt) >= -1e-9

def partial_trace(rho, keep='A', dims=(3, 3)):
    d1, d2 = dims
    rho = rho.reshape(d1, d2, d1, d2)
    if keep == 'A':
        return np.einsum('ijkj->ik', rho)
    else:
        return np.einsum('ijil->jl', rho)

# =============================
#  3. Coordinate Transformation
# =============================

@dataclass
class BudgetPoint:
    X: float
    Y: float
    BL: float  # Added
    BNL: float # Added
    R: float
    p: float
    label: str

def rho_to_coords_qutrit(rho):
    rho = np.asarray(rho, dtype=complex)
    P = np.real(np.trace(rho @ rho))
    if P < 1e-9: P = 1e-9
    
    rhoA = partial_trace(rho, keep='A', dims=(3,3))
    rhoB = partial_trace(rho, keep='B', dims=(3,3))
    pA = np.real(np.trace(rhoA @ rhoA))
    pB = np.real(np.trace(rhoB @ rhoB))
    
    B_L = 3.0 * (pA + pB) - 2.0
    B_tot = 9.0 * P - 1.0
    B_NL = B_tot - B_L
    
    B_L = max(0.0, B_L)
    B_NL = max(0.0, B_NL)
    
    X_sq = B_L / (8.0 * P)
    Y_sq = B_NL / (8.0 * P)
    
    # Return raw budgets as well
    return np.sqrt(X_sq), np.sqrt(Y_sq), B_L, B_NL, X_sq + Y_sq

# =============================
#  4. State Generators
# =============================

def random_ket(d=3):
    v = np.random.normal(0, 1, size=d) + 1j * np.random.normal(0, 1, size=d)
    return v / np.linalg.norm(v)

def ginibre_mixed_state(dim=9):
    Gr = np.random.normal(0, 1, size=(dim, dim))
    Gi = np.random.normal(0, 1, size=(dim, dim))
    G = Gr + 1j * Gi
    rho = G @ dagger(G)
    rho /= np.trace(rho)
    return rho

def qutrit_bell_state_fixed():
    """Specific Bell state: |Phi+>"""
    v = np.zeros(9, dtype=complex)
    v[0] = v[4] = v[8] = 1.0 / np.sqrt(3)
    return np.outer(v, v.conj())

def qutrit_pure_product_general():
    """Generalized Random Pure Product State."""
    va = random_ket(3)
    vb = random_ket(3)
    v = tensor(va, vb)
    return np.outer(v, v.conj())

def qutrit_classical_cc_state():
    """Random Classical-Classical State (Dirichlet)."""
    p = np.random.dirichlet(np.ones(9)) 
    return np.diag(p).astype(complex)

def qutrit_mixed_polarized_general(target_type="PPT"):
    """
    Finds a mixed state.
    - If PPT: Mixes Product + Noise (biases towards separable).
    - If NPT: Mixes Maximally Entangled + Noise (biases towards high B_NL).
      Ensures NPT states are strictly ABOVE the CC envelope.
    """
    max_iter = 50000 # Increased steps significantly
    
    for i in range(max_iter):
        mix = np.random.uniform(0.1, 0.9)
        rho_ginibre = ginibre_mixed_state(dim=9)
        
        # === METHOD CHANGE ===
        # Use different base states to target different regions
        if target_type == "NPT":
            # Start from Bell state (Top of graph) and mix down
            rho_base = qutrit_bell_state_fixed()
        else:
            # Start from Product state (Bottom/Right of graph)
            rho_base = qutrit_pure_product_general()

        rho_cand = (1.0 - mix) * rho_base + mix * rho_ginibre
        
        is_ppt_state = is_ppt(rho_cand)
        
        # Calculate Coordinates to check envelope
        _, _, BL, BNL, _ = rho_to_coords_qutrit(rho_cand)
        
        # Define CC Envelope for checking
        BNL_upper = 8 - BL if BL <= 4 else 0 
        BNL_cc_boundary = (BL + 4) / 2
        if BL > 4: BNL_cc_boundary = 0
        
        if target_type == "PPT" and is_ppt_state:
            return rho_cand
        elif target_type == "NPT" and not is_ppt_state:
            # Strict check: Must be ABOVE CC envelope
            if BNL > BNL_cc_boundary:
                # print(f"Found NPT state above CC line at iter {i}")
                return rho_cand
            
    print(f"Warning: Could not find suitable {target_type} state after {max_iter} steps.")
    return rho_cand

# =============================
#  5. Channel Definitions
# =============================

def amplitude_damping_kraus(p):
    K0 = np.array([[1, 0, 0], [0, np.sqrt(1-p), 0], [0, 0, np.sqrt(1-p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p), 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
    K2 = np.array([[0, 0, np.sqrt(p)], [0, 0, 0], [0, 0, 0]], dtype=complex)
    return [K0, K1, K2]

def bit_flip_kraus(p):
    K0 = np.sqrt(1-p) * I3
    Shift = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    RevShift = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
    K1 = np.sqrt(p/2.0) * Shift
    K2 = np.sqrt(p/2.0) * RevShift
    return [K0, K1, K2]

def phase_flip_kraus(p):
    K0 = np.sqrt(1-p) * I3
    P1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=complex)
    P2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=complex)
    K1 = np.sqrt(p/2.0) * P1
    K2 = np.sqrt(p/2.0) * P2
    return [K0, K1, K2]

def depolarizing_kraus(p):
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

def fcad_kraus(p):
    A1 = np.zeros((9,9), dtype=complex); A1[0, 4] = np.sqrt(p)
    A2 = np.zeros((9,9), dtype=complex); A2[0, 8] = np.sqrt(p)
    diag = np.ones(9, dtype=complex)
    diag[4] = np.sqrt(1-p); diag[8] = np.sqrt(1-p)
    A0 = np.diag(diag)
    return [A0, A1, A2]

# --- Application Logic ---
def apply_single_qutrit_kraus(rho, kraus_ops, which='A'):
    rho = rho.reshape(9, 9)
    new_rho = np.zeros_like(rho, dtype=complex)
    if which == 'AB':
        for Ka in kraus_ops:
            for Kb in kraus_ops:
                K_full = tensor(Ka, Kb)
                new_rho += K_full @ rho @ dagger(K_full)
    else:
        for K in kraus_ops:
            K_full = tensor(K, I3) if which == 'A' else tensor(I3, K)
            new_rho += K_full @ rho @ dagger(K_full)
    return new_rho

def apply_correlated_ad(rho, p, mu):
    Ks_single = amplitude_damping_kraus(p)
    rho_uncorr = apply_single_qutrit_kraus(rho, Ks_single, which='AB')
    Ks_fcad = fcad_kraus(p)
    rho_fcad = np.zeros_like(rho, dtype=complex)
    for K in Ks_fcad:
        rho_fcad += K @ rho @ dagger(K)
    return (1 - mu) * rho_uncorr + mu * rho_fcad

# =============================
#  6. Simulation
# =============================

def generate_trajectory(rho0, channel_name, which, p_values, mu=0.5):
    traj = []
    for p in p_values:
        if channel_name == 'correlated_amplitude_damping':
            rho_t = apply_correlated_ad(rho0, p, mu)
        else:
            if channel_name == 'amplitude_damping':   Ks = amplitude_damping_kraus(p)
            elif channel_name == 'bit_flip':          Ks = bit_flip_kraus(p)
            elif channel_name == 'phase_flip':        Ks = phase_flip_kraus(p)
            elif channel_name == 'depolarizing':      Ks = depolarizing_kraus(p)
            else: raise ValueError(f"Unknown channel {channel_name}")
            rho_t = apply_single_qutrit_kraus(rho0, Ks, which=which)
        
        # Calculate coords
        X, Y, BL, BNL, R = rho_to_coords_qutrit(rho_t)
        traj.append(BudgetPoint(X, Y, BL, BNL, R, p, f"{channel_name}_{which}"))
    return traj

# =============================
#  7. PLOTTING
# =============================

def plot_geometry_with_trajectories(trajectories_by_state, mode="XY", save_prefix="Qutrit_Geometry"):
    """
    mode: "XY" or "BL_BNL"
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Common Colors ---
    col_R1, col_CC, col_Sep = "#5AA9E6", "#FFEBEB", "#FFEE99"
    col_Unfeas = "lightgray"
    purple_line = "#975AE6"
    ppt_color = "#FF6392"
    
    if mode == "XY":
        # === XY PLANE GEOMETRY ===
        X = np.linspace(0, 1.02, 2000)
        Y_R1 = np.sqrt(np.maximum(0, 1 - X**2))
        Y_pos = np.sqrt(np.maximum(0, 1.5 - 2*X**2))
        Y_cc_raw = np.sqrt(np.maximum(0, (1.5 - X**2)/2))
        Y_sep = np.sqrt(np.maximum(0, 0.125 - X**2))
        
        x_pure_prod, x_pure_mixed = 1/np.sqrt(2), np.sqrt(3)/2
        Y_feasible = np.zeros_like(X)
        mask_pure = X <= x_pure_prod
        mask_pos  = (X > x_pure_prod) & (X <= x_pure_mixed)
        Y_feasible[mask_pure] = Y_R1[mask_pure]
        Y_feasible[mask_pos]  = Y_pos[mask_pos]
        Y_cc = np.minimum(Y_cc_raw, Y_feasible)

        # Fills
        ax.fill_between(X, 0, Y_feasible, color=col_R1, alpha=0.45, zorder=1)
        ax.fill_between(X, 0, Y_cc, color=col_CC, alpha=1, zorder=3)
        ax.fill_between(X, 0, Y_sep, color=col_Sep, alpha=0.65, zorder=4)
        
        mask_fill = (X > x_pure_prod) & (X <= 1.0)
        ax.fill_between(X[mask_fill], Y_feasible[mask_fill], Y_R1[mask_fill], color=col_Unfeas, zorder=20)

        # Lines
        ax.plot(X[mask_pure], Y_R1[mask_pure], color="#274C77", linewidth=2.5, zorder=22)
        mask_dotted = (X > x_pure_prod) & (X <= 1.0)
        if np.any(mask_dotted):
            ax.plot(np.append(X[mask_dotted], 1.0), np.append(Y_R1[mask_dotted], 0.0), 
                    "--", color="#49616E", linewidth=2.5, zorder=22)
        
        X_pos_pl = np.append(X[mask_pos], x_pure_mixed)
        Y_pos_pl = np.append(Y_pos[mask_pos], 0.0)
        ax.plot(X_pos_pl, Y_pos_pl, color="black", linewidth=2.5, zorder=23)

        ax.plot(X[X <= x_pure_prod], Y_cc_raw[X <= x_pure_prod], color=purple_line, linewidth=2.5, zorder=24)
        ax.plot(X[X <= x_pure_mixed], Y_sep[X <= x_pure_mixed], "--", color=ppt_color, linewidth=2.5, zorder=24)
        ax.plot([0, x_pure_mixed], [0, 0], color="red", linewidth=1.5, zorder=25)

        # Labels
        ax.set_xlabel("$X$", fontsize=20)
        ax.set_ylabel("$Y$", fontsize=20)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

    elif mode == "BL_BNL":
        # === BL-BNL PLANE GEOMETRY ===
        BL = np.linspace(0, 8.5, 2000)
        
        # 1. Purity Line (Global Purity P=1 => BL + BNL = 8)
        BNL_pure_line = np.maximum(0, 8 - BL)

        # 2. Polygon Boundaries
        BNL_upper = np.where(BL <= 4, 8 - BL, 0)
        BNL_lower = np.maximum(0, 2 * BL - 4)
        
        # 3. Envelopes
        BNL_cc_raw = (BL + 4) / 2
        BNL_cc = np.minimum(BNL_upper, BNL_cc_raw)
        BNL_cc = np.maximum(BNL_cc, BNL_lower)
        
        BNL_sep_raw = 0.125 - BL
        BNL_sep = np.minimum(BNL_upper, BNL_sep_raw)
        BNL_sep = np.maximum(BNL_sep, BNL_lower)
        BNL_sep = np.where(BL <= 0.125, BNL_sep, 0)
        
        mask_poly = (BL >= 0) & (BL <= 4)
        
        # Fills
        ax.fill_between(BL[mask_poly], BNL_lower[mask_poly], BNL_upper[mask_poly], color=col_R1, alpha=0.45, zorder=1)
        ax.fill_between(BL[mask_poly], BNL_lower[mask_poly], BNL_cc[mask_poly], color=col_CC, alpha=1, zorder=3)
        ax.fill_between(BL[BL <= 0.125], 0, BNL_sep[BL <= 0.125], color=col_Sep, alpha=0.65, zorder=8)
        
        # Unfeasible Grey
        mask_unfeas_right = (BL > 4) & (BL <= 8)
        ax.fill_between(BL[mask_unfeas_right], 0, BNL_pure_line[mask_unfeas_right], color=col_Unfeas, zorder=5)
        mask_unfeas_bottom = (BL >= 2) & (BL <= 4)
        ax.fill_between(BL[mask_unfeas_bottom], 0, BNL_lower[mask_unfeas_bottom], color=col_Unfeas, zorder=5)
        
        # Lines
        ax.plot(BL[mask_poly], BNL_upper[mask_poly], color="#274C77", linewidth=2.5, zorder=6)
        ax.plot(BL[mask_unfeas_right], BNL_pure_line[mask_unfeas_right], "--", color="#274C77", linewidth=2.5, zorder=6)
        
        mask_wall = (BL >= 2) & (BL <= 4)
        ax.plot(BL[mask_wall], BNL_lower[mask_wall], color="black", linewidth=2.5, zorder=7)
        ax.plot([0, 2], [0, 0], color="black", linewidth=2.5, zorder=7)
        
        # CC Envelope Line
        mask_cc_line = (BL <= 4) 
        ax.plot(BL[mask_cc_line], BNL_cc_raw[mask_cc_line], color=purple_line, linewidth=2.5, zorder=8)
        
        # Separable Line
        mask_sep_line = (BL <= 0.125)
        ax.plot(BL[mask_sep_line], BNL_sep_raw[mask_sep_line], "--", color=ppt_color, linewidth=2.5, zorder=8)
        
        # Baseline
        ax.plot([0, 2], [0, 0], "--", color="red", linewidth=1.5, zorder=9)

        # Labels
        ax.set_xlabel(r"$B_\mathrm{L}$", fontsize=20)
        ax.set_ylabel(r"$B_\mathrm{NL}$", fontsize=20)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 8)

    # ax.set_aspect("equal")
    ax.grid(alpha=0.25)

    # --- Trajectory Styling ---
    channel_colors = {
        "amplitude_damping": "#E61089",            # Pink
        "correlated_amplitude_damping": "#0257E0", # Blue
        "bit_flip": "#03AD3F",                     # Green
        "phase_flip": "#FF9100",                   # Orange
        "depolarizing": "#FF3C00"                  # Red
    }

    marker_map = {
        "A": "D",    "B": "s",    "AB": "X",   "corr": "h"
    }

    # START COLORS
    start_colors = {
        "Bell": "#99CAF0",    # Blue (Pure Entangled)
        "Product": "#FF0000", # Red (Pure Product)
        "PPT": "#FFEE99",     # Yellow (Mixed Separable)
        "NPT": "#5AA9E6",     # Blue (Mixed Entangled)
        "UPB": "#B19CD9",     # Pastel Purple
        "Classical": "#BBBBBB" # Pastel Orange
    }

    # LEGEND MAPPING
    legend_name_map = {
        "dephasing_B": "Dephasing (one-sided)",
        "dephasing_AB": "Dephasing",
        "dephasing_A": "Dephasing (one-sided)",
        
        "phase_flip_A": "Dephasing (one-sided)",
        "phase_flip_B": "Dephasing (one-sided)",
        "phase_flip_AB": "Dephasing",

        "bit_flip_A":   "Bit flip (one-sided)",
        "bit_flip_B":   "Bit flip (one-sided)",
        "bit_flip_AB":  "Bit flip",

        "depolarizing_A": "Depolarising (one-sided)",
        "depolarizing_B": "Depolarising (one-sided)",
        "depolarizing_AB": "Depolarising",

        "amplitude_damping_A": "Amplitude damping (one-sided)",
        "amplitude_damping_B": "Amplitude damping (one-sided)",
        "amplitude_damping_AB": "Amplitude damping",

        "correlated_phase_flip_corr": "Correlated phase flip",
        "correlated_amplitude_damping_corr": "Correlated amplitude damping",
    }

    legend_order = OrderedDict()

    for state_label, traj_dict in trajectories_by_state.items():
        s_color = "#CCCCCC"
        for k, v in start_colors.items():
            if k in state_label: s_color = v

        for key, traj in traj_dict.items():
            if not traj: continue
            
            parts = key.split('_')
            target = parts[-1]
            channel = "_".join(parts[:-1])
            
            # SWITCH DATA BASED ON MODE
            if mode == "BL_BNL":
                Xs, Ys = [p.BL for p in traj], [p.BNL for p in traj]
            else:
                # Default to XY
                Xs, Ys = [p.X for p in traj], [p.Y for p in traj]

            color = channel_colors.get(channel, "gray")
            
            if channel == "amplitude_damping" and target == "AB":
                end_marker = "*"
            else:
                end_marker = marker_map.get(target, "o")
            
            if target in ['A', 'B']: dashes = [2, 1] 
            else: dashes = [1, 0] 

            line, = ax.plot(Xs, Ys, color=color, linewidth=2, alpha=0.8, zorder=50)
            line.set_dashes(dashes)
            
            ax.scatter(Xs[0], Ys[0], s=60, marker='o', facecolor=s_color, 
                       edgecolor='black', linewidth=0.9, zorder=200, clip_on=False)
            
            ax.scatter(Xs[-1], Ys[-1], s=80, marker=end_marker, facecolor=color,
                       edgecolor='black', linewidth=0.9, zorder=300, clip_on=False)

            # --- Legend Logic ---
            legend_key = f"{channel}_{target}"
            if legend_key in legend_name_map:
                legend_label = legend_name_map[legend_key]
            else:
                legend_label = f"{channel.replace('_', ' ').title()} ({target})"

            if legend_label not in legend_order:
                from matplotlib.lines import Line2D
                proxy = Line2D([0], [0], 
                               color=color,
                               linewidth=0.001, 
                               marker=end_marker, 
                               markersize=9, 
                               markeredgecolor='black', 
                               markerfacecolor=color)
                try: proxy.set_dashes(dashes)
                except: pass
                
                legend_order[legend_label] = proxy

    handles = list(legend_order.values())
    labels = list(legend_order.keys())
    
    # from matplotlib.lines import Line2D
    # handles.append(Line2D([0], [0], color=purple_line, lw=2.5)); labels.append("CC/QC")
    # handles.append(Line2D([0], [0], color=ppt_color, lw=2.5, linestyle="--")); labels.append("Separable")

    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=10, ncol=1)
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{mode}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    np.random.seed(11) # 45 for Pure Prod, 2 for Mixed PPT & NPT, 11 for Classical
    
    # === CHOOSE MODE HERE ===
    # Options: "XY", "BL_BNL", or "BOTH"
    view_mode = "BOTH"
    
    p_values = np.linspace(0.0, 1.0, 50)
    mu_corr = 1
    
    # States
    initial_states = {
        #   "Bell State": qutrit_bell_state_fixed(),
        #    "Pure Product": qutrit_pure_product_general(),
            # "Mixed PPT": qutrit_mixed_polarized_general(target_type="PPT"),
        #    "Mixed NPT": qutrit_mixed_polarized_general(target_type="NPT"),
          "Classical CC": qutrit_classical_cc_state(),
           }
    
    # Channels
    sim_configs = [
        ("depolarizing", ["AB",'A']),
        ("amplitude_damping", ["AB",'A']),
        ("correlated_amplitude_damping", ["corr"]),
        # ("dephasing", ['A']),
        # ("phase_flip", ["A"]),
    ]
    
    # 1. GENERATE TRAJECTORIES ONCE
    trajectories = {}
    for name, rho in initial_states.items():
        trajectories[name] = {}
        for ch, targets in sim_configs:
            for t in targets:
                traj = generate_trajectory(rho, ch, t, p_values, mu=mu_corr)
                trajectories[name][f"{ch}_{t}"] = traj
    
    # 2. PLOT BASED ON MODE
    if view_mode == "BOTH":
        print("Generating XY Plot...")
        plot_geometry_with_trajectories(trajectories, mode="XY", save_prefix="Qutrit_Analysis")
        print("Generating BL-BNL Plot...")
        plot_geometry_with_trajectories(trajectories, mode="BL_BNL", save_prefix="Qutrit_Analysis")
    else:
        plot_geometry_with_trajectories(trajectories, mode=view_mode, save_prefix="Qutrit_Analysis")

if __name__ == "__main__":
    main()