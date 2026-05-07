#!/usr/bin/env python3
"""
Decoherence trajectories in the THREE-QUBIT budget geometry.
Features:
  - Exact 3-Qubit Geometry boundaries (Geometric & Budget planes).
  - Generators: Full Suite (Product, Bisep, GME, W, Classical).
  - Multi-Target Noise: One-sided (A,B,C), Two-sided (AB,BC,AC), Global (ABC).
  - Programmatic styling matching the 2-qubit reference script.
  - Werner-like state implementation for Mixed Biseparable states.
  - Outputs TWO separate PDFs: Geometric Plane (X vs Y) and Budget Plane (BL vs BNL).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from dataclasses import dataclass
from collections import OrderedDict
import os
import scipy.linalg as la

# =============================
#  1. Style Configuration
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
#  2. Math Helpers
# =============================
def dagger(M):
    return np.conjugate(M).T

def tensor(*args):
    res = args[0]
    for mat in args[1:]:
        res = np.kron(res, mat)
    return res

I2 = np.eye(2, dtype=complex)

def partial_trace_3qubit(rho, keep_indices):
    rho = rho.reshape(2, 2, 2, 2, 2, 2)
    distinct_rows = ['a', 'b', 'c']
    distinct_cols = ['d', 'e', 'f']
    trace_chars   = ['x', 'y', 'z']
    
    in_subs = [''] * 6
    out_subs_row = []
    out_subs_col = []
    
    for q in range(3):
        if q in keep_indices:
            r_char = distinct_rows[q]
            c_char = distinct_cols[q]
            in_subs[q]     = r_char
            in_subs[q + 3] = c_char
            out_subs_row.append(r_char)
            out_subs_col.append(c_char)
        else:
            t_char = trace_chars[q]
            in_subs[q]     = t_char
            in_subs[q + 3] = t_char

    ein_in = "".join(in_subs)
    ein_out = "".join(out_subs_row) + "".join(out_subs_col)
    result = np.einsum(f"{ein_in}->{ein_out}", rho)
    dim_out = 2 ** len(keep_indices)
    return result.reshape(dim_out, dim_out)

def rho_to_coords_3qubit(rho):
    rho = np.asarray(rho, dtype=complex)
    P = np.real(np.trace(rho @ rho))
    if P < 1e-9: P = 1e-9
    
    B_L = 0.0
    for k in range(3):
        rho_k = partial_trace_3qubit(rho, keep_indices=(k,))
        pk = np.real(np.trace(rho_k @ rho_k))
        B_L += (2.0 * pk - 1.0)
        
    B_tot = 8.0 * P - 1.0
    B_NL = B_tot - B_L
    
    # Ensure non-negative before square root
    B_L = max(0.0, B_L)
    B_NL = max(0.0, B_NL)
    
    norm = 7.0 * P
    X_sq = B_L / norm
    Y_sq = B_NL / norm
    
    return np.sqrt(X_sq), np.sqrt(Y_sq), B_L, B_NL

# =============================
#  3. State Generators
# =============================
def random_ket_d(d):
    v = np.random.normal(0, 1, size=d) + 1j * np.random.normal(0, 1, size=d)
    return v / np.linalg.norm(v)

def random_density_d(d):
    A = np.random.normal(0, 1, (d, d)) + 1j * np.random.normal(0, 1, (d, d))
    rho = A @ dagger(A)
    return rho / np.trace(rho)

def random_pure_product():
    v = tensor(random_ket_d(2), random_ket_d(2), random_ket_d(2))
    return np.outer(v, v.conj())

def random_pure_pure_mixed():
    v = tensor(random_ket_d(2), random_ket_d(2))
    pure = np.outer(v, v.conj())
    rho = tensor(pure, np.eye(2)/2)
    return rho

def random_pure_biseparable(min_concurrence=0.1):
    """
    Generates a pure biseparable state: |psi_A> (x) |phi_BC>.
    Explicitly checks that the BC pair is entangled using Concurrence.
    """
    while True:
        # 1. Generate the independent 1-qubit state
        q_pure = random_ket_d(2)
        
        # 2. Generate the 2-qubit state
        pair_pure = random_ket_d(4) 
        
        # 3. Check entanglement of the 2-qubit pair
        # pair_pure = [alpha, beta, gamma, delta]
        concurrence = 2 * np.abs(pair_pure[0]*pair_pure[3] - pair_pure[1]*pair_pure[2])
        
        # 4. If it is sufficiently entangled, construct the 3-qubit state and break
        if concurrence >= min_concurrence:
            v = tensor(q_pure, pair_pure)
            return np.outer(v, v.conj())

def random_mixed_separable(n_mix=6):
    rho = np.zeros((8,8), dtype=complex)
    weights = np.random.dirichlet(np.ones(n_mix))
    for w in weights:
        rho += w * random_pure_product()
    return rho

def random_mixed_biseparable(p=0.85):
    """
    Generates a mixed biseparable state using a pure state and a Werner state 
    to strictly lie above the CCQ envelope and below the pure P=1 line.
    Modified so Qubits A & B are entangled, and Qubit C is separable.
    """
    # 1. Create the entangled Werner state for Qubits A & B
    v_bell = np.zeros(4, dtype=complex)
    v_bell[0] = 1.0 / np.sqrt(2)
    v_bell[3] = 1.0 / np.sqrt(2)
    rho_bell = np.outer(v_bell, v_bell.conj())
    rho_werner = p * rho_bell + (1 - p) * np.eye(4) / 4.0
    
    # 2. Create the independent pure state for Qubit C
    q_pure = random_ket_d(2)
    rho_C = np.outer(q_pure, q_pure.conj())
    
    # 3. Assemble the 3-qubit state: AB (x) C
    rho = tensor(rho_werner, rho_C)
    return rho

def random_mixed_gme_nonzero_bloch():
    theta = np.random.uniform(0.1, 0.35 * np.pi) 
    c = np.cos(theta); s = np.sin(theta)
    v = np.zeros(8, dtype=complex)
    v[0] = c; v[7] = s
    rho_gghz = np.outer(v, v.conj())
    
    p = np.random.uniform(0.85, 0.98) 
    rho_mixed = p * rho_gghz + (1-p) * np.eye(8)/8.0
    return rho_mixed

def random_classical_ccc():
    """Generates a fully classical 3-qubit state (diagonal density matrix)."""
    probs = np.random.dirichlet(np.ones(8))
    return np.diag(probs).astype(complex)

def random_classical_cqq():
    probs = np.random.dirichlet(np.ones(2))
    rho_bc_0 = random_density_d(4)
    term0 = tensor(np.array([[1,0],[0,0]]), rho_bc_0)
    rho_bc_1 = random_density_d(4)
    term1 = tensor(np.array([[0,0],[0,1]]), rho_bc_1)
    return probs[0]*term0 + probs[1]*term1

def ghz_state():
    """Generates the 3-qubit GHZ state: (|000> + |111>)/sqrt(2)"""
    v = np.zeros(8, dtype=complex)
    v[0] = 1.0 / np.sqrt(2)
    v[7] = 1.0 / np.sqrt(2)
    return np.outer(v, v.conj())

def w_state():
    """Generates the 3-qubit W state: (|001> + |010> + |100>)/sqrt(3)"""
    v = np.zeros(8, dtype=complex)
    v[1] = 1.0 / np.sqrt(3)
    v[2] = 1.0 / np.sqrt(3)
    v[4] = 1.0 / np.sqrt(3)
    return np.outer(v, v.conj())

def bell_tensor_pure_state():
    """Generates a pure biseparable state: |Phi+> (x) |psi>"""
    v_bell = np.zeros(4, dtype=complex)
    v_bell[0] = 1.0 / np.sqrt(2)
    v_bell[3] = 1.0 / np.sqrt(2)
    v_pure = random_ket_d(2)
    v = tensor(v_bell, v_pure)
    return np.outer(v, v.conj())

# =============================
#  4. Simulation
# =============================
def qubit_amplitude_damping(p):
    K0 = np.array([[1, 0], [0, np.sqrt(1-p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return [K0, K1]

def qubit_depolarizing(p):
    p = np.clip(p, 0.0, 1.0)
    k0_val = 1.0 - 0.75 * p
    k_err_val = 0.25 * p
    K0 = np.sqrt(k0_val) * I2
    f = np.sqrt(k_err_val)
    KX = f * np.array([[0, 1], [1, 0]], dtype=complex)
    KY = f * np.array([[0, -1j], [1j, 0]], dtype=complex)
    KZ = f * np.array([[1, 0], [0, -1]], dtype=complex)
    return [K0, KX, KY, KZ]

def qubit_phase_flip(p):
    p = np.clip(p, 0.0, 1.0)
    q = 0.5 * p
    K0 = np.sqrt(1-q) * I2
    K1 = np.sqrt(q) * np.array([[1, 0], [0, -1]], dtype=complex)
    return [K0, K1]

def correlated_amplitude_damping_3qubit_kraus(p):
    p = np.clip(p, 0.0, 1.0)
    K0 = np.eye(8, dtype=complex)
    K0[7, 7] = np.sqrt(1 - p)
    K1 = np.zeros((8, 8), dtype=complex)
    K1[0, 7] = np.sqrt(p) 
    return [K0, K1]

def apply_global_kraus(rho, kraus_ops):
    new_rho = np.zeros_like(rho, dtype=complex)
    for K in kraus_ops:
        new_rho += K @ rho @ dagger(K)
    return new_rho

def apply_3qubit_kraus(rho, kraus_ops_single, targets='A'):
    target_indices = []
    if 'A' in targets: target_indices.append(0)
    if 'B' in targets: target_indices.append(1)
    if 'C' in targets: target_indices.append(2)
    
    current_ops = [[I2, I2, I2]]
    for q_idx in range(3):
        if q_idx in target_indices:
            new_ops_list = []
            for existing_op_set in current_ops:
                for k_op in kraus_ops_single:
                    temp = list(existing_op_set)
                    temp[q_idx] = k_op
                    new_ops_list.append(temp)
            current_ops = new_ops_list
            
    new_rho = np.zeros_like(rho, dtype=complex)
    for op_set in current_ops:
        K = tensor(op_set[0], op_set[1], op_set[2])
        new_rho += K @ rho @ dagger(K)
    return new_rho

@dataclass
class BudgetPoint:
    X: float
    Y: float
    BL: float 
    BNL: float
    p: float
    label: str

def generate_trajectory(rho0, channel_name, target, p_values):
    traj = []
    for p in p_values:
        if channel_name == 'phase_flip':          
            Ks = qubit_phase_flip(p)
            rho_t = apply_3qubit_kraus(rho0, Ks, targets=target)
        elif channel_name == 'depolarizing':      
            Ks = qubit_depolarizing(p)
            rho_t = apply_3qubit_kraus(rho0, Ks, targets=target)
        elif channel_name == 'amplitude_damping': 
            Ks = qubit_amplitude_damping(p)
            rho_t = apply_3qubit_kraus(rho0, Ks, targets=target)
        elif channel_name == 'correlated_amplitude_damping':
            Ks = correlated_amplitude_damping_3qubit_kraus(p)
            rho_t = apply_global_kraus(rho0, Ks)
        else: 
            raise ValueError(f"Unknown channel: {channel_name}")
        
        X, Y, BL, BNL = rho_to_coords_3qubit(rho_t)
        traj.append(BudgetPoint(X, Y, BL, BNL, p, f"{channel_name}_{target}"))
    return traj

# =============================
#  5. Dynamic Style Generation
# =============================
custom_colors = {}
legend_name_map = {}
styles_exact = {}
end_marker_exact = {}

channels = ['phase_flip', 'depolarizing', 'amplitude_damping', 'correlated_amplitude_damping']
targets = ['A', 'B', 'C', 'AB', 'BC', 'AC', 'ABC', 'corr'] 

col_map = {
    'phase_flip': '#FF9100', 
    'depolarizing': '#FF3C00', 
    'amplitude_damping': '#E61089',
    'correlated_amplitude_damping': '#0257E0'
}

name_map = {
    'phase_flip': 'Dephasing', 
    'depolarizing': 'Depolarising', 
    'amplitude_damping': 'Amplitude damping',
    'correlated_amplitude_damping': 'Correlated amplitude damping'
}

markers = {
    'phase_flip': {'A': '^', 'B': 'v', 'C': '<', 'AB': '>', 'BC': 'p', 'AC': 'h', 'ABC': 'X', 'corr': 'o'},
    'depolarizing': {'A': '^', 'B': 'v', 'C': '<', 'AB': '>', 'BC': 'p', 'AC': 'h', 'ABC': 'X', 'corr': 'o'},
    'amplitude_damping': {'A': '^', 'B': 'v', 'C': '<', 'AB': '>', 'BC': 'p', 'AC': 'h', 'ABC': '*', 'corr': 'o'},
    'correlated_amplitude_damping': {'corr': 'h', 'ABC': 'h'}
}

for ch in channels:
    for t in targets:
        key = f"{ch}_{t}"
        custom_colors[key] = col_map[ch]
        display_t = t.replace('A', 'S1').replace('B', 'S2').replace('C', 'S3')

        if t == 'corr':
            legend_name_map[key] = f"{name_map[ch]}"
            styles_exact[key] = {"dashes": [1, 0]} 
        elif len(t) == 1:
            legend_name_map[key] = f"{name_map[ch]} one-sided"# ({display_t})"
            styles_exact[key] = {"dashes": [2, 1]} 
        elif len(t) == 2:
            legend_name_map[key] = f"{name_map[ch]} two-sided"# ({display_t})"
            styles_exact[key] = {"dashes": [4, 2]} 
        else:
            legend_name_map[key] = f"{name_map[ch]} "
            styles_exact[key] = {"dashes": [1, 0]} 

        end_marker_exact[key] = markers[ch].get(t, 'o')

# =============================
#  6. Shared Trajectory Plotter
# =============================
def _plot_trajectories_loop(ax, trajectories_by_state, mode="XY"):
    """
    Shared logic to plot the customized lines and markers, ensuring visual 
    consistency across both the Geometric and Budget planes.
    """
    start_colors = {
        "Product": "#FF0000",      
        "Pure Bisep": "#99CAF0",    
        "Mixed Sep": "#FFEE99",     
        "Mixed Bisep": "#96E5F0",   
        "GME": "#5AA9E6",           
        "CQQ": "#508050",             
        "CCC": "#BBBBBB",
        "GHZ": "#1D8AAC",       
        "W State": "#00BCD4",   
        "BellxPure": "#6CBCC7",
        "PureMixed": "#9E7F67" 
    }
    
    legend_order = OrderedDict()

    for state_label, traj_dict in trajectories_by_state.items():
        s_color = "#CCCCCC" # Fallback
        for k, v in start_colors.items():
            if k in state_label: s_color = v
            
        for key, traj in traj_dict.items():
            if not traj: continue
            
            color = custom_colors.get(key, "gray")
            style = styles_exact.get(key, {"dashes": [1,0]})
            end_marker = end_marker_exact.get(key, "o")
            
            if mode == "XY":
                xs = [p.X for p in traj]
                ys = [p.Y for p in traj]
            elif mode == "Budget":
                xs = [p.BL for p in traj]
                ys = [p.BNL for p in traj]
            
            # Start Point
            ax.scatter(xs[0], ys[0], s=60, marker='o', facecolor=s_color, edgecolor='black', linewidths=0.9, zorder=200, clip_on=False)
            
            # Line
            line, = ax.plot(xs, ys, color=color, linewidth=2, alpha=0.6, zorder=80)
            line.set_dashes(style["dashes"])
            

            current_zorder = 300  # Default zorder for markers
            if key == "amplitude_damping_ABC":
                current_zorder = 500  # Higher than other markers
            # End Point
            ax.scatter(xs[-1], ys[-1], s=80, marker=end_marker, facecolors=color, edgecolors='black', linewidths=0.9, zorder=current_zorder, clip_on=False)
            
            

            # Legend Proxy
            lbl = legend_name_map.get(key, key)
            if lbl not in legend_order:
                from matplotlib.lines import Line2D
                proxy = Line2D([0], [0], color=color, linewidth=1.5, marker=end_marker, markersize=9, markeredgecolor='black')
                proxy.set_dashes(style["dashes"])
                legend_order[lbl] = proxy

    return legend_order

# =============================
#  7. Plotting - GEOMETRIC PLANE
# =============================
def plot_geometric_plane(trajectories_by_state, save_name="3Qubit_GeometricPlane.pdf"):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    X_plot = np.linspace(0, 1.05, 5000)
    X_bell = np.sqrt(1/7); X_pp = np.sqrt(3/7); X_cliff = np.sqrt(4/7); R_ball = np.sqrt(1/7)
    Y_purity_arc = np.sqrt(np.maximum(0, 1 - X_plot**2))
    
    def get_physical_boundary(x_vals):
        y_vals = np.zeros_like(x_vals)
        mask_1 = x_vals <= X_pp
        y_vals[mask_1] = np.sqrt(np.maximum(0, 1 - x_vals[mask_1]**2))
        mask_2 = (x_vals > X_pp) & (x_vals <= X_cliff)
        y_vals[mask_2] = np.sqrt(np.maximum(0, 10/7 - 2*x_vals[mask_2]**2))
        return y_vals

    Y_physical = get_physical_boundary(X_plot)
    Y_cqq_val = np.sqrt(6/7)
    Y_ccq = np.sqrt(np.maximum(0, (18 - 14*X_plot**2) / 21))
    Y_ball = np.sqrt(np.maximum(0, R_ball**2 - X_plot**2))

    mask_phys = X_plot <= X_cliff
    ax.fill_between(X_plot, Y_physical, Y_purity_arc, color="lightgray", zorder=3) 
    ax.fill_between(X_plot, Y_cqq_val, Y_purity_arc, where=(X_plot <= X_bell), color="#7FC8F8", alpha=0.4, zorder=2) 
    
    Y_top_bisep = np.minimum(Y_physical, Y_cqq_val)
    Y_bot_bisep = np.minimum(Y_physical, Y_ccq)
    ax.fill_between(X_plot, Y_bot_bisep, Y_top_bisep, where=mask_phys, color="#5AA9E6", alpha=0.45, zorder=2) 
    ax.fill_between(X_plot, Y_ball, Y_bot_bisep, where=mask_phys, color="#FFEBEB", alpha=1, zorder=3) 
    ax.fill_between(X_plot, 0, Y_ball, where=mask_phys, color="#FFEE99", alpha=1, zorder=8) 

    ax.plot(X_plot, Y_purity_arc, color="#49616E", linestyle="--", linewidth=2.5, zorder=5)
    ax.plot(X_plot[X_plot <= X_pp], Y_physical[X_plot <= X_pp], color="#274C77", linewidth=2.5, zorder=11)
    ax.plot(X_plot[(X_plot >= X_pp) & (X_plot <= X_cliff)], Y_physical[(X_plot >= X_pp) & (X_plot <= X_cliff)], color="black", linewidth=2.5, zorder=10)
    ax.plot([X_cliff, X_cliff], [0, np.sqrt(2/7)], color="black", linewidth=2.5, zorder=10)
    
    ax.plot(X_plot[X_plot <= X_bell], [Y_cqq_val]*sum(X_plot <= X_bell), linestyle="-.", color="#274C77", linewidth=2, zorder=9, label="CQQ Limit")
    ax.plot(X_plot[X_plot <= X_pp], Y_ccq[X_plot <= X_pp], color="#975AE6", linewidth=2.0, zorder=9, label="CCQ Limit")
    ax.plot(X_plot[X_plot <= R_ball], Y_ball[X_plot <= R_ball], color="#D62728", linestyle="--", linewidth=2.0, zorder=15, label="Abs. Separable")

    # Draw trajectories
    legend_order = _plot_trajectories_loop(ax, trajectories_by_state, mode="XY")

    handles = list(legend_order.values())
    labels = list(legend_order.keys())
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, handlelength=2.5)
    
    ax.set_xlabel("$X$", fontsize=20)
    ax.set_ylabel("$Y$", fontsize=20)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.25)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_zorder(50)
        
    fig.tight_layout()
    fig.savefig(save_name, dpi=300)
    plt.show()
    plt.close(fig)

# =============================
#  8. Plotting - BUDGET PLANE
# =============================
def plot_budget_plane(trajectories_by_state, save_name="3Qubit_BudgetPlane.pdf"):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Boundary mathematical definitions
    BL_vals = np.linspace(0, 3.2, 500)
    valid_mask = BL_vals <= 3
    BNL_roof = 7 - BL_vals
    
    def get_budget_wall(b_l):
        res = np.zeros_like(b_l)
        mask1 = (b_l >= 0) & (b_l < 1)
        res[mask1] = 0
        mask2 = (b_l >= 1) & (b_l < 2)
        res[mask2] = b_l[mask2] - 1
        mask3 = (b_l >= 2) & (b_l <= 3)
        res[mask3] = 3*b_l[mask3] - 5
        res[b_l > 3] = np.nan
        return res
        
    BNL_wall = get_budget_wall(BL_vals)
    BNL_cqq = 3 * BL_vals + 3
    BNL_ccq = BL_vals / 3.0 + 3.0
    val_1_7 = 1.0/7.0
    BNL_ball_line = val_1_7 - BL_vals
    
    # --- FILLS ---
    ax.fill_between(BL_vals, 0, BNL_wall, where=valid_mask, color="lightgray", zorder=3)
    
    top_gme = np.minimum(BNL_roof, 8) 
    bot_gme = BNL_cqq
    ax.fill_between(BL_vals, bot_gme, top_gme, where=(BL_vals <= 1), color="#7FC8F8", alpha=0.4, zorder=2)
    
    top_bisep = np.minimum(BNL_roof, BNL_cqq)
    bot_bisep = np.maximum(BNL_wall, BNL_ccq)
    ax.fill_between(BL_vals, bot_bisep, top_bisep, where=valid_mask, color="#5AA9E6", alpha=0.45, zorder=2)
    
    top_sep_outer = BNL_ccq
    bot_sep_outer = np.maximum(BNL_wall, BNL_ball_line)
    ax.fill_between(BL_vals, bot_sep_outer, top_sep_outer, where=valid_mask, color="#FFEBEB", alpha=1, zorder=3)
    
    mask_ball = (BL_vals >= 0) & (BL_vals <= val_1_7)
    ax.fill_between(BL_vals, 0, BNL_ball_line, where=mask_ball, color="#FFEE99", alpha=1, zorder=8)
    
    # --- LINES ---
    ax.plot(BL_vals[valid_mask], BNL_roof[valid_mask], color="#49616E", linestyle="-", lw=2.5, zorder=5)
    ax.plot(BL_vals[valid_mask], BNL_wall[valid_mask], color="black", lw=2.5, zorder=10)
    ax.plot(BL_vals[BL_vals<=1], BNL_cqq[BL_vals<=1], color="#274C77", linestyle="-.", lw=2.0, zorder=9, label="CQQ Limit")
    ax.plot(BL_vals[valid_mask], BNL_ccq[valid_mask], color="#975AE6", lw=2.0, zorder=9, label="CCQ Limit")
    ax.plot(BL_vals[mask_ball], BNL_ball_line[mask_ball], color="#D62728", linestyle="--", lw=2.0, zorder=15, label="Abs. Sep")

    # Draw trajectories
    legend_order = _plot_trajectories_loop(ax, trajectories_by_state, mode="Budget")

    handles = list(legend_order.values())
    labels = list(legend_order.keys())
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=7, handlelength=2.5)

    ax.set_xlabel(r"$B_\mathrm{L}$", fontsize=20)
    ax.set_ylabel(r"$B_\mathrm{NL}$", fontsize=20)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 7)
    ax.grid(alpha=0.25)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_zorder(50)
    
    fig.tight_layout()
    fig.savefig(save_name, dpi=300)
    plt.show()
    plt.close(fig)

# =============================
#  9. Main Execution
# =============================
def main():
    np.random.seed(19) #6 for GME, 7 for Pure Bisep, 120 for Mixed Bisep, 30 for PureProd, 28 for Mixed Sep, 19 for Classical
    p_values = np.linspace(0.0, 1.0, 50)
    
    states = {
        #    "Product": random_pure_product(),
        #    "Pure Bisep": random_pure_biseparable(),
        #    "Mixed Bisep": random_mixed_biseparable(p=0.82),
        #    "Mixed Sep": random_mixed_separable(),
        #    "GME": random_mixed_gme_nonzero_bloch(),
           "CCC": random_classical_ccc(),
        #    "CQQ": random_classical_cqq(),
        #    "GHZ": ghz_state(),
        #    "W State": w_state(),
        #    "BellxPure": bell_tensor_pure_state(),
        #   "PureMixed": random_pure_pure_mixed(),
    }
    
    sim_configs = [
        ("depolarizing", ["ABC","AB", "A"]), 
        ("amplitude_damping", ["ABC","AB" ,"A"]),
        # ("correlated_amplitude_damping", ["corr"])
        #  ("depolarizing", ["ABC","AB","A"]),
        #  ("amplitude_damping", ["ABC","AB","A"]),
         ("correlated_amplitude_damping", ["corr"])
    ]
    
    trajectories = {}
    for name, rho in states.items():
        trajectories[name] = {}
        for ch, targets in sim_configs:
            for t in targets:
                traj = generate_trajectory(rho, ch, t, p_values)
                trajectories[name][f"{ch}_{t}"] = traj
                
    print("Generating Geometric Plane PDF...")
    plot_geometric_plane(trajectories)
    print("Generating Budget Plane PDF...")
    plot_budget_plane(trajectories)
    print("Done!")
   

if __name__ == "__main__":
    main()