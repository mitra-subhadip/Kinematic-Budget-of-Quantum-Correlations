# %%
#!/usr/bin/env python3
"""
Decoherence trajectories in the P-Q geometry.
Updates:
  - Removed X-Y and BL-BNL plotters and logic as requested.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from dataclasses import dataclass
from collections import OrderedDict

# %%
# =============================
#  Font configuration
# =============================
try:
    font_path = "./times.ttf"
    fm.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.8,
    })
except:
    pass

# %%
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

# %%
# ---------------------------------------------------------------------
# Random state generators & States
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

def random_classical_CC_state():
    p = np.random.dirichlet([1, 1, 1, 1])
    return np.diag(p.astype(complex))

def partial_transpose(rho, sys='A'):
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

def bell_state_phi_plus():
    v = np.zeros(4, dtype=complex); v[0] = 1/np.sqrt(2); v[3] = 1/np.sqrt(2)
    return np.outer(v, np.conjugate(v))

def chsh_strong_werner_state(p=0.9):
    rho_bell = bell_state_phi_plus()
    rho_mix = np.eye(4, dtype=complex)/4.0
    return p * rho_bell + (1-p) * rho_mix

def random_mixed_entangled_state(max_tries=5000):
    for _ in range(max_tries):
        rho = random_mixed_state_2qubit()
        if negativity(rho) > .1:  
            return rho
    raise RuntimeError("Could not find entangled mixed state")

def random_mixed_separable_state(max_tries=10000):
    for _ in range(max_tries):
        rho = random_mixed_state_2qubit()
        neg = negativity(rho)
        P = np.real_if_close(np.trace(rho @ rho))
        if neg < 1e-8 and P < 0.999:  # PPT + not essentially pure
            return rho
    raise RuntimeError("Could not find mixed separable state")

# %%
# ---------------------------------------------------------------------
# Fano decomposition & Budget Calculation
# ---------------------------------------------------------------------

def rho_to_fano(rho):
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
    BL: float
    BNL: float
    P: float
    Q: float
    p_param: float
    label: str

def compute_budget_from_r_s_t(r, s, t, p_value, label=""):
    r = np.asarray(r, dtype=float).reshape(3)
    s = np.asarray(s, dtype=float).reshape(3)
    t = np.asarray(t, dtype=float).reshape(3, 3)

    BL  = float(np.dot(r, r) + np.dot(s, s))
    BNL = float(np.sum(t**2))
    
    # P, Q calculation
    P = (1.0 + BL + BNL) / 4.0
    Q = (1.0 - BL + BNL) / 4.0

    # X, Y calculation
    eps = 1e-12
    P_safe = max(P, eps)
    X_sq = BL / (3.0 * P_safe)
    Y_sq = BNL / (3.0 * P_safe)
    
    X = float(np.sqrt(max(0, X_sq)))
    Y = float(np.sqrt(max(0, Y_sq)))

    return BudgetPoint(X=X, Y=Y, BL=BL, BNL=BNL, P=P, Q=Q, p_param=p_value, label=label)

# %%
# ---------------------------------------------------------------------
# Noise channels (Kraus form)
# ---------------------------------------------------------------------

def dephasing_channel_kraus(p):
    p = float(np.clip(p, 0.0, 1.0))
    q = 0.5 * p
    K0 = np.sqrt(1 - q) * identity_2
    K1 = np.sqrt(q)     * sigma_z
    return [K0, K1]

def depolarizing_channel_kraus(p):
    p = float(np.clip(p, 0.0, 1.0))
    p_eff = 0.75 * p
    K0 = np.sqrt(1 - p_eff) * identity_2
    Kx = np.sqrt(p_eff / 3.0) * sigma_x
    Ky = np.sqrt(p_eff / 3.0) * sigma_y
    Kz = np.sqrt(p_eff / 3.0) * sigma_z
    return [K0, Kx, Ky, Kz]

def amplitude_damping_channel_kraus(p):
    K0 = np.array([[1, 0],[0, np.sqrt(1-p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)],[0, 0]], dtype=complex)
    return [K0, K1]

def apply_single_qubit_kraus_channel(rho, kraus_ops, which='A'):
    rho = np.asarray(rho, dtype=complex).reshape(4, 4)
    new_rho = np.zeros_like(rho, dtype=complex)
    if which in ('A', 'B'):
        for K in kraus_ops:
            K_full = tensor(K, identity_2) if which == 'A' else tensor(identity_2, K)
            new_rho += K_full @ rho @ dagger(K_full)
    elif which == 'AB':
        for K_a in kraus_ops:
            for K_b in kraus_ops:
                K_full = tensor(K_a, K_b)
                new_rho += K_full @ rho @ dagger(K_full)
    return new_rho

def apply_two_qubit_kraus_channel(rho, kraus_ops):
    rho = np.asarray(rho, dtype=complex).reshape(4, 4)
    new_rho = np.zeros_like(rho, dtype=complex)
    for K in kraus_ops:
        new_rho += K @ rho @ dagger(K)
    return new_rho

def correlated_phase_flip_channel_kraus(p):
    I4 = np.eye(4, dtype=complex)
    ZZ = tensor(sigma_z, sigma_z)
    K0 = np.sqrt(1 - p) * I4; K1 = np.sqrt(p) * ZZ
    return [K0, K1]

def correlated_amplitude_damping_channel_kraus(p):
    K0 = np.diag([1, 1, 1, np.sqrt(1-p)]).astype(complex)
    K1 = np.zeros((4, 4), dtype=complex); K1[0, 3] = np.sqrt(p)
    return [K0, K1]

# %%
# ---------------------------------------------------------------------
# Trajectory simulation
# ---------------------------------------------------------------------

def generate_trajectory(rho0, channel_name, which, p_values):
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

# %%
# ---------------------------------------------------------------------
# Plotting Functions for Transformed Geometries
# ---------------------------------------------------------------------

def transform_XY_to_Target(X_arr, Y_arr, mode="PQ"):
    X = np.asarray(X_arr)
    Y = np.asarray(Y_arr)
    
    R2 = X**2 + Y**2
    P = 1.0 / (4.0 - 3.0 * R2)
    
    BL_vals = 3.0 * P * X**2
    BNL_vals = 3.0 * P * Y**2
    
    if mode == "PQ":
        Q_vals = (1.0 - BL_vals + BNL_vals) / 4.0
        return P, Q_vals
    else:
        raise ValueError("Unknown mode")

def get_boundary_polygons(mode="PQ"):
    x_dense = np.linspace(0, 1, 5000)
    y_R1 = np.sqrt(np.maximum(0, 1 - x_dense**2))
    y_PPT = np.sqrt(np.maximum(0, 1.0/3.0 - x_dense**2))
    y_chsh_raw = np.sqrt(np.maximum(0, (4 - 3*x_dense**2)/5))
    y_chsh = np.minimum(y_chsh_raw, y_R1)
    
    x_max_cc = np.sqrt(2/3)
    mask_cc = x_dense <= x_max_cc
    y_cc = np.zeros_like(x_dense)
    y_cc[mask_cc] = np.sqrt(np.maximum(0, 2/3 - 0.5 * x_dense[mask_cc]**2))
    
    def make_poly(x_curve, y_curve):
        x_top = x_curve
        y_top = y_curve
        x_bot = x_curve[::-1]
        y_bot = np.zeros_like(x_bot)
        x_poly = np.concatenate([x_top, x_bot])
        y_poly = np.concatenate([y_top, y_bot])
        u, v = transform_XY_to_Target(x_poly, y_poly, mode)
        return u, v

    poly_R1 = make_poly(x_dense, y_R1)
    poly_CHSH = make_poly(x_dense, y_chsh)
    y_cc_eff = y_cc.copy()
    y_cc_eff[~mask_cc] = 0
    poly_CC = make_poly(x_dense, y_cc_eff)
    poly_PPT = make_poly(x_dense, y_PPT)
    
    line_CHSH = transform_XY_to_Target(x_dense, y_chsh, mode)
    line_CC = transform_XY_to_Target(x_dense[mask_cc], y_cc[mask_cc], mode)
    line_PPT = transform_XY_to_Target(x_dense, y_PPT, mode)
    line_Pure = transform_XY_to_Target(x_dense, y_R1, mode)

    return {
        "poly_R1": poly_R1, "poly_CHSH": poly_CHSH, "poly_CC": poly_CC, "poly_PPT": poly_PPT,
        "line_CHSH": line_CHSH, "line_CC": line_CC, "line_PPT": line_PPT, "line_Pure": line_Pure
    }

def plot_transformed_geometry(trajectories_by_state, mode="PQ", 
                              start_color_map=None, show=True, save_name="plot.pdf"):
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Draw Background Regions (zorder 1-4)
    geo = get_boundary_polygons(mode)
    col_R1 = "#5AA9E6"; col_CHSH_below = "#7FC8F8"; col_CC = "#FFEBEB"; ppt_color = "#FF6392"
    
    ax.fill(geo["poly_R1"][0], geo["poly_R1"][1], color=col_R1, alpha=0.4, zorder=1)
    ax.fill(geo["poly_CHSH"][0], geo["poly_CHSH"][1], color=col_CHSH_below, alpha=0.45, zorder=2)
    ax.fill(geo["poly_CC"][0], geo["poly_CC"][1], color=col_CC, alpha=1, zorder=3)
    ax.fill(geo["poly_PPT"][0], geo["poly_PPT"][1], color="#FFEE99", alpha=0.65, zorder=4)

    # Draw Boundary Lines (zorder 5+)
    ax.plot(geo["line_CHSH"][0], geo["line_CHSH"][1], "-.", color="#274C77", linewidth=2, zorder=6)
    ax.plot(geo["line_CC"][0], geo["line_CC"][1], color="#975AE6", linewidth=2.5, zorder=7)
    ax.plot(geo["line_PPT"][0], geo["line_PPT"][1], "--", color=ppt_color, linewidth=2.5, zorder=9)
    ax.plot(geo["line_Pure"][0], geo["line_Pure"][1], color="#274C77", linewidth=2.5, zorder=5)

    # Trajectories Configuration
    custom_colors = {
        "dephasing_A":  "#FF9100", "dephasing_B":  "#FF9100", "dephasing_AB": "#FF9100",
        "depolarizing_A":  "#FF3C00", "depolarizing_B":  "#FF3C00", "depolarizing_AB": "#FF3C00",
        "amplitude_damping_A":  "#E61089", "amplitude_damping_B":  "#E61089", "amplitude_damping_AB": "#E61089",
        "correlated_phase_flip_corr": "#03AD3F", "correlated_amplitude_damping_corr": "#0257E0",
    }
    legend_name_map = {
        "dephasing_B": "Dephasing (one-sided)", "dephasing_AB": "Dephasing",
        "depolarizing_A": "Depolarising (one-sided)", "depolarizing_AB": "Depolarising",
        "amplitude_damping_A": "Amplitude damping (one-sided)", "amplitude_damping_AB": "Amplitude damping",
        "correlated_phase_flip_corr": "Correlated phase flip",
        "correlated_amplitude_damping_corr": "Correlated amplitude damping",
    }
    styles_exact = {
        "dephasing_A":  {"dashes": [2, 1]}, "dephasing_B":  {"dashes": [2, 1]}, "dephasing_AB": {"dashes": [1, 0]},
        "depolarizing_A":  {"dashes": [2, 1]}, "depolarizing_AB": {"dashes": [1, 0]},
        "amplitude_damping_A":  {"dashes": [2, 1]}, "amplitude_damping_AB": {"dashes": [1, 0]},
        "correlated_phase_flip_corr": {"dashes": [1, 0]}, "correlated_amplitude_damping_corr": {"dashes": [1, 0]},
    }
    end_marker_exact = {
        "dephasing_A": "D", "dephasing_B": "s", "dephasing_AB": "P",
        "depolarizing_A": "^", "depolarizing_AB": "X",
        "amplitude_damping_A": ">", "amplitude_damping_AB": "*",
        "correlated_phase_flip_corr": "o", "correlated_amplitude_damping_corr": "h",
    }
    default_category_colors = {
        "bell": "#99CAF0", "pure_product": "#FF0000", "mixed_entangled": "#5AA9E6",
        "mixed_separable": "#FFEE99", "class": "#BBBBBB", "CHSH": "#99CAF0"
    }
    def infer_state_category(state_label):
        s = str(state_label).lower()
        if ("bell" in s) or ("phi" in s) or ("psi" in s): return "bell"
        if ("pure" in s) or ("product" in s): return "pure_product"
        if ("mixed" in s) and ("ent" in s): return "mixed_entangled"
        if ("classical" in s):  return "class"
        if ("mixed" in s) and ("sep" in s): return "mixed_separable"
        if ("chsh" in s) or ("werner" in s): return "CHSH"

    if start_color_map is None: start_color_map = {}
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_index = 0
    legend_order = OrderedDict()
    
    for state_label, traj_dict in trajectories_by_state.items():
        if not isinstance(traj_dict, dict): continue
        cat = infer_state_category(state_label)
        start_color = start_color_map.get(state_label, default_category_colors.get(cat, "#CCCCCC"))
        for traj_label, traj in traj_dict.items():
            if not traj: continue
            
            # Select coordinates for PQ mode
            if mode == "PQ":
                u = np.array([bp.P for bp in traj])
                v = np.array([bp.Q for bp in traj])
                
            p_vals = np.array([bp.p_param for bp in traj])
            mask = np.isfinite(u) & np.isfinite(v)
            u = u[mask]; v = v[mask]; p_vals = p_vals[mask]
            if len(u) == 0: continue
            traj_key = traj_label
            color = custom_colors.get(traj_key, colors[color_index % len(colors)])
            dashes = styles_exact.get(traj_key, {"dashes": [1,0]})["dashes"]
            end_marker = end_marker_exact.get(traj_key, "X")
            if traj_key not in custom_colors: color_index += 1
            ax.plot(u, v, linewidth=2, alpha=0.6, color=color, dashes=dashes, zorder=80)
            
            # START POINT: Add clip_on=False
            ax.scatter(u[0], v[0], s=60, marker='o', facecolor=start_color, 
                       edgecolor="black", linewidths=0.9, zorder=200, clip_on=False)
            
            # END POINT: Add clip_on=False
            ax.scatter(u[-1], v[-1], s=80, marker=end_marker, facecolors=color, 
                       edgecolors='black', linewidths=0.9, zorder=300, clip_on=False)
                       
            legend_label = legend_name_map.get(traj_label, traj_label)
            if legend_label not in legend_order:
                from matplotlib.lines import Line2D
                proxy = Line2D([0], [0], color=color, linewidth=0.01, marker=end_marker, markersize=9, markeredgecolor='black')
                try: proxy.set_dashes(dashes)
                except: pass
                legend_order[legend_label] = proxy

    # Axis Labels and Limits for PQ Mode
    if mode == "PQ":
        ax.set_xlabel(r"$P$", fontsize=20)
        ax.set_ylabel(r"$Q$", fontsize=20)
        ax.set_xlim(0.25, 1.0)
        ax.set_ylim(0.0, 1.0)

    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.25)
    
    if len(legend_order) > 0:
        handles = list(legend_order.values())
        labels = list(legend_order.keys())
        ax.legend(handles=handles, labels=labels, loc="best", fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

# %%
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    np.random.seed(42) #420 for mixed entangled, 124 for Pure Product, 42 for Mixed Sep and Classical
    p_values = np.linspace(0.0, 1.0, 50)
    rho_classical_cc = random_classical_CC_state()
    rho_chsh_guarantee = chsh_strong_werner_state(p=0.7)
    bell_state_phi_plus1 = bell_state_phi_plus()
    mixed_sep = random_mixed_separable_state()
    pure_state = random_product_pure_state()

    initial_states = {
        "classical": rho_classical_cc,
        "Bell": bell_state_phi_plus1, 
        "MixedSep": mixed_sep,
        "Pure": pure_state,
        "CHSH": rho_chsh_guarantee,
        "mixed_ent": random_mixed_entangled_state(),
    }
    local_channel_targets = {
          'depolarizing': ['AB','A'],
           'amplitude_damping': [ 'AB','A'],
        #    "dephasing": ['AB'],
    }
    correlated_channels = ['correlated_amplitude_damping']
    trajectories_by_state = {}

    for s_label, rho0 in initial_states.items():
        trajectories_by_state[s_label] = {}
        for ch, targets in local_channel_targets.items():
            for t in targets:
                traj = generate_trajectory(rho0, ch, which=t, p_values=p_values)
                key = f"{ch}_{t}"
                trajectories_by_state[s_label][key] = traj
        for ch_corr in correlated_channels:
            traj = generate_trajectory(rho0, ch_corr, which='corr', p_values=p_values)
            key  = f"{ch_corr}_corr"
            trajectories_by_state[s_label][key] = traj

    print("Plotting P - Q geometry...")
    plot_transformed_geometry(trajectories_by_state, mode="PQ", save_name="Fig_5F.pdf")
    

if __name__ == "__main__":
    main()