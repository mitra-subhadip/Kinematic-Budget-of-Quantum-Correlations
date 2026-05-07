import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections import defaultdict, OrderedDict
import matplotlib.font_manager as fm

font_path = "./times.ttf"
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.8,
    })
except:
    pass


# ==========================================
# 1. THE COORDINATE ENGINE (UPDATED)
# ==========================================

def get_pauli_matrices():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I, X, Y, Z]

def get_coordinates(rho):
    """Maps a 4x4 density matrix to all geometry variables."""
    paulis = get_pauli_matrices()
    
    P = np.real(np.trace(rho @ rho))
    if P > 1.0: P = 1.0 
    
    r = np.zeros(3); s = np.zeros(3); t = np.zeros((3, 3))
    
    for i in range(3):
        sigma_i = paulis[i+1]
        op_r = np.kron(sigma_i, paulis[0])
        r[i] = np.real(np.trace(rho @ op_r))
        op_s = np.kron(paulis[0], sigma_i)
        s[i] = np.real(np.trace(rho @ op_s))
        for j in range(3):
            sigma_j = paulis[j+1]
            op_t = np.kron(sigma_i, sigma_j)
            t[i, j] = np.real(np.trace(rho @ op_t))

    B_L = np.sum(r**2) + np.sum(s**2)
    B_NL = np.sum(t**2)
    Q = (1.0 - B_L + B_NL) / 4.0
    
    if P < 1e-9: 
        return 0.0, 0.0, B_L, B_NL, P, Q
        
    X = np.sqrt(max(0, B_L / (3 * P)))
    Y = np.sqrt(max(0, B_NL / (3 * P)))
    
    # Return full tuple to support XY, BL-BNL, and PQ modes natively
    return X, Y, B_L, B_NL, P, Q

# ==========================================
# 2. STATE GENERATION (UNCHANGED)
# ==========================================

def tensor(a, b): return np.kron(a, b)
def dagger(M): return np.conjugate(M).T

def random_complex_vector(d):
    v = np.random.normal(size=(d,)) + 1j * np.random.normal(size=(d,))
    v /= np.linalg.norm(v)
    return v

def random_single_qubit_pure():
    v = random_complex_vector(2)
    return np.outer(v, np.conjugate(v))

def bell_state_phi_plus():
    v = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return np.outer(v, np.conjugate(v))

def random_product_pure_state():
    rhoA = random_single_qubit_pure()
    rhoB = random_single_qubit_pure()
    return tensor(rhoA, rhoB)

def random_mixed_state_2qubit():
    G = np.random.normal(size=(4, 4)) + 1j * np.random.normal(size=(4, 4))
    M = G @ dagger(G)
    return M / np.trace(M)

def negativity(rho):
    rho = np.asarray(rho, dtype=complex).reshape(4, 4)
    rho_pt = np.zeros_like(rho)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    rho_pt[2*c+b, 2*a+d] = rho[2*a+b, 2*c+d]
    evals = np.linalg.eigvalsh(rho_pt)
    return float(-np.sum(evals[evals < 0]))

def random_mixed_entangled_state(max_tries=5000):
    for _ in range(max_tries):
        rho = random_mixed_state_2qubit()
        if negativity(rho) > 0.1:  
            return rho
    raise RuntimeError("Could not find entangled mixed state")

def random_mixed_separable_state(max_tries=10000):
    for _ in range(max_tries):
        rho = random_mixed_state_2qubit()
        neg = negativity(rho)
        P = np.real(np.trace(rho @ rho))
        if neg < 1e-8 and P < 0.999:
            return rho
    raise RuntimeError("Could not find mixed separable state")

def chsh_strong_werner_state(p=0.9):
    rho_bell = bell_state_phi_plus()
    rho_mix = np.eye(4, dtype=complex)/4.0
    return p * rho_bell + (1-p) * rho_mix

def random_classical_CC_state():
    p = np.random.dirichlet([1, 1, 1, 1])
    return np.diag(p.astype(complex))

def maximally_mixed_state_2qubit():
    return np.eye(4, dtype=complex) / 4.0

# ==========================================
# 3. PURIFICATION PROTOCOLS (UPDATED APPENDS)
# ==========================================

def protocol_local_cooling(rho, steps=50, rate=0.08):
    trajectory = []
    current_rho = rho.copy()
    target_rho = np.outer([1,0,0,0], [1,0,0,0]) 
    
    for _ in range(steps):
        trajectory.append(get_coordinates(current_rho))
        current_rho = (1 - rate) * current_rho + rate * target_rho
        current_rho /= np.trace(current_rho)
        
    return np.array(trajectory)

def protocol_dissipative_bell_stabilization(rho, steps=50, rate=0.08):
    trajectory = []
    current_rho = rho.copy()
    target_rho = bell_state_phi_plus()
    
    for _ in range(steps):
        trajectory.append(get_coordinates(current_rho))
        current_rho = (1 - rate) * current_rho + rate * target_rho
        current_rho /= np.trace(current_rho)
        
    return np.array(trajectory)

def protocol_thermal_purification(rho, step_size=0.1, steps=50):
    trajectory = []
    current_rho = rho.copy()
    for _ in range(steps):
        trajectory.append(get_coordinates(current_rho))
        evals, evecs = np.linalg.eigh(current_rho)
        evals = np.power(np.maximum(np.real(evals), 0), 1 + step_size)
        new_rho = evecs @ np.diag(evals) @ evecs.conj().T
        norm = np.trace(new_rho)
        if norm > 1e-12: current_rho = new_rho / norm
        else: break
    return np.array(trajectory)

def protocol_local_filtering(rho, strength=0.05, steps=50):
    trajectory = []
    current_rho = rho.copy()
    filt_1q = np.array([[1, 0], [0, np.sqrt(1 - strength)]], dtype=complex)
    F = np.kron(filt_1q, filt_1q)
    for _ in range(steps):
        trajectory.append(get_coordinates(current_rho))
        unnorm_rho = F @ current_rho @ F.conj().T
        norm = np.real(np.trace(unnorm_rho))
        if norm > 1e-12: current_rho = unnorm_rho / norm
        else: break
    return np.array(trajectory)

# ==========================================
# 4. MULTI-PLANE PLOTTING LOGIC (NEW/MERGED)
# ==========================================

def transform_XY_to_Target(X_arr, Y_arr, mode="BL_BNL"):
    X = np.asarray(X_arr)
    Y = np.asarray(Y_arr)
    if mode == "XY": return X, Y

    R2 = X**2 + Y**2
    P = 1.0 / (4.0 - 3.0 * R2)
    
    BL_vals = 3.0 * P * X**2
    BNL_vals = 3.0 * P * Y**2
    
    if mode == "BL_BNL":
        return BL_vals, BNL_vals
    elif mode == "PQ":
        Q_vals = (1.0 - BL_vals + BNL_vals) / 4.0
        return P, Q_vals
    else:
        raise ValueError("Unknown mode")

def get_boundary_polygons(mode="BL_BNL"):
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
        x_poly = np.concatenate([x_curve, x_curve[::-1]])
        y_poly = np.concatenate([y_curve, np.zeros_like(x_curve)])
        return transform_XY_to_Target(x_poly, y_poly, mode)

    return {
        "poly_R1": make_poly(x_dense, y_R1),
        "poly_CHSH": make_poly(x_dense, y_chsh),
        "poly_CC": make_poly(x_dense, np.where(mask_cc, y_cc, 0)),
        "poly_PPT": make_poly(x_dense, y_PPT),
        "line_CHSH": transform_XY_to_Target(x_dense, y_chsh, mode),
        "line_CC": transform_XY_to_Target(x_dense[mask_cc], y_cc[mask_cc], mode),
        "line_PPT": transform_XY_to_Target(x_dense, y_PPT, mode),
        "line_Pure": transform_XY_to_Target(x_dense, y_R1, mode)
    }

def plot_purification_geometry(trajectories_by_state, mode="XY", show=True, save_prefix="Master_Purification_Map"):
    fig, ax = plt.subplots(figsize=(7, 7))

    # --- Background Fills & Lines ---
    geo = get_boundary_polygons(mode)
    col_R1 = "#5AA9E6"; col_CHSH_below = "#7FC8F8"; col_CC = "#FFEBEB"; ppt_color = "#FF6392"
    
    ax.fill(geo["poly_R1"][0], geo["poly_R1"][1], color=col_R1, alpha=0.4, zorder=1)
    ax.fill(geo["poly_CHSH"][0], geo["poly_CHSH"][1], color=col_CHSH_below, alpha=0.45, zorder=2)
    ax.fill(geo["poly_CC"][0], geo["poly_CC"][1], color=col_CC, alpha=1, zorder=3)
    ax.fill(geo["poly_PPT"][0], geo["poly_PPT"][1], color="#FFEE99", alpha=0.65, zorder=4)

    # Dimensional Contextual Overlays
    if mode == "XY":
        x_cutoff = np.sqrt(2/3)
        
        mask_pure = geo["line_Pure"][0] <= x_cutoff
        geo["line_Pure"] = (geo["line_Pure"][0][mask_pure], geo["line_Pure"][1][mask_pure])
        
        mask_ppt = geo["line_PPT"][0] <= x_cutoff
        geo["line_PPT"] = (geo["line_PPT"][0][mask_ppt], geo["line_PPT"][1][mask_ppt])

        x_bound = np.sqrt(2/3)
        x_shade = np.linspace(x_bound, 1.0, 500)
        y_shade = np.sqrt(np.maximum(0, 1 - x_shade**2))
        ax.fill_between(x_shade, 0, y_shade, color='lightgray', zorder=5)

        y_max_bound = np.sqrt(1 - x_bound**2)
        ax.plot([x_bound, x_bound], [0, y_max_bound], color='black', linewidth=2.5, zorder=30, label=r"$X=\sqrt{2/3}$")
        ax.plot(x_shade, y_shade, '--', color="#274C77", linewidth=2, zorder=31)

    elif mode == "BL_BNL":
        ax.axvspan(2.0, 4.0, color='white', zorder=4.5, alpha=1.0)
        
        bl_grid = np.linspace(1.0, 2.0, 200) 
        bnl_bound = np.maximum(0, bl_grid - 1.0)
        ax.fill_between(bl_grid, 0, bnl_bound, color='lightgray', alpha=1, zorder=25, label=r"$Q<0$")

        x_val = np.sqrt(2/3)
        y_vals = np.linspace(0, np.sqrt(1/3), 200)
        x_arr = np.full_like(y_vals, x_val)
        u_bound, v_bound = transform_XY_to_Target(x_arr, y_vals, mode="BL_BNL")
        ax.plot(u_bound, v_bound, color='black', linewidth=2.5, zorder=30, label=r"$X=\sqrt{2/3}$")

    ax.plot(geo["line_CHSH"][0], geo["line_CHSH"][1], "-.", color="#274C77", linewidth=2, zorder=6)
    ax.plot(geo["line_CC"][0], geo["line_CC"][1], color="#975AE6", linewidth=2.5, zorder=7)
    ax.plot(geo["line_PPT"][0], geo["line_PPT"][1], "--", color=ppt_color, linewidth=2.5, zorder=9)
    ax.plot(geo["line_Pure"][0], geo["line_Pure"][1], color="#274C77", linewidth=2.5, zorder=5)

    # --- Trajectory Colors & Dictionaries ---
    protocol_colors = {
        "Thermal": "#FF3C00", "LocalCooling": "#E61089", 
        "Filtering": "#FF9100", "BellStabilization": "#0257E0",
    }
    protocol_dashes = {
        "Thermal": [1, 0], "LocalCooling": [1, 0], 
        "Filtering": [2, 1], "BellStabilization": [1, 0],
    }
    state_colors = {
        "bell": "#99CAF0", "pure": "#FF0000", "mixedent": "#5AA9E6",
        "mixedsep": "#FFEE99", "classical": "#BBBBBB", "chsh": "#7FC8F8"
    }

    legend_entries = OrderedDict()

    for state_lbl, protocol_dict in trajectories_by_state.items():
        label_lower = state_lbl.lower()
        if "classical" in label_lower: canonical_label = "Classical state"; start_color = state_colors["classical"]
        elif "bell" in label_lower: canonical_label = "Bell state"; start_color = state_colors["bell"]
        elif "pure" in label_lower: canonical_label = "Pure product"; start_color = state_colors["pure"]
        elif "mixed_ent" in label_lower or "mixedent" in label_lower: canonical_label = "Mixed entangled"; start_color = state_colors["mixedent"]
        elif "mixed_sep" in label_lower or "mixedsep" in label_lower: canonical_label = "Mixed separable"; start_color = state_colors["mixedsep"]
        elif "chsh" in label_lower: canonical_label = "CHSH state"; start_color = state_colors["chsh"]
        else: canonical_label = state_lbl; start_color = "#CCCCCC"

        state_dot = None

        for proto_name, traj in protocol_dict.items():
            if len(traj) == 0: continue
            
            # Unpack coordinates based on requested mode plane
            # traj format is now array of (X, Y, BL, BNL, P, Q)
            if mode == "XY": u, v = traj[:, 0], traj[:, 1]
            elif mode == "BL_BNL": u, v = traj[:, 2], traj[:, 3]
            elif mode == "PQ": u, v = traj[:, 4], traj[:, 5]

            color = protocol_colors.get(proto_name, "black")
            dashes = protocol_dashes.get(proto_name, [1, 0])

            line, = ax.plot(u, v, linewidth=2, alpha=0.6, color=color, zorder=80)
            line.set_dashes(dashes)

            start_dot = ax.scatter(u[0], v[0], s=70, facecolors=start_color, edgecolors="black", linewidths=0.9, zorder=200, clip_on=False)
            ax.scatter(u[-1], v[-1], s=80, marker="^", facecolors=color, edgecolors="black", linewidths=0.9, zorder=300, clip_on=False)

            if state_dot is None: state_dot = start_dot

        if canonical_label not in legend_entries and state_dot is not None:
            legend_entries[canonical_label] = state_dot

    # --- Axes, Limits & Legend ---
    if mode == "BL_BNL":
        ax.set_xlabel(r"$B_\mathrm{L}$", fontsize=20)
        ax.set_ylabel(r"$B_{\mathrm{NL}}$", fontsize=20)
        ax.set_xlim(0, 2); ax.set_ylim(0, 3)
    elif mode == "PQ":
        ax.set_xlabel(r"$P$", fontsize=20)
        ax.set_ylabel(r"$Q$", fontsize=20)
        ax.set_xlim(0.25, 1); ax.set_ylim(0.0, 1)
    elif mode == "XY":
        ax.set_xlabel(r"$X$", fontsize=20)
        ax.set_ylabel(r"$Y$", fontsize=20)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)

    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.25)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_zorder(50)

    ax.legend(handles=list(legend_entries.values()), labels=list(legend_entries.keys()), loc="best", fontsize=12, frameon=False)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_{mode}.pdf", dpi=300, bbox_inches="tight")
    else:
        plt.close(fig)

# ==========================================
# 5. MAIN EXECUTION (UPDATED)
# ==========================================

def main():
    np.random.seed(70)
    
    states = {
         "Mixed_Ent1": random_mixed_entangled_state(),
         "Mixed_Ent2": random_mixed_entangled_state(),
         "Mixed_Ent3": random_mixed_entangled_state(),
         
         "Mixed_Sep1": random_mixed_separable_state(),
         "Mixed_Sep2": random_mixed_separable_state(),
         "Mixed_Sep3": random_mixed_separable_state(),
         
         "Classical_CC1": random_classical_CC_state(),
         "Classical_CC2": random_classical_CC_state(),
         "Classical_CC3": random_classical_CC_state(),
         
         "CHSH_Werner": chsh_strong_werner_state(p=0.01),}

    data = defaultdict(dict)
    print("Running Simulations...")
    
    for label, rho in states.items():
        # Using Thermal Purification as an example
        traj_thermal = protocol_thermal_purification(rho, step_size=0.1, steps=100)
        data[label]["Thermal"] = traj_thermal
        
    print("Generating Master Plots in all 3 Geometries...")
    for target_plane in ["XY", "BL_BNL", "PQ"]:
        print(f" -> Rendering {target_plane} Plane...")
        plot_purification_geometry(data, mode=target_plane, save_prefix="Purification", show=False)
    print("Done. Check your directory for the generated PDF files.")

if __name__ == "__main__":
    main()
