import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import os
import itertools
from scipy.optimize import brentq

# ==========================================
# 1. UNIFIED CONFIGURATION
# ==========================================
CONFIG = {
    "N_QUBITS": 7,              # Change to 7 if you have enough VRAM!
    "R_VALUES": [0.4, 0.6, 0.8],
    "N_SAMPLES_PER_R": 10000,
    "BATCH_SIZE": 500,          # Safe for 16GB VRAM
    "PAULI_CHUNK_SIZE": 2048,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # File I/O
    "OUTPUT_FILE": "magic_data_combined.npz",
    "OUTPUT_PLOT": f"{CONFIG['N_QUBITS']}theta_magic_binned.pdf",
    
    # Plotting Settings
    "N_BINS": 10000,
    "R_COLORS": {
        "R_0.4": "#A25D5F",
        "R_0.6": "#5FA25D",
        "R_0.8": "#5D5FA2",
    },
    "FONT_PATH": "./times.ttf"
}

# Set double precision (Critical for Magic stability)
torch.set_default_dtype(torch.float64)

# ==========================================
# 2. OPTIONAL FONT SETUP
# ==========================================
if os.path.exists(CONFIG["FONT_PATH"]):
    fm.fontManager.addfont(CONFIG["FONT_PATH"])
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.8,
    })
else:
    print("Custom font not found. Using matplotlib defaults.")
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.8,
    })

# ==========================================
# 3. GPU GENERATION FUNCTIONS
# ==========================================
def get_spectrum_tensor(R_target, n_qubits, device):
    dim = 2**n_qubits
    P_target = 1.0 / (dim - R_target*(dim - 1))
    
    def purity_diff(x):
        l1 = x + (1-x)/dim
        l2 = (1-x)/dim
        current_P = l1**2 + (dim-1)*(l2**2)
        return current_P - P_target
    
    x_sol = brentq(purity_diff, 0.0, 1.0)
    evals = torch.full((dim,), (1-x_sol)/dim, dtype=torch.complex128, device=device)
    evals[0] += x_sol
    return torch.diag(evals), P_target

def random_unitary_batch(batch_size, dim, device):
    Z = torch.randn(batch_size, dim, dim, dtype=torch.complex128, device=device)
    Q, R = torch.linalg.qr(Z)
    diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
    phases = diag_R / diag_R.abs()
    return Q * phases.unsqueeze(-2)

def random_local_unitary_batch(batch_size, n_qubits, device):
    U = torch.eye(1, dtype=torch.complex128, device=device).unsqueeze(0).expand(batch_size, 1, 1)
    for _ in range(n_qubits):
        u_sub = random_unitary_batch(batch_size, 2, device)
        b, d1, _ = U.shape
        b, d2, _ = u_sub.shape
        U = U.view(b, d1, 1, d1, 1) * u_sub.view(b, 1, d2, 1, d2)
        U = U.reshape(b, d1*d2, d1*d2)
    return U

def generate_pauli_strings(n_qubits):
    indices = np.array(list(itertools.product(range(4), repeat=n_qubits)))
    mask = np.sum(indices, axis=1) > 0
    return indices[mask]

def construct_paulis_on_gpu(indices_chunk, device):
    n_ops, n_qubits = indices_chunk.shape
    dim = 2**n_qubits
    I = torch.eye(2, dtype=torch.complex128, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    bases = torch.stack([I, X, Y, Z])
    
    paulis = torch.eye(1, dtype=torch.complex128, device=device).unsqueeze(0).repeat(n_ops, 1, 1)
    for k in range(n_qubits):
        op_idx = indices_chunk[:, k]
        current_ops = bases[op_idx]
        b, d1, _ = paulis.shape
        paulis = paulis.view(b, d1, 1, d1, 1) * current_ops.view(b, 1, 2, 1, 2)
        paulis = paulis.reshape(b, d1*2, d1*2)
    return paulis

def solve_and_measure_batch(R_val, n_samples, n_qubits, pauli_codes):
    dim = 2**n_qubits
    device = CONFIG['DEVICE']
    batch_size = CONFIG['BATCH_SIZE']
    
    Lambda, P_val = get_spectrum_tensor(R_val, n_qubits, device)
    weights = np.sum(pauli_codes != 0, axis=1)
    is_local_np = (weights == 1)
    is_local_gpu = torch.tensor(is_local_np, device=device)
    
    local_indices = pauli_codes[is_local_np]
    local_paulis_ops = construct_paulis_on_gpu(local_indices, device)
    
    all_X, all_Y, all_M = [], [], []
    
    for b_start in range(0, n_samples, batch_size):
        curr_b_size = min(batch_size, n_samples - b_start)
        
        U_loc = random_local_unitary_batch(curr_b_size, n_qubits, device)
        U_glo = random_unitary_batch(curr_b_size, dim, device)
        
        rho_loc = U_loc @ Lambda @ U_loc.mH
        rho_glo = U_glo @ Lambda @ U_glo.mH
        
        def measure_X_batch(rho_batch):
            coeffs = torch.einsum('bij, kij -> bk', rho_batch, local_paulis_ops)
            BL = torch.sum(coeffs.real**2, dim=1)
            denom = (dim - 1) * P_val
            return torch.sqrt(torch.clamp(BL / denom, min=0))

        X_max = measure_X_batch(rho_loc)
        X_min = measure_X_batch(rho_glo)
        target_X = torch.rand(curr_b_size, device=device) * (X_max - X_min) + X_min
        
        V = U_glo @ U_loc.mH
        evals, W = torch.linalg.eig(V)
        log_evals = torch.log(evals)
        
        t_low = torch.zeros(curr_b_size, device=device)
        t_high = torch.ones(curr_b_size, device=device)
        
        for _ in range(20):
            t_mid = (t_low + t_high) / 2
            interp_evals = torch.exp(t_mid.unsqueeze(1) * log_evals)
            U_t = (W * interp_evals.unsqueeze(1)) @ W.mH @ U_loc
            rho_t = U_t @ Lambda @ U_t.mH
            X_curr = measure_X_batch(rho_t)
            mask_too_local = X_curr > target_X
            t_low = torch.where(mask_too_local, t_mid, t_low)
            t_high = torch.where(mask_too_local, t_high, t_mid)

        t_final = (t_low + t_high) / 2
        interp_evals = torch.exp(t_final.unsqueeze(1) * log_evals)
        U_t = (W * interp_evals.unsqueeze(1)) @ W.mH @ U_loc
        rho_final = U_t @ Lambda @ U_t.mH
        
        sum_sq_total = torch.zeros(curr_b_size, device=device)
        sum_sq_local = torch.zeros(curr_b_size, device=device)
        sum_fourth = torch.zeros(curr_b_size, device=device)
        
        n_all_ops = len(pauli_codes)
        chunk_size = CONFIG['PAULI_CHUNK_SIZE']
        
        for k in range(0, n_all_ops, chunk_size):
            end_k = min(k + chunk_size, n_all_ops)
            idx_chunk = pauli_codes[k:end_k]
            ops_chunk = construct_paulis_on_gpu(idx_chunk, device)
            is_loc_chunk = is_local_gpu[k:end_k]
            
            c_vals = torch.einsum('bij, kij -> bk', rho_final, ops_chunk).real
            sq = c_vals**2
            sum_sq_total += torch.sum(sq, dim=1)
            sum_fourth += torch.sum(sq**2, dim=1)
            sum_sq_local += torch.sum(sq * is_loc_chunk.unsqueeze(0), dim=1)
            
            del ops_chunk, c_vals
            
        D_float = float(dim)
        BL = sum_sq_local
        BNL = sum_sq_total - sum_sq_local
        P_calc = (sum_sq_total + 1.0) / D_float
        denom = (D_float - 1.0) * P_calc
        
        X_res = torch.sqrt(torch.clamp(BL / denom, min=0))
        Y_res = torch.sqrt(torch.clamp(BNL / denom, min=0))
        
        num = 1.0 + sum_fourth
        den = 1.0 + sum_sq_total
        M_res = -torch.log2(num / den)
        
        all_X.append(X_res.cpu().numpy())
        all_Y.append(Y_res.cpu().numpy())
        all_M.append(M_res.cpu().numpy())
        
        if b_start % 2000 == 0:
            print(f"   > Processed {b_start}/{n_samples}...")

    return np.concatenate(all_X), np.concatenate(all_Y), np.concatenate(all_M)

# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================
def magic_upper_bound_general(R, n_qubits):
    D = 2.0**n_qubits
    D_minus_1 = D - 1.0
    N_ops = 4.0**n_qubits - 1.0
    
    numerator = R * D_minus_1
    denominator = D - (R * D_minus_1)
    B = np.divide(numerator, denominator, out=np.zeros_like(R), where=denominator!=0)
    
    term1 = np.log2(1.0 + B)
    term2 = np.log2(1.0 + (B**2 / N_ops))
    return term1 - term2

def bin_arc_max(X, Y, Z, n_bins=1000):
    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[valid], Y[valid], Z[valid]
    
    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])
        
    theta = np.arctan2(Y, X)
    t_min, t_max = theta.min(), theta.max()
    
    if np.isclose(t_min, t_max):
        bins = np.array([t_min, t_max + 1e-5])
    else:
        bins = np.linspace(t_min, t_max, n_bins + 1)
        
    idx = np.digitize(theta, bins) - 1
    Xb, Yb, Zb = [], [], []
    
    for k in range(n_bins):
        mask = (idx == k)
        if not np.any(mask): continue
            
        subset_Z = Z[mask]
        subset_X = X[mask]
        subset_Y = Y[mask]
        
        j = np.argmax(subset_Z)
        Xb.append(subset_X[j])
        Yb.append(subset_Y[j])
        Zb.append(subset_Z[j])
        
    return np.array(Xb), np.array(Yb), np.array(Zb)

def plot_theta_magic(XY_by_R, Magic_by_R, R_list, filename=None):
    if not XY_by_R:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for key in R_list:
        if key not in XY_by_R: continue
            
        X, Y = XY_by_R[key]
        Z = Magic_by_R[key]

        print(f"Binning {key} with {CONFIG['N_BINS']} bins...")
        Xb, Yb, Zb = bin_arc_max(X, Y, Z, n_bins=CONFIG['N_BINS'])
        
        if len(Xb) == 0: continue

        theta = np.arctan2(Yb, Xb) / (np.pi / 2)
        color = CONFIG['R_COLORS'].get(key, "black")
        
        try:
            R_val = float(key.split("_")[1])
        except:
            R_val = 0.5

        ax.plot(theta, Zb, marker="o", linestyle="none", markersize=2, 
                alpha=0.6, color=color, label=rf"$R={R_val:.2f}$", rasterized=True)

        Mmax = magic_upper_bound_general(np.array([R_val]), CONFIG['N_QUBITS'])[0]

        ax.hlines(Mmax, 0, 1, colors=color, linestyles="--", linewidth=2.0, alpha=0.9)
        ax.text(0.92, Mmax + 0.02, rf"$\widetilde{{M}}_2={Mmax:.3f}$",
                fontsize=11, color=color, ha="center", va="bottom", fontweight='bold')

    ax.set_xlim(0.735, 0.965)
    
    max_theory = magic_upper_bound_general(np.array([max(CONFIG['R_VALUES'])]), CONFIG['N_QUBITS'])[0]
    ax.set_ylim(0, max_theory + 0.5) 

    ax.set_xlabel(r"$2\theta/\pi$", fontsize=16)
    ax.set_ylabel(r"$\widetilde{M}_2$", fontsize=16)

    ticks = ax.yaxis.get_major_ticks()
    if len(ticks) > 0:
        ticks[0].label1.set_visible(False)

    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=12, handletextpad=0.02)
    ax.set_box_aspect(1)
    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {filename}")

    plt.show()

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    t0 = time.time()
    
    print("Generating Pauli indices...")
    pauli_codes = generate_pauli_strings(CONFIG['N_QUBITS'])
    print(f"Total Pauli Operators: {len(pauli_codes)}\n")
    
    final_data = {}
    XY_by_R = {}
    Magic_by_R = {}
    r_keys = []
    
    # --- PHASE 1: GENERATION ---
    with torch.no_grad():
        for R in CONFIG['R_VALUES']:
            print(f"=== Processing R={R} ===")
            st = time.time()
            X, Y, M = solve_and_measure_batch(R, CONFIG['N_SAMPLES_PER_R'], CONFIG['N_QUBITS'], pauli_codes)
            
            # Save raw data dict for .npz file
            final_data[f"R_{R}_X"] = X
            final_data[f"R_{R}_Y"] = Y
            final_data[f"R_{R}_M"] = M
            
            # Format directly for plotting function
            r_key = f"R_{R}"
            r_keys.append(r_key)
            XY_by_R[r_key] = (X, Y)
            Magic_by_R[r_key] = M
            
            print(f"   > Generation done in {time.time()-st:.2f}s\n")
            
    # Save the arrays to disk just in case
    print(f"Saving raw data to {CONFIG['OUTPUT_FILE']}...")
    np.savez_compressed(CONFIG['OUTPUT_FILE'], **final_data)
    
    # --- PHASE 2: PLOTTING ---
    print("\n=== Generating Plot ===")
    plot_theta_magic(XY_by_R, Magic_by_R, R_list=r_keys, filename=CONFIG['OUTPUT_PLOT'])
