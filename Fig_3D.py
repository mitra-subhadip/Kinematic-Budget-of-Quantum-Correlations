import torch
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ==========================================
#  1. GPU CONFIGURATION & SAFETY
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    try:
        torch.backends.cuda.preferred_linalg_library('magma')
    except:
        pass

# Constants (Double Precision)
X_TRINE, Y_TRINE = 0.0, 0.7745966692414834
X_DIMER, Y_DIMER = 0.4472135954999579, 0.7745966692414834
X_PURE,  Y_PURE  = 0.7745966692414834, 0.6324555320336759

# ==========================================
#  2. VECTORIZED PHYSICS KERNEL (PyTorch)
# ==========================================

def get_purity_wall_gpu(x):
    """Vectorized calculation of the Purity Wall (Elliptical Roof)."""
    val = 1.6 - 2 * (x ** 2)
    return torch.sqrt(torch.clamp(val, min=0))

def get_qc_envelope_gpu(x):
    """Vectorized calculation of QC Envelope (Curved Segment)."""
    y_out = torch.zeros_like(x)
    
    mask_zero = x > X_PURE
    mask_flat = x <= X_DIMER
    
    # Segment 1: Trine to Dimer (Flat)
    y_out[mask_flat] = Y_DIMER
    
    # Segment 2: Dimer to Pure (Curved)
    # Analytically derived from the parametric B_L and B_NL equations
    mask_curve = (~mask_flat) & (~mask_zero)
    x_curve = x[mask_curve]
    y_out[mask_curve] = torch.sqrt(0.7 - 0.5 * (x_curve ** 2))
    
    y_out[mask_zero] = 0.0
    return y_out

def check_ppt_gpu(rho_batch):
    """Batched PPT Check on GPU (Double Precision)."""
    # Reshape: (N, 2, 3, 2, 3)
    rho_tensor = rho_batch.view(-1, 2, 3, 2, 3)
    # Partial Transpose on B: Swap indices 2 (B1) and 4 (B2)
    rho_pt = rho_tensor.permute(0, 1, 4, 3, 2).contiguous()
    rho_pt = rho_pt.view(-1, 6, 6)
    
    try:
        eigvals = torch.linalg.eigvalsh(rho_pt)
    except RuntimeError:
        return torch.zeros(rho_batch.shape[0], dtype=torch.bool, device=device)
        
    min_eigs = eigvals[:, 0]
    return min_eigs > -1e-12

def generate_batch_with_data(batch_size, rank):
    """Generates a batch of states and returns stats AND data for plotting."""
    
    # 1. Generate Random States
    real_part = torch.randn(batch_size, 6, rank, device=device, dtype=torch.float64)
    imag_part = torch.randn(batch_size, 6, rank, device=device, dtype=torch.float64)
    G = torch.complex(real_part, imag_part)
    rho = torch.matmul(G, G.mH)
    
    # Normalize
    traces = torch.einsum('nii->n', rho).real
    traces = torch.clamp(traces, min=1e-9)
    rho = rho / traces[:, None, None]
    
    # 2. Coordinates Calculation
    # Global Purity
    P = (rho.abs() ** 2).sum(dim=(1, 2))
    
    # Marginals (CORRECTED EINSUM)
    r_view = rho.view(-1, 2, 3, 2, 3)
    
    # Trace over B (indices 2 and 4): Use 'nijkj->nik' to sum diagonal
    rho_A = torch.einsum('nijkj->nik', r_view)
    
    # Trace over A (indices 1 and 3): Use 'nijil->njl' to sum diagonal
    rho_B = torch.einsum('nijil->njl', r_view)
    
    PA = (rho_A.abs() ** 2).sum(dim=(1, 2))
    PB = (rho_B.abs() ** 2).sum(dim=(1, 2))
    
    BL = torch.clamp((2 * PA - 1) + (3 * PB - 1), min=0)
    BNL = torch.clamp((6 * P - 1) - BL, min=0)
    
    denom = torch.clamp(5 * P, min=1e-9)
    X = torch.sqrt(BL / denom)
    Y = torch.sqrt(BNL / denom)
    
    # 3. Filter: Strictly Above QC Envelope AND Below Wall
    Y_wall = get_purity_wall_gpu(X)
    Y_qc = get_qc_envelope_gpu(X)
    
    valid_mask = (Y <= Y_wall + 1e-5) & (Y > Y_qc + 0.002) & (Y > Y_PURE)
    
    if not valid_mask.any():
        return 0, 0, None, None, None
        
    # 4. PPT Check
    valid_rho = rho[valid_mask]
    is_separable = check_ppt_gpu(valid_rho)
    
    n_ppt = is_separable.sum().item()
    n_npt = (~is_separable).sum().item()
    
    # Return Data for Plotting
    out_X = X[valid_mask].cpu().numpy()
    out_Y = Y[valid_mask].cpu().numpy()
    out_is_ppt = is_separable.cpu().numpy()
    
    return n_ppt, n_npt, out_X, out_Y, out_is_ppt

# ==========================================
#  3. PLOTTING UTILS (IMMUTABLE GEOMETRY)
# ==========================================
def plot_results(ppt_x, ppt_y, npt_x, npt_y, total_points):
    """Generates the final PDF plot using the Immutable Geometry Standard."""
    print("\nGenerating Plot...")
    
    # Font Config
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
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # --- A. DEFINE GEOMETRY DATA ---
    X_grid = np.linspace(0, 1.0, 5000)
    X_max_ent = np.sqrt(0.1)  # ~0.316
    X_prod    = np.sqrt(0.6)  # ~0.775

    # 1. Full Purity Arc
    Y_R1 = np.sqrt(np.maximum(0, 1 - X_grid**2))

    # 2. Entanglement Wedge
    w = np.linspace(2/3, 1, 500)
    w2 = w**2
    P_qq = (3*w2 - 2*w + 1) / 2.0
    BL_qq = 4.5*w2 - 6*w + 2
    BNL_qq = 4.5*w2
    X_qq = np.sqrt(BL_qq / (5 * P_qq))
    Y_qq = np.sqrt(BNL_qq / (5 * P_qq))

    def get_ent_fill(x_vals):
        y = np.interp(x_vals, X_qq, Y_qq, left=0, right=0)
        y[x_vals > X_max_ent] = 0
        return y
    Y_ent_fill = get_ent_fill(X_grid)

    # 3. QC Envelope (Corrected Piecewise Projection)
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

    Y_qc = get_qc_envelope(X_grid)

    # 4. Q-Curve (Positivity Wall)
    Y_qcurve_raw = np.sqrt(np.maximum(0, 1.6 - 2*X_grid**2))

    # 5. Abs Separability Ball
    R_ball = np.sqrt(0.2)
    Y_sep_ball = np.sqrt(np.maximum(0, R_ball**2 - X_grid**2))

    # --- B. COLORS ---
    col_Blue   = "#5AA9E6"    # Entangled Region
    col_Sep    = "#FFEE99"     # Separable region (Ball)
    col_Gray   = "lightgray" # Unfeasible pockets
    col_CC     = "#FFEBEB"   # Classical Region (Reddish tint)
    col_UPunfeas = "#B8A597"
    
    line_purple = "#975AE6"
    line_red    = "#FF6392"
    line_sep    = line_red 

    # --- C. DRAW FILLED REGIONS ---
    mask_left  = X_grid <= X_max_ent          
    mask_mid   = (X_grid > X_max_ent) & (X_grid <= X_prod) 
    mask_tail  = (X_grid > X_prod)            

    # Yellow/Red Region (Classical)
    ax.fill_between(X_grid, 0, Y_qc, where=(X_grid <= X_prod), color=col_CC, zorder=3,alpha=1)
    ax.fill_between(X_grid, 0, Y_qcurve_raw, where=mask_tail, color=col_CC, zorder=3,alpha=1)

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

        
    
    # Abs Separability Ball (Yellow)
    ax.fill_between(X_grid, 0, Y_sep_ball, where=(X_grid <= R_ball), color=col_Sep, zorder=8,alpha=0.65)

    # Blue Region (Entangled) - MERGED 
    # 1. Create a single 'Upper Boundary' array for the entire entangled region
    Y_upper_combined = np.where(X_grid <= X_max_ent, Y_ent_fill, Y_R1)

    # 2. Define the total horizontal range for entanglement (from 0 to X_prod)
    mask_entangled_total = (X_grid <= X_prod)

    # 3. Fill the entire region in ONE call
    ax.fill_between(X_grid, Y_qc, Y_upper_combined, 
                    where=mask_entangled_total, 
                    alpha=0.45, 
                    color=col_Blue, 
                    edgecolor="none", 
                    linewidth=0,
                    zorder=2)

    # Gray Pockets
    ax.fill_between(X_grid, Y_ent_fill, Y_R1, where=mask_left, color=col_UPunfeas, zorder=2)
    ax.fill_between(X_grid, Y_qcurve_raw, Y_R1, where=mask_tail, color=col_Gray, zorder=2)
    ax.plot(X_root, Y_root, color="deeppink", linewidth=2.5, linestyle='-', zorder=12, label="C boundary")
    # --- D. SCATTER SIMULATION DATA ---
    # NPT (Red) - Downsample
    if len(npt_x) > 10000000001:
        indices = np.random.choice(len(npt_x), 10000000001, replace=False)
        ax.scatter(np.array(npt_x)[indices], np.array(npt_y)[indices], s=2, c="#274C77", alpha=0.3, edgecolors='none', zorder=10,rasterized=True, label = "Entangled")
    else:
        ax.scatter(npt_x, npt_y, s=2, c="#274C77", alpha=0.3, edgecolors='none', zorder=10,rasterized=True, label = "Entangled")

    # PPT (Green) - All
    if len(ppt_x) > 0:
        ax.scatter(ppt_x, ppt_y, s=15, c="#2CA02C", alpha=1.0, edgecolors='black', linewidth=0.3, zorder=20,rasterized=True)

    # --- E. DRAW LINES ---
    # Purity Arc Segments
    ax.plot(X_grid[mask_left], Y_R1[mask_left], color="#49616E", linewidth=2.5, linestyle='--', zorder=11)
    mask_mid_idx = (X_grid >= X_max_ent) & (X_grid <= X_prod)
    ax.plot(X_grid[mask_mid_idx], Y_R1[mask_mid_idx], color="#274C77", linewidth=2.5, linestyle='-', zorder=11)
    ax.plot(X_grid[mask_tail], Y_R1[mask_tail], color="#49616E", linewidth=2.5, linestyle='--', zorder=11)

    # Entanglement Boundary
    ax.plot(X_qq, Y_qq, color="black", linewidth=2.0, zorder=9)

    # QC Envelope
    ax.plot(X_grid[X_grid <= X_prod], Y_qc[X_grid <= X_prod], color=line_purple, linewidth=2.5, zorder=8, label="QC envelope")

    # Q-Curve Tail
    ax.plot(X_grid[mask_tail], Y_qcurve_raw[mask_tail], color="black", linewidth=2.0, linestyle='-', zorder=8)
    
    # Abs Sep Ball
    ax.plot(X_grid[X_grid <= R_ball], Y_sep_ball[X_grid <= R_ball], color=line_sep, linewidth=2.0, linestyle='--', zorder=15, label="Separable boundary")

    # Floor Line
    ax.plot([0, np.sqrt(0.8)], [0, 0], color="red", linewidth=1.5, linestyle='--', zorder=11)

    # --- F. ANCHORS & LABELS ---
    pt_max_ent = (X_qq[-1], Y_qq[-1])
    ax.scatter(*pt_max_ent, s=60, color="#99CAF0", edgecolors="black", zorder=20, clip_on=False)

    pt_mixed_ent = (X_qq[0], Y_qq[0])
    ax.scatter(*pt_mixed_ent, s=60, color="#5CD68F", edgecolors="black", zorder=20, clip_on=False)

    pt_prod = (np.sqrt(0.6), np.sqrt(0.4))
    ax.scatter(*pt_prod, s=60, color="red", edgecolors="black", zorder=20, clip_on=False)

    ax.scatter(X_root[-1], Y_root[-1], s=60, color="deeppink", edgecolors="black", zorder=30, clip_on=False)


    # Other points
    ax.scatter(np.sqrt(0.6), 0, s=60, color="#BD7F55", edgecolors="black", zorder=20, clip_on=False)
    ax.scatter(np.sqrt(0.8), 0, s=60, color="orange", edgecolors="black", zorder=20, clip_on=False)
    ax.scatter(0, 0, s=60, color="#FFEE99", edgecolors="black", zorder=20, clip_on=False)

    # --- H. FINAL FORMAT ---
    ax.set_xlabel("$\mathrm{X}$", fontsize=20)
    ax.set_ylabel("$\mathrm{Y}$", fontsize=20)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = "Fig_3D.pdf"
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")
    plt.show()

# ==========================================
#  4. MAIN LOOP
# ==========================================

def main():
    # SETTINGS
    TARGET_POINTS = 100000000  # 1 Lakh
    BATCH_SIZE = 250_000 * 10    # Batch size
    
    # Data Collectors
    ppt_x_list, ppt_y_list = [], []
    npt_x_list, npt_y_list = [], []
    
    total_ppt = 0
    total_npt = 0
    total_valid = 0
    
    ranks = [1, 2, 3, 4]#, 5, 6]
    probs = [0.25, 0.25, 0.25, 0.25]#, 0.1, 0.1]
    
    print(f"==================================================")
    print(f" STARTING GPU SIMULATION (FIXED EINSUM TRACE)")
    print(f"==================================================")
    print(f" Target Points : {TARGET_POINTS:,}")
    print(f" Batch Size    : {BATCH_SIZE:,}")
    print(f"--------------------------------------------------")
    
    start_time = time.time()
    iteration = 0
    
    try:
        while total_valid < TARGET_POINTS:
            r = np.random.choice(ranks, p=probs)
            
            # Use the new batch function that returns data
            n_ppt, n_npt, bx, by, b_is_ppt = generate_batch_with_data(BATCH_SIZE, r)
            
            total_ppt += n_ppt
            total_npt += n_npt
            total_valid += (n_ppt + n_npt)
            

            # --- DATA COLLECTION FOR PLOTTING ---
            if bx is not None:
                # Store ALL Green (PPT) points
                if n_ppt > 0:
                    mask_ppt = b_is_ppt
                    ppt_x_list.extend(bx[mask_ppt])
                    ppt_y_list.extend(by[mask_ppt])
                
                # Store ALL Red (NPT) points
                if n_npt > 0:
                    mask_npt = ~b_is_ppt
                    npt_x_list.extend(bx[mask_npt])
                    npt_y_list.extend(by[mask_npt])
            # ------------------------------------

            iteration += 1
            
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_valid / elapsed if elapsed > 0 else 0
                percent = (total_valid / TARGET_POINTS) * 100
                sys.stdout.write(f"\r Progress: {percent:5.2f}% | Valid: {total_valid:,} | Rate: {rate/1e6:.2f} M/sec")
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        print("\nStopping early...")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    total_time = time.time() - start_time
    
    print(f"\n==================================================")
    print(f" SIMULATION COMPLETE")
    print(f"==================================================")
    print(f" Time Elapsed     : {total_time:.2f} s")
    print(f" Total Valid      : {total_valid:,}")
    print(f" Total PPT (Green): {total_ppt:,}")
    print(f" Total NPT (Red)  : {total_npt:,}")
    print(f"--------------------------------------------------")
    if total_valid > 0:
        ratio = total_npt / total_valid * 100
        print(f" Entangled Fraction Above QC: {ratio:.4f}%")
    print(f"==================================================")
    
    # Call Plotter
    plot_results(ppt_x_list, ppt_y_list, npt_x_list, npt_y_list, total_valid)

if __name__ == "__main__":
    main()