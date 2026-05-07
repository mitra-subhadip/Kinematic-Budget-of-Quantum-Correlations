import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time
from tqdm import tqdm 

# ==========================================
# 1. User Configuration
# ==========================================
# Choose: "CCQ" (Biseparable states) or "CQQ" (GME states)
ENVELOPE_CHOICE = "CCQ"  
TARGET_POINTS = 100000  
BATCH_SIZE = 20000     

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
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.size": 12
})

# ==========================================
# 2. PyTorch GPU Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 3. Core Physics Functions (Optimized)
# ==========================================
def generate_random_global_matrices(batch_size):
    """Generates random 8x8 matrices (Standard for CQQ/GME)."""
    rank = torch.randint(1, 5, (1,)).item()
    G = torch.randn((batch_size, 8, rank), dtype=torch.float64, device=device) + \
        1j * torch.randn((batch_size, 8, rank), dtype=torch.float64, device=device)
    rho = torch.bmm(G, G.transpose(1, 2).conj())
    return rho / torch.einsum('bii->b', rho).real.view(-1, 1, 1)

def generate_strictly_biseparable_matrices(batch_size):
    """Explicitly constructs rho_A (x) rho_BC (Standard for CCQ)."""
    # 1. Subsystem A (2x2)
    rank_A = torch.randint(1, 3, (1,)).item()
    G_A = torch.randn((batch_size, 2, rank_A), dtype=torch.float64, device=device) + \
          1j * torch.randn((batch_size, 2, rank_A), dtype=torch.float64, device=device)
    rho_A = torch.bmm(G_A, G_A.transpose(1, 2).conj())
    rho_A = rho_A / torch.einsum('bii->b', rho_A).real.view(-1, 1, 1)

    # 2. Subsystem BC (4x4)
    rank_BC = torch.randint(1, 5, (1,)).item()
    G_BC = torch.randn((batch_size, 4, rank_BC), dtype=torch.float64, device=device) + \
           1j * torch.randn((batch_size, 4, rank_BC), dtype=torch.float64, device=device)
    rho_BC = torch.bmm(G_BC, G_BC.transpose(1, 2).conj())
    rho_BC = rho_BC / torch.einsum('bii->b', rho_BC).real.view(-1, 1, 1)

    # 3. Tensor Product
    rho = torch.einsum('bij,bkl->bikjl', rho_A, rho_BC).reshape(batch_size, 8, 8)

    # 4. Randomize Cut
    rho_tensor = rho.view(batch_size, 2, 2, 2, 2, 2, 2)
    perm_choice = torch.randint(0, 3, (1,)).item()
    if perm_choice == 1:
        rho_tensor = rho_tensor.permute(0, 2, 1, 3, 5, 4, 6)
    elif perm_choice == 2:
        rho_tensor = rho_tensor.permute(0, 3, 2, 1, 6, 5, 4)
    return rho_tensor.reshape(batch_size, 8, 8)

def compute_XY_coordinates(rho):
    B = rho.shape[0]
    rho_tensor = rho.view(B, 2, 2, 2, 2, 2, 2)
    rho1 = torch.einsum('b i j k m j k -> b i m', rho_tensor)
    rho2 = torch.einsum('b j i k j m k -> b i m', rho_tensor)
    rho3 = torch.einsum('b j k i j k m -> b i m', rho_tensor)
    P_global = torch.einsum('bij,bji->b', rho, rho).real
    P1 = torch.einsum('bij,bji->b', rho1, rho1).real
    P2 = torch.einsum('bij,bji->b', rho2, rho2).real
    P3 = torch.einsum('bij,bji->b', rho3, rho3).real
    BL = (2*P1 - 1) + (2*P2 - 1) + (2*P3 - 1)
    BNL = (8*P_global - 1) - BL
    X = torch.sqrt(torch.clamp(BL / (7 * P_global), min=0))
    Y = torch.sqrt(torch.clamp(BNL / (7 * P_global), min=0))
    return X, Y

def check_entanglement_criteria(rho, choice):
    B = rho.shape[0]
    rho_tensor = rho.view(B, 2, 2, 2, 2, 2, 2)
    rho_pt1 = rho_tensor.permute(0, 4, 2, 3, 1, 5, 6).reshape(B, 8, 8)
    rho_pt2 = rho_tensor.permute(0, 1, 5, 3, 4, 2, 6).reshape(B, 8, 8)
    rho_pt3 = rho_tensor.permute(0, 1, 2, 6, 4, 5, 3).reshape(B, 8, 8)
    min_eig1 = torch.linalg.eigvalsh(rho_pt1)[:, 0]
    min_eig2 = torch.linalg.eigvalsh(rho_pt2)[:, 0]
    min_eig3 = torch.linalg.eigvalsh(rho_pt3)[:, 0]
    tol = -1e-6
    npt1, npt2, npt3 = min_eig1 < tol, min_eig2 < tol, min_eig3 < tol
    if choice == "CCQ":
        return (npt1.int() + npt2.int() + npt3.int()) == 2
    return npt1 & npt2 & npt3

# ==========================================
# 4. Main Generation Loop
# ==========================================
print(f"Starting MC Generation for {ENVELOPE_CHOICE}...")
start_time = time.time()
collected_X_pass, collected_Y_pass = [], []
collected_X_fail, collected_Y_fail = [], []

pbar = tqdm(total=TARGET_POINTS, desc="States Found")

while (len(collected_X_pass) + len(collected_X_fail)) < TARGET_POINTS:
    # Switch generator based on target
    if ENVELOPE_CHOICE == "CCQ":
        rho = generate_strictly_biseparable_matrices(BATCH_SIZE)
        X, Y = compute_XY_coordinates(rho)
        env_mask = (21 * Y**2 + 14 * X**2) > 18
    else:
        rho = generate_random_global_matrices(BATCH_SIZE)
        X, Y = compute_XY_coordinates(rho)
        env_mask = Y > np.sqrt(6/7)
        
    rho_f, X_f, Y_f = rho[env_mask], X[env_mask], Y[env_mask]
    if len(rho_f) == 0: continue
    
    ent_mask = check_entanglement_criteria(rho_f, ENVELOPE_CHOICE)
    
    collected_X_pass.extend(X_f[ent_mask].cpu().numpy())
    collected_Y_pass.extend(Y_f[ent_mask].cpu().numpy())
    collected_X_fail.extend(X_f[~ent_mask].cpu().numpy())
    collected_Y_fail.extend(Y_f[~ent_mask].cpu().numpy())
    
    pbar.update(min(len(collected_X_pass)+len(collected_X_fail), TARGET_POINTS) - pbar.n)

pbar.close()
X_pass, Y_pass = np.array(collected_X_pass), np.array(collected_Y_pass)

# ==========================================
# 5. IMMUTABLE PLOTTING BLOCK (Untouched)
# ==========================================
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
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

X_plot = np.linspace(0, 1.05, 5000)
X_bell = np.sqrt(1/7) 
X_pp = np.sqrt(3/7)   
X_cliff = np.sqrt(4/7) 

Y_purity_arc = np.sqrt(np.maximum(0, 1 - X_plot**2))

def get_physical_boundary(x_vals):
    y_vals = np.zeros_like(x_vals)
    mask_1 = x_vals <= X_pp
    y_vals[mask_1] = np.sqrt(np.maximum(0, 1 - x_vals[mask_1]**2))
    mask_2 = (x_vals > X_pp) & (x_vals <= X_cliff)
    y_vals[mask_2] = np.sqrt(np.maximum(0, 10/7 - 2*x_vals[mask_2]**2))
    mask_3 = x_vals > X_cliff
    y_vals[mask_3] = 0
    return y_vals

Y_physical = get_physical_boundary(X_plot)
Y_cqq_val = np.sqrt(6/7)
Y_cqq = np.full_like(X_plot, Y_cqq_val)
Y_cqq[X_plot > X_bell] = np.nan 

Y_ccq = np.sqrt(np.maximum(0, (18 - 14*X_plot**2) / 21))
Y_ccq = np.minimum(Y_ccq, Y_purity_arc) 

R_ball = np.sqrt(1/7)
Y_ball = np.sqrt(np.maximum(0, R_ball**2 - X_plot**2))

fig, ax = plt.subplots(figsize=(7, 7))

# Colors
col_Entangled = "#7FC8F8"     # Blue
col_Bisep     = "#5AA9E6" # Purple Tint
col_Sep        = "#FFEE99"   # Yellow (Inner Abs Sep)
col_Sep_Outer = "#FFEBEB"   # Very Light Green (Outer Separable)
col_Gray      = "lightgray"   # Unfeasible Gap

line_blue   = "#274C77"     
line_purple = "#975AE6"
line_red     = "#D62728"      

mask_phys = X_plot <= X_cliff

ax.fill_between(X_plot, Y_physical, Y_purity_arc, color=col_Gray, zorder=6)
ax.fill_between(X_plot, Y_cqq_val, Y_purity_arc, where=(X_plot <= X_bell), color=col_Entangled, zorder=1,alpha=0.4)
Y_top_bisep = np.minimum(Y_physical, Y_cqq_val)
Y_bot_bisep = np.minimum(Y_physical, Y_ccq)
ax.fill_between(X_plot, Y_bot_bisep, Y_top_bisep, where=mask_phys, color=col_Bisep, zorder=2,alpha=0.45)
ax.fill_between(X_plot, 0, Y_bot_bisep, where=mask_phys, color=col_Sep_Outer, zorder=2,alpha=1)
ax.fill_between(X_plot, 0, Y_ball, where=(X_plot <= R_ball), color=col_Sep, zorder=2,alpha=0.65)

ax.plot(X_plot, Y_purity_arc, color="#49616E", linestyle="--", linewidth=2.5,  zorder=7)
ax.plot(X_plot[X_plot <= X_pp], Y_physical[X_plot <= X_pp], color=line_blue, linewidth=2.5, zorder=11)
ax.plot(X_plot[(X_plot >= X_pp) & (X_plot <= X_cliff)], Y_physical[(X_plot >= X_pp) & (X_plot <= X_cliff)], 
        color="black", linewidth=2.5, zorder=10)
ax.plot([X_cliff, X_cliff], [0, np.sqrt(2/7)], color="black", linewidth=2.5, zorder=10)
ax.plot(X_plot[X_plot <= X_bell], Y_cqq[X_plot <= X_bell],"-.", color="#274C77", linewidth=2.0, zorder=9, label="$Q^2C$ envelope")
ax.plot(X_plot[X_plot <= X_pp], Y_ccq[X_plot <= X_pp], color=line_purple, linewidth=2.0, zorder=9, label="$QC^2$ envelope")
ax.plot(X_plot[X_plot <= R_ball], Y_ball[X_plot <= R_ball], color=line_red, linestyle="--", linewidth=2.0, zorder=15)
ax.plot([R_ball, X_cliff], [0, 0], color=line_red, linestyle="--", linewidth=2.0, zorder=15)

ax.scatter(X_pass, Y_pass, color='#274C77', edgecolors='none', linewidth=0.2, s=2, alpha=0.6, zorder=8,rasterized=True, label="GME states")

def plot_point(x, y, color, label, offset=(0,0), ha='center'):
    ax.scatter(x, y, s=70, color=color, edgecolors="black", zorder=20, clip_on=False)
    if label:
        ax.text(x+offset[0], y+offset[1], label, fontsize=11, ha=ha, zorder=25)

plot_point(0.0, 1.0, "#99CAF0", "", offset=(0.03, 0.01), ha='center')
plot_point(np.sqrt(1/21), np.sqrt(20/21), "#99CAF0", "", offset=(0.02, 0.01), ha='center')
plot_point(X_bell, Y_cqq_val, "#99CAF0", "", offset=(0.025, 0.02))
plot_point(X_pp, np.sqrt(4/7), "red", "", offset=(0.01, 0.01), ha='left')
plot_point(X_cliff, np.sqrt(2/7), "#9E7F67", "", offset=(0,0))
plot_point(X_cliff, 0, "orange", "", offset=(0,0))
ax.scatter(0, 0, s=70, color="#FFEE99", edgecolors="black", zorder=20, clip_on=False)
ax.set_xlabel("$X$", fontsize=20)
ax.set_ylabel("$Y$", fontsize=20)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
    spine.set_zorder(10)

plt.tight_layout()


if ENVELOPE_CHOICE == "CCQ":
    save_name = "Fig_3F.pdf"
elif ENVELOPE_CHOICE == "CQQ":
    save_name = "Fig_3G.pdf"

plt.savefig(save_name, dpi=300)
print(f"Figure saved as: {save_name}")

plt.show()