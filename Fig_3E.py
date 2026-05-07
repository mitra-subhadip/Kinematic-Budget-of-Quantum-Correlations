import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import os
import torch



num_sample=50000000
batchsize=30000
maxrank=9

def generate_and_filter_states_gpu(num_samples=30000, batch_size=15000, max_rank=4):
    """
    Batched generation and rejection sampling of two-qutrit states using PyTorch.
    Optimized to generate lower-rank states (1 to max_rank) 
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Running state generation on: {device} with max rank {max_rank}")
    
    # Mathematical constants
    x_pure_prod = 1 / np.sqrt(2)
    x_pure_mixed = np.sqrt(3) / 2
    
    sampled_x, sampled_y, sampled_is_npt = [], [], []
    total_accepted = 0
    
    while total_accepted < num_samples:
        # 1. Generate Batched Random Density Matrices with Rank Constraint
        G = torch.randn(batch_size, 9, max_rank, dtype=torch.complex128, device=device) + \
            1j * torch.randn(batch_size, 9, max_rank, dtype=torch.complex128, device=device)
            
        ranks = torch.randint(1, max_rank + 1, (batch_size,), device=device)
        col_indices = torch.arange(max_rank, device=device).unsqueeze(0)
        rank_mask = (col_indices < ranks.unsqueeze(1)).unsqueeze(1) 
        
        G = G * rank_mask
        rho = G @ G.conj().transpose(-2, -1)
        
        # FORCE Hermiticity to prevent cuSOLVER crashes
        rho = (rho + rho.conj().transpose(-2, -1)) / 2.0
        
        # Trace normalization (clamped to prevent /0)
        traces = torch.clamp(torch.einsum('bii->b', rho).real, min=1e-12) 
        rho = rho / traces.view(-1, 1, 1) 
        
        # 2. Compute Global Purity P = Tr(rho^2)
        P = torch.clamp((rho * rho.conj()).sum(dim=(-2, -1)).real, min=1e-12)
        
        # 3. Partial Traces for Subsystems A and B
        rho_tensor = rho.view(-1, 3, 3, 3, 3)
        rho_A = torch.einsum('b i j k j -> b i k', rho_tensor)
        rho_B = torch.einsum('b i j i l -> b j l', rho_tensor)
        
        # Reduced purities
        p_A = (rho_A * rho_A.conj()).sum(dim=(-2, -1)).real
        p_B = (rho_B * rho_B.conj()).sum(dim=(-2, -1)).real
        
        # 4. Calculate Budgets and XY Coordinates
        B_L = torch.clamp(3 * (p_A + p_B) - 2, min=0.0)
        B_NL = torch.clamp((9 * P - 1) - B_L, min=0.0)
        
        X = torch.sqrt(B_L / (8 * P))
        Y = torch.sqrt(B_NL / (8 * P))
        
        # 5. Constraints Evaluation (Rejection Masks)
        mask_x_bounds = (X >= 0) & (X <= x_pure_mixed)
        
        y_max_pure = torch.sqrt(torch.clamp(1 - X**2, min=0.0))
        y_max_pos = torch.sqrt(torch.clamp(1.5 - 2*X**2, min=0.0))
        y_max = torch.where(X <= x_pure_prod, y_max_pure, y_max_pos)
        mask_upper_bound = Y <= y_max
        
        y_cc_bound = torch.sqrt(torch.clamp((1.5 - X**2)/2, min=0.0))
        mask_above_cc = Y > y_cc_bound
        
        valid_mask = mask_x_bounds & mask_upper_bound & mask_above_cc
        
        if not valid_mask.any():
            continue
            
        # 6. Batched NPT Check
        valid_rho_tensor = rho_tensor[valid_mask]
        
        # EXPLICIT .contiguous() is required here before reshaping for cuSOLVER!
        rho_pt_tensor = valid_rho_tensor.transpose(2, 4).contiguous()
        rho_pt = rho_pt_tensor.view(-1, 9, 9)
        
        # FORCE Hermiticity on the partial transpose
        rho_pt = (rho_pt + rho_pt.conj().transpose(-2, -1)) / 2.0
        
        # Now safe to calculate eigenvalues
        eigvals = torch.linalg.eigvalsh(rho_pt)
        is_npt_batch = torch.any(eigvals < -1e-8, dim=1)
        
        # 7. Store accepted samples
        sampled_x.append(X[valid_mask].cpu())
        sampled_y.append(Y[valid_mask].cpu())
        sampled_is_npt.append(is_npt_batch.cpu())
        
        total_accepted += valid_mask.sum().item()
        
    final_x = torch.cat(sampled_x)[:num_samples].numpy()
    final_y = torch.cat(sampled_y)[:num_samples].numpy()
    final_npt = torch.cat(sampled_is_npt)[:num_samples].numpy()
    
    final_colors = np.where(final_npt, "#274C77", "green")
    
    return final_x, final_y, final_colors
# =============================
#  Font & Style Configuration
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

# =============================
#  Physics / Coordinate Definitions
# =============================
X = np.linspace(0, 1.02, 2000)

Y_R1 = np.sqrt(np.maximum(0, 1 - X**2))
Y_pos = np.sqrt(np.maximum(0, 1.5 - 2*X**2))
Y_cc_raw = np.sqrt(np.maximum(0, (1.5 - X**2)/2))
Y_sep = np.sqrt(np.maximum(0, 0.125 - X**2))

x_pure_prod  = 1/np.sqrt(2)      
x_pure_mixed = np.sqrt(3)/2      

Y_feasible = np.zeros_like(X)
mask_pure = X <= x_pure_prod
mask_pos  = (X > x_pure_prod) & (X <= x_pure_mixed)

Y_feasible[mask_pure] = Y_R1[mask_pure]
Y_feasible[mask_pos]  = Y_pos[mask_pos]

Y_cc = np.minimum(Y_cc_raw, Y_feasible)

# =============================
#  Main Execution
# =============================
# Generate 30k valid points (runs fast on GPU!)
sampled_x, sampled_y, sampled_colors = generate_and_filter_states_gpu(num_samples=num_sample, batch_size=batchsize, max_rank=maxrank)

# =============================
#  Plotting
# =============================
col_Quantum    = "#5AA9E6"   # Light Blue
col_CC         = "#FFEBEB"   # Classical Pink
col_Unfeasible = "lightgray" # Unfeasible
col_Sep        = "#FFEE99"   # Separable Yellow
line_dark      = "#274C77"
line_purple    = "#975AE6"
line_pink      = "#FF6392" 

fig, ax = plt.subplots(figsize=(7, 7))

ax.fill_between(X, 0, Y_feasible, color=col_Quantum, zorder=1,alpha=0.45)
ax.fill_between(X, 0, Y_cc, color=col_CC, zorder=2,alpha=1)
ax.fill_between(X, 0, Y_sep, color=col_Sep, zorder=2,alpha=0.65)

mask_fill = (X > x_pure_prod) & (X <= 1.0)
ax.fill_between(X[mask_fill], Y_feasible[mask_fill], Y_R1[mask_fill], color=col_Unfeasible, zorder=5)

ax.plot(X[mask_pure], Y_R1[mask_pure], color=line_dark, linewidth=2.5, zorder=6)

mask_dotted = (X > x_pure_prod) & (X <= 1.0)
X_dot = np.append(X[mask_dotted], 1.0) if X[mask_dotted][-1] < 1.0 else X[mask_dotted]
Y_dot = np.append(Y_R1[mask_dotted], 0.0) if X[mask_dotted][-1] < 1.0 else Y_R1[mask_dotted]
ax.plot(X_dot, Y_dot, "--", color="#49616E", linewidth=2.5, zorder=6)

X_pos_plot = np.append(X[mask_pos], x_pure_mixed)
Y_pos_plot = np.append(Y_pos[mask_pos], 0.0)
ax.plot(X_pos_plot, Y_pos_plot, color="black", linewidth=2.5, zorder=7)

CC_line, = ax.plot(X[X <= x_pure_prod], Y_cc_raw[X <= x_pure_prod], color=line_purple, linewidth=2.5, zorder=8, label="CC/QC/CQ boundary")
Sep_line, = ax.plot(X[X <= x_pure_mixed], Y_sep[X <= x_pure_mixed], "--", color=line_pink, linewidth=2.5, zorder=8, label="Separable boundary")
ax.plot([0, x_pure_mixed], [0, 0], color="red", linewidth=1.5, zorder=9)

pt_bell = (0, 1)
pt_prod = (1/np.sqrt(2), 1/np.sqrt(2))
pt_mixed = (0, 0)
pt_onesided = (np.sqrt(3)/2, 0)

# =============================
#  Isolate and Count States
# =============================
# Create boolean masks based on the color arrays returned by the generator
mask_ppt = sampled_colors == "green"
mask_npt = sampled_colors == "#274C77"

x_ppt, y_ppt = sampled_x[mask_ppt], sampled_y[mask_ppt]
x_npt, y_npt = sampled_x[mask_npt], sampled_y[mask_npt]

count_ppt = np.sum(mask_ppt)
count_npt = np.sum(mask_npt)

print(f"Total NPT states generated: {count_npt}")
print(f"Total PPT states generated: {count_ppt}")
print(f"Percentage of PPT states: {(count_ppt / len(sampled_colors)) * 100:.2f}%")

# =============================
#  Plotting the Scatter Points
# =============================
# Scatter NPT states
ax.scatter(
    x_npt, y_npt, 
    color="#274C77", 
    s=1, 
    edgecolors='none', 
    linewidths=0.5, 
    alpha=0.8, 
    zorder=15,
    rasterized=True
)

# Scatter PPT states 
# (Slightly larger marker 's=3' and higher 'zorder=16' so they pop out)
ax.scatter(
    x_ppt, y_ppt, 
    color="green", 
    s=3, 
    edgecolors='none', 
    linewidths=0.5, 
    alpha=0.9, 
    zorder=16
)

ax.scatter(*pt_bell, s=60, color="#99CAF0", edgecolors="black", zorder=20, clip_on=False)
ax.scatter(*pt_prod, s=60, color="red", edgecolors="black", zorder=20, clip_on=False)
ax.scatter(*pt_mixed, s=60, color="#FFEE99", edgecolors="black", zorder=20, clip_on=False)
ax.scatter(*pt_onesided, s=60, color="orange", edgecolors="black", zorder=20, clip_on=False)

ax.set_xlabel(r"$X$", fontsize=20)
ax.set_ylabel(r"$Y$", fontsize=20)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_aspect("equal")
ax.tick_params(axis='both', which='major', labelsize=14)

red_marker = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='#274C77', markeredgecolor='black', markersize=7, label='NPT Entangled')
green_marker = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='green', markeredgecolor='black', markersize=7, label='PPT State')

# ax.legend(handles=[CC_line, Sep_line, red_marker], loc="upper right", frameon=False, fontsize=11)

ax.grid(alpha=0.25)
for spine in ax.spines.values():
    spine.set_linewidth(1.8)

plt.tight_layout()
plt.savefig("Fig_3E.pdf", dpi=300)
