import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# =============================
#  1. Style & Configuration
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
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Common Colors
col_Entangled  = "#7FC8F8"   
col_Bisep       = "#5AA9E6"   
col_Sep         = "#FFEE99"   
col_Sep_Bud     = "#FFF8C7"   
col_Sep_Outer   = "#FFEBEB"   
col_Gray        = "lightgray" 
line_blue       = "#274C77"      
line_purple     = "#975AE6"
line_red        = "#D62728" 
line_ghost      = "#49616E"

# =============================
# 2. Plotting Functions
# =============================

def plot_xy_phase_space():
    X = np.linspace(0, 1.05, 5000)
    X_bell, X_pp, X_cliff = np.sqrt(1/7), np.sqrt(3/7), np.sqrt(4/7)
    R_ball = np.sqrt(1/7)

    Y_purity_arc = np.sqrt(np.maximum(0, 1 - X**2))

    def get_physical_boundary(x_vals):
        y_vals = np.zeros_like(x_vals)
        mask_1 = x_vals <= X_pp
        y_vals[mask_1] = np.sqrt(np.maximum(0, 1 - x_vals[mask_1]**2))
        mask_2 = (x_vals > X_pp) & (x_vals <= X_cliff)
        y_vals[mask_2] = np.sqrt(np.maximum(0, 10/7 - 2*x_vals[mask_2]**2))
        return y_vals

    Y_physical = get_physical_boundary(X)
    Y_cqq_val = np.sqrt(6/7)
    Y_ccq = np.sqrt(np.maximum(0, (18 - 14*X**2) / 21))
    Y_ccq = np.minimum(Y_ccq, Y_purity_arc) 
    Y_ball = np.sqrt(np.maximum(0, R_ball**2 - X**2))

    fig, ax = plt.subplots(figsize=(7, 7))

    # --- FILLS ---
    mask_phys = X <= X_cliff
    ax.fill_between(X, Y_physical, Y_purity_arc, color=col_Gray, zorder=5)
    ax.fill_between(X, Y_cqq_val, Y_purity_arc, where=(X <= X_bell), color=col_Entangled, zorder=1, alpha=0.4)
    Y_top_bisep = np.minimum(Y_physical, Y_cqq_val)
    Y_bot_bisep = np.minimum(Y_physical, Y_ccq)
    ax.fill_between(X, Y_bot_bisep, Y_top_bisep, where=mask_phys, color=col_Bisep, zorder=1, alpha=0.45)
    ax.fill_between(X, 0, Y_bot_bisep, where=mask_phys, color=col_Sep_Outer, zorder=2, alpha=1)
    ax.fill_between(X, 0, Y_ball, where=(X <= R_ball), color=col_Sep, zorder=2, alpha=0.65)

    # --- CURVES ---
    ax.plot(X, Y_purity_arc, color=line_ghost, linestyle="--", linewidth=2.5, alpha=0.6, zorder=5)
    ax.plot(X[X <= X_pp], Y_physical[X <= X_pp], color=line_blue, linewidth=2.5, zorder=11)
    ax.plot(X[(X >= X_pp) & (X <= X_cliff)], Y_physical[(X >= X_pp) & (X <= X_cliff)], color="black", linewidth=2.5, zorder=10)
    ax.plot([X_cliff, X_cliff], [0, np.sqrt(2/7)], color="black", linewidth=2.5, zorder=10)
    ax.plot(X[X <= X_bell], np.full_like(X[X <= X_bell], Y_cqq_val), "-.", color="#274C77", linewidth=2.0, zorder=9, label="Q$^2$C boundary")
    ax.plot(X[X <= X_pp], Y_ccq[X <= X_pp], color=line_purple, linewidth=2.0, zorder=9, label="QC$^2$ boundary")
    ax.plot(X[X <= R_ball], Y_ball[X <= R_ball], color=line_red, linestyle="--", linewidth=2.0, zorder=15, label=" Biseparable boundary")
    ax.plot([R_ball, X_cliff], [0, 0], color=line_red, linestyle="--", linewidth=2.0, zorder=15)

    # --- POINTS & LABELS ---
    pts_x = [0.0, np.sqrt(1/21), X_bell, X_pp, X_cliff, X_cliff, 0]
    pts_y = [1.0, np.sqrt(20/21), Y_cqq_val, np.sqrt(4/7), np.sqrt(2/7), 0, 0]
    colors = ["#99CAF0", "#99CAF0", "#99CAF0", "red", "#9E7F67", "orange", "#FFEE99"]
    for px, py, c in zip(pts_x, pts_y, colors):
        ax.scatter(px, py, s=70, color=c, edgecolors="black", zorder=20, clip_on=False)

    ax.text(0.04, 1.01, "GHZ", fontsize=14, color="#1B71B3", ha='center', zorder=20)
    ax.text(0.25, 0.98, "W", fontsize=14, color="#1B71B3", ha='center', zorder=20)
    ax.text(0.51, 0.93, "Bell$\otimes$Pure states", fontsize=14, color="#1B71B3", ha='center', zorder=20)
    ax.text(0.78, 0.77, "Pure prod. states", fontsize=14, color="red", ha='center', zorder=20)
    ax.text(X_cliff - 0.2, 0.52, "Pure$\otimes$pure$\otimes$\nmixed states", fontsize=14, color="#9E7F67", ha='left', zorder=20)
    ax.text(X_cliff - 0.2, 0.02, "Pure$\otimes$mixed$\otimes$\nmixed states", fontsize=14, color="#CD6B1F", ha='left', zorder=20)
    ax.text(0.015, 0.015, "Max.\nmixed\nstate", fontsize=14, color="#49616E", zorder=20)
    ax.text(0.143, 0.94, "Multipartite entanglement", fontsize=11.5, color="#49616E", ha='center', zorder=20)
    ax.text(0.43, 0.835, "Biseparable region", fontsize=11, color="#49616E", ha='center', rotation=-21.4, zorder=20)
    ax.text(0.88, 0.13, "$Q<0$\nUnfeasible", fontsize=18, color="#49616E", ha='center', zorder=20)
    ax.text(0.15, 0.15, "Bi-separable", fontsize=18, color="#49616E", ha='center', zorder=20)
    ax.text(0.3, 0.81, "Classical envelope", fontsize=18, color="#49616E", ha='center', rotation=-13.4, zorder=20)

    ax.set_xlabel("$X$", fontsize=20); ax.set_ylabel("$Y$", fontsize=20)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False); ax.grid(alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_zorder(10)
    plt.tight_layout(); plt.savefig("Fig_3C.pdf", dpi=300)

def plot_bl_bnl_budget_space():
    BL = np.linspace(0, 3.5, 5000)
    BL_bell, BL_pp, BL_cliff_top, BL_cliff_bot = 1.0, 3.0, 2.0, 1.0
    BNL_purity = 7 - BL

    def get_physical_boundary(bl_vals):
        bnl_vals = np.zeros_like(bl_vals)
        mask_1 = (bl_vals > 2.0) & (bl_vals <= 3.0)
        bnl_vals[mask_1] = 3 * bl_vals[mask_1] - 5
        mask_2 = (bl_vals > 1.0) & (bl_vals <= 2.0)
        bnl_vals[mask_2] = bl_vals[mask_2] - 1
        mask_3 = (bl_vals >= 0) & (bl_vals <= 1.0)
        bnl_vals[mask_3] = 0
        bnl_vals[bl_vals > 3.0] = np.nan
        return bnl_vals

    BNL_physical = get_physical_boundary(BL)
    BNL_cqq = np.where(BL <= BL_bell, 3 * BL + 3, np.nan)
    BNL_ccq = (BL + 9) / 3
    BNL_ball = np.maximum(0, 1/7 - BL)

    fig, ax = plt.subplots(figsize=(7, 7))

    # --- FILLS ---
    mask_phys = BL <= BL_pp
    ax.fill_between(BL, 0, BNL_physical, where=mask_phys, color=col_Gray, zorder=2)
    ax.fill_between(BL, 0, 7.5, where=(BL > BL_pp), color=col_Gray, zorder=2)
    ax.fill_between(BL, BNL_cqq, BNL_purity, where=(BL <= BL_bell), color=col_Entangled, zorder=1, alpha=0.4)
    Y_top_bisep = np.minimum(BNL_purity, np.where(np.isnan(BNL_cqq), 10, BNL_cqq))
    ax.fill_between(BL, BNL_ccq, Y_top_bisep, where=mask_phys, color=col_Bisep, zorder=1, alpha=0.45)
    ax.fill_between(BL, BNL_physical, BNL_ccq, where=mask_phys, color=col_Sep_Outer, zorder=3, alpha=1)
    ax.fill_between(BL, 0, BNL_ball, where=(BL <= 1/7), color=col_Sep_Bud, zorder=2, alpha=0.65)

    # --- CURVES ---
    ax.plot(BL[mask_phys], BNL_purity[mask_phys], color=line_ghost, linewidth=2.5, zorder=11)
    ax.plot(BL[mask_phys], BNL_physical[mask_phys], color="black", linewidth=2.5, zorder=10)
    ax.plot(BL[BL <= BL_bell], BNL_cqq[BL <= BL_bell], "-.", color="#274C77", linewidth=2.0, zorder=9, label="Q$^2$C boundary")
    ax.plot(BL[mask_phys], BNL_ccq[mask_phys], color=line_purple, linewidth=2.0, zorder=9, label="QC$^2$ boundary")
    ax.plot(BL[BL <= 1/7], BNL_ball[BL <= 1/7], color=line_red, linestyle="--", linewidth=2.0, zorder=15, label=" Biseparable boundary")
    ax.plot([1/7, BL_cliff_bot], [0, 0], color=line_red, linestyle="--", linewidth=2.0, zorder=15)

    # --- POINTS & LABELS ---
    pts_x = [0.0, 1/3, BL_bell, BL_pp, BL_cliff_top, BL_cliff_bot, 0]
    pts_y = [7.0, 20/3, 6.0, 4.0, 1.0, 0.0, 0.0]
    colors = ["#99CAF0", "#99CAF0", "#99CAF0", "red", "#9E7F67", "orange", "#FFEE99"]
    for px, py, c in zip(pts_x, pts_y, colors):
        ax.scatter(px, py, s=70, color=c, edgecolors="black", zorder=20, clip_on=False)

    ax.text(0.1, 6.65, "GHZ", fontsize=13, color="#1B71B3", ha='center', zorder=20)
    ax.text(0.4, 6.65, "W", fontsize=14, color="#1B71B3", ha='center', zorder=20)
    ax.text(1.25, 6.0, "Bell$\otimes$Pure", fontsize=14, color="#1B71B3", ha='center', zorder=20)
    ax.text(2.5, 4.0, "Pure prod. states", fontsize=14, color="red", ha='center', zorder=20)
    ax.text(BL_cliff_top - 0.54, 0.95, "Pure$\otimes$pure$\otimes$\nmixed states", fontsize=14, color="#9E7F67", ha='left', zorder=20)
    ax.text(BL_cliff_bot - 0.34, 0.15, "Pure$\otimes$mixed$\otimes$\nmixed states", fontsize=14, color="#CD6B1F", ha='left', zorder=20)
    ax.text(0.02, 0.15, "Max.\nmixed\nstate", fontsize=14, color="#49616E", ha='left', zorder=20)
    ax.text(0.4, 5.5, "Multipartite \n entanglement", fontsize=18, color="#49616E", ha='center', zorder=20)
    ax.text(1.3, 4.3, "Biseparable region", fontsize=18, color="#49616E", ha='center', zorder=20)
    ax.text(2.55, 0.9, "$Q<0$\nUnfeasible", fontsize=18, color="#49616E", ha='center', zorder=20)
    ax.text(1.3, 3., "Classical envelope", fontsize=18, color="#49616E", ha='center', rotation=8.9, zorder=20)

    ax.set_xlabel("$B_\mathrm{L}$", fontsize=18); ax.set_ylabel("$B_\mathrm{NL}$", fontsize=18)
    ax.set_xlim(0, 3); ax.set_ylim(0, 7)
    ax.legend(loc="upper right", frameon=False); ax.grid(alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_zorder(10)
    plt.tight_layout(); plt.savefig("Fig_6C.pdf", dpi=300)

# =============================
# 3. Execution
# =============================
plot_xy_phase_space()
plot_bl_bnl_budget_space()
plt.show()
