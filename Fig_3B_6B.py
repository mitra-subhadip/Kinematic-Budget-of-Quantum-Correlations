import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

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

# Shared Colors
col_Quantum    = "#5AA9E6"   # Light Blue
col_CC         = "#FFEBEB"   # Classical Pink
col_Unfeasible = "lightgray" # Unfeasible
col_Sep        = "#FFEE99"   # Separable Yellow (Phase Space)
col_Sep_Bud    = "#FFF8C7"   # Separable Yellow (Budget Space)
line_dark      = "#274C77"
line_purple    = "#975AE6"
line_pink      = "#FF6392" 

def plot_phase_space():
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

    fig, ax = plt.subplots(figsize=(7, 7))

    # --- 1. Fill Regions ---
    ax.fill_between(X, 0, Y_feasible, alpha=0.45, color=col_Quantum, zorder=2)
    ax.fill_between(X, 0, Y_cc, alpha=1, color=col_CC, zorder=3)
    ax.fill_between(X, 0, Y_sep, alpha=0.65, color=col_Sep, zorder=8)

    mask_fill = (X > x_pure_prod) & (X <= 1.0)
    ax.fill_between(X[mask_fill], Y_feasible[mask_fill], Y_R1[mask_fill], color=col_Unfeasible, zorder=20)

    # --- 2. Boundaries & Lines ---
    ax.plot(X[mask_pure], Y_R1[mask_pure], color=line_dark, linewidth=2.5, zorder=22)
    mask_dotted = (X > x_pure_prod) & (X <= 1.0)
    X_dot, Y_dot = X[mask_dotted], Y_R1[mask_dotted]
    if len(X_dot) > 0 and X_dot[-1] < 1.0:
        X_dot = np.append(X_dot, 1.0); Y_dot = np.append(Y_dot, 0.0)
    ax.plot(X_dot, Y_dot, "--", color="#49616E", linewidth=2.5, zorder=22)

    X_pos_plot = np.append(X[mask_pos], x_pure_mixed)
    Y_pos_plot = np.append(Y_pos[mask_pos], 0.0)
    ax.plot(X_pos_plot, Y_pos_plot, color="black", linewidth=2.5, zorder=27)

    CC_line, = ax.plot(X[X <= x_pure_prod], Y_cc_raw[X <= x_pure_prod], color=line_purple, linewidth=2.5, zorder=8, label="QC boundary")
    Sep_line, = ax.plot(X[X <= x_pure_mixed], Y_sep[X <= x_pure_mixed], "--", color=line_pink, linewidth=2.5, zorder=8, label="Separable boundary")
    ax.plot([0, x_pure_mixed], [0, 0], color="red", linewidth=1.5, zorder=9)

    # --- 3. Points & Text ---
    ax.scatter(0, 1, s=60, color="#99CAF0", edgecolors="black", zorder=30, clip_on=False)
    ax.text(0.02, 1.01, "Max. entangled states", fontsize=16, color="#1B71B3", va='bottom')
    ax.scatter(1/np.sqrt(2), 1/np.sqrt(2), s=60, color="red", edgecolors="black", zorder=30, clip_on=False)
    ax.text(1/np.sqrt(2)+0.01, 1/np.sqrt(2)+0.01, "Pure prod. states", fontsize=16, color="red", va='bottom')
    ax.scatter(0, 0, s=60, color="#FFEE99", edgecolors="black", zorder=20, clip_on=False)
    ax.text(0.015, 0.015, "Max.\nmixed\nstate", fontsize=14, color="#49616E", zorder=20)
    ax.scatter(np.sqrt(3)/2, 0, s=60, color="orange", edgecolors="black", zorder=50, clip_on=False)
    ax.text(np.sqrt(3)/2-0.12, 0.013, "Pure $\otimes$ max.\n mixed states", fontsize=14, color="#CD6B1F", ha='center', zorder=20)

    ax.text(0.225, 0.865, "Guaranteed entanglement", fontsize=18, color="#49616E", rotation=-1., ha='center')
    ax.text(0.30, 0.76, "Classical envelope", fontsize=18, color="#49616E", rotation=-12, ha='center')
    ax.text(0.92, 0.14, "$Q<0$", fontsize=16, color="#49616E", ha='center',zorder = 20)
    ax.text(0.93, 0.1, "Unfeasible", fontsize=12.3, color="#49616E", ha='center',zorder = 20)
    ax.text(0.15, 0.15, "Separable", fontsize=18, color="#49616E", ha='center',zorder = 100)
    ax.text(0.77, 0.32, "$4X^2+2Y^2=3$", fontsize=18, color="gray", rotation=-76, ha='center', zorder=40)

    ax.set_xlabel("$X$", fontsize=20)
    ax.set_ylabel("$Y$", fontsize=20)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_aspect("equal")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(handles=[CC_line, Sep_line], loc="upper right", frameon=False, fontsize=13)
    ax.grid(alpha=0.25)
    for spine in ax.spines.values(): spine.set_linewidth(1.8)
    
    plt.tight_layout()
    plt.savefig("Fig_3B.pdf", dpi=300)

def plot_budget_space():
    # =============================
    #  Physics / Coordinate Definitions
    # =============================
    BL = np.linspace(0, 8.5, 2000)
    BNL_pure_line = np.maximum(0, 8 - BL)
    BNL_upper = np.where(BL <= 4, 8 - BL, 0)
    BNL_lower = np.maximum(0, 2 * BL - 4)
    mask_poly = (BL >= 0) & (BL <= 4)
    
    BNL_cc_raw = (BL + 4) / 2
    BNL_cc = np.minimum(BNL_upper, BNL_cc_raw)
    BNL_cc = np.maximum(BNL_cc, BNL_lower) 
    BNL_cc = np.where(mask_poly, BNL_cc, 0)

    BNL_sep_raw = 0.125 - BL
    BNL_sep = np.minimum(BNL_upper, BNL_sep_raw)
    BNL_sep = np.maximum(BNL_sep, BNL_lower)
    BNL_sep = np.where(BL <= 0.125, BNL_sep, 0)

    fig, ax = plt.subplots(figsize=(7, 7))

    # --- 1. Fill Regions ---
    ax.fill_between(BL[mask_poly], BNL_lower[mask_poly], BNL_upper[mask_poly], alpha=0.45, color=col_Quantum, zorder=1)
    ax.fill_between(BL[mask_poly], BNL_lower[mask_poly], BNL_cc[mask_poly], color=col_CC, zorder=3, alpha=1)
    ax.fill_between(BL[BL <= 0.125], 0, BNL_sep[BL <= 0.125], color=col_Sep_Bud, zorder=2, alpha=0.65)
    
    mask_unfeas_right = (BL > 4) & (BL <= 8)
    ax.fill_between(BL[mask_unfeas_right], 0, BNL_pure_line[mask_unfeas_right], color=col_Unfeasible, zorder=5)
    mask_unfeas_bottom = (BL >= 2) & (BL <= 4)
    ax.fill_between(BL[mask_unfeas_bottom], 0, BNL_lower[mask_unfeas_bottom], color=col_Unfeasible, zorder=5)

    # --- 2. Boundaries & Lines ---
    ax.plot(BL[mask_poly], BNL_upper[mask_poly], color=line_dark, linewidth=2, zorder=6)
    ax.plot(BL[mask_unfeas_right], BNL_pure_line[mask_unfeas_right], "--", color=line_dark, linewidth=2, zorder=6)
    ax.plot(BL[(BL >= 2) & (BL <= 4)], BNL_lower[(BL >= 2) & (BL <= 4)], color="black", linewidth=2.5, zorder=7)
    ax.plot([0, 2], [0, 0], color="black", linewidth=2.5, zorder=7)

    ax.plot(BL[BL <= 4], BNL_cc_raw[BL <= 4], color=line_purple, linewidth=2.5, zorder=8)
    ax.plot(BL[BL <= 0.125], BNL_sep_raw[BL <= 0.125], "--", color=line_pink, linewidth=2.5, zorder=8)
    ax.plot([0, 2], [0, 0], "--", color="red", linewidth=1.5, zorder=9)

    # --- 3. Points & Text ---
    ax.scatter(0, 8, s=60, color="#99CAF0", edgecolors="black", zorder=20, clip_on=False)
    ax.scatter(4, 4, s=60, color="red", edgecolors="black", zorder=20, clip_on=False)
    ax.scatter(0, 0, s=60, color="#FFEE99", edgecolors="black", zorder=20, clip_on=False)
    ax.scatter(2, 0, s=60, color="orange", edgecolors="black", zorder=20, clip_on=False)

    ax.text(0.24, 7.7, "Max. entangled states", fontsize=16, color="#1B71B3", va='bottom')
    ax.text(2.9, 3.9, "Pure prod. states", fontsize=16, color="red", va='bottom')
    ax.text(0.025, 0.045, "Max. \nmixed\nstate", fontsize=16, color="#49616E", zorder=20)
    ax.text(1.61, 0.05, "Pure $\otimes$ max.\nmixed states", fontsize=16, color="#CD6B1F", ha='center', zorder=20)
    ax.text(1.5, 4.5, "Guaranteed entanglement", fontsize=18, color="#49616E", rotation=0, ha='center',zorder = 200)
    ax.text(3.5, .8, "$Q<0$ \nUnfeasible", fontsize=18, color="#49616E", rotation=0, ha='center', zorder=200)
    ax.text(1.6, 2.2, "Classical envelope", fontsize=18, color="#49616E", rotation=15, ha='center', zorder=200)

    ax.set_xlabel(r"$B_\mathrm{L}$", fontsize=20); ax.set_ylabel(r"$B_\mathrm{NL}$", fontsize=20)
    ax.set_xlim(0, 4); ax.set_ylim(0, 8); ax.tick_params(axis='both', which='major', labelsize=14)

    ax.legend(handles=[plt.Line2D([],[],color=line_purple, linewidth=2.5), 
                       plt.Line2D([],[],color=line_pink, linewidth=2.5, linestyle="--")], 
              labels=["QC boundary", "Separable boundary"], loc="upper right", frameon=False, fontsize=13)

    ax.grid(alpha=0.25)
    for spine in ax.spines.values(): spine.set_linewidth(1.8)

    plt.tight_layout()
    plt.savefig("Fig_6B.pdf", dpi=300)

# Run both
plot_phase_space()
plot_budget_space()
plt.show()
