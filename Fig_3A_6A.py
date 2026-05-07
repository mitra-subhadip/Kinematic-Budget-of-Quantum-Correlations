import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# =============================
#  2. Transformation Helper
# =============================
def xy_to_blbnl(x, y):
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    r2 = x**2 + y**2
    denom = 6.0 - 5.0 * r2
    denom[denom < 1e-9] = 1e-9 
    P = 1.0 / denom
    BL = 5.0 * P * x**2
    BNL = 5.0 * P * y**2
    return BL, BNL

# =============================
#  3. Plot: Qubit-Qutrit Phase Space
# =============================
def plot_qubit_qutrit_phase_space():
    X = np.linspace(0, 1.0, 5000)
    X_max_ent, X_prod = np.sqrt(0.1), np.sqrt(0.6)

    # Physics Curves
    Y_R1 = np.sqrt(np.maximum(0, 1 - X**2))
    w = np.linspace(2/3, 1, 500); w2 = w**2
    P_qq = (3*w2 - 2*w + 1) / 2.0
    BL_qq, BNL_qq = 4.5*w2 - 6*w + 2, 4.5*w2
    X_qq, Y_qq = np.sqrt(BL_qq / (5 * P_qq)), np.sqrt(BNL_qq / (5 * P_qq))

    def get_entanglement_wedge_interp(x_vals):
        y_interp = np.interp(x_vals, X_qq, Y_qq, left=0, right=0)
        y_interp[x_vals > X_max_ent] = 0
        return y_interp
    Y_ent_fill = get_entanglement_wedge_interp(X)

    # QC Envelope Piecewise
    BL_qc1, BL_qc2 = np.linspace(0, 0.5, 500), np.linspace(0.5, 3.0, 500)
    BNL_qc1, BNL_qc2 = 1 + BL_qc1, 1.5 + 0.2 * (BL_qc2 - 0.5)
    P_qc1, P_qc2 = (BL_qc1 + BNL_qc1 + 1) / 6.0, (BL_qc2 + BNL_qc2 + 1) / 6.0
    X_qc_true = np.concatenate([np.sqrt(BL_qc1/(5*P_qc1)), np.sqrt(BL_qc2/(5*P_qc2))])
    Y_qc_true = np.concatenate([np.sqrt(BNL_qc1/(5*P_qc1)), np.sqrt(BNL_qc2/(5*P_qc2))])
    Y_qc = np.interp(X, X_qc_true, Y_qc_true, left=np.nan, right=0)
    Y_qc[X > X_prod] = 0

    # Root Activation
    BL_root = np.linspace(0, 0.125, 300)
    BNL_root = 2/3 + (11 * np.sqrt(2) / 12) * np.sqrt(BL_root)
    P_root = (BL_root + BNL_root + 1) / 6.0
    X_root, Y_root = np.sqrt(BL_root / (5 * P_root)), np.sqrt(BNL_root / (5 * P_root))
    
    Y_qcurve_raw = np.sqrt(np.maximum(0, 1.6 - 2*X**2))
    Y_sep_ball = np.sqrt(np.maximum(0, 0.2 - X**2))

    fig, ax = plt.subplots(figsize=(7, 7))
    col_UPunfeas, col_CC, col_Blue, col_Gray, col_Sep = "#B8A597", "#FFEBEB", "#5AA9E6", "lightgray", "#FFEE99"
    line_purple, line_red = "#975AE6", "#FF6392"

    # Fills
    ax.fill_between(X, 0, Y_qc, where=(X <= X_prod), color=col_CC, zorder=3)
    ax.fill_between(X, 0, Y_qcurve_raw, where=(X > X_prod), color=col_CC, zorder=3)
    ax.fill_between(X, 0, Y_sep_ball, where=(X <= np.sqrt(0.2)), color=col_Sep, zorder=8, alpha=0.65)
    Y_upper_combined = np.where(X <= X_max_ent, Y_ent_fill, Y_R1)
    ax.fill_between(X, Y_qc, Y_upper_combined, where=(X <= X_prod), alpha=0.45, color=col_Blue, zorder=2, lw=0)
    ax.fill_between(X, Y_ent_fill, Y_R1, where=(X <= X_max_ent), color=col_UPunfeas, zorder=2)
    ax.fill_between(X, Y_qcurve_raw, Y_R1, where=(X > X_prod), color=col_Gray, zorder=2)
    
    # Root Activation Fill
    X_fill_root = np.linspace(0, X_root[-1], 300)
    ax.fill_between(X_fill_root, np.interp(X_fill_root, X_root, Y_root), np.interp(X_fill_root, X_qc_true, Y_qc_true), color="#F38989", alpha=0.6, zorder=4)

    # Plot Lines
    ax.plot(X[X < X_max_ent], Y_R1[X < X_max_ent], color="#49616E", lw=2.5, ls='--', zorder=22)
    ax.plot(X[(X >= X_max_ent) & (X <= X_prod)], Y_R1[(X >= X_max_ent) & (X <= X_prod)], color="#274C77", lw=2.5, zorder=22)
    ax.plot(X[X > X_prod], Y_R1[X > X_prod], color="#49616E", lw=2.5, ls='--', zorder=22)
    ax.plot(X_qq, Y_qq, color="black", lw=2.0, zorder=9)
    ax.plot(X[X <= X_prod], Y_qc[X <= X_prod], color=line_purple, lw=2.5, zorder=8, label="QC boundary")
    ax.plot(X_root, Y_root, color="deeppink", lw=2.5, zorder=12, label="C boundary")
    ax.plot(X[X > X_prod], Y_qcurve_raw[X > X_prod], color="black", lw=2.0, zorder=8)
    ax.plot(X[X <= np.sqrt(0.2)], Y_sep_ball[X <= np.sqrt(0.2)], color=line_red, lw=2.0, ls='--', zorder=15, label="Separable boundary")
    ax.plot([0, np.sqrt(0.8)], [0, 0], color="red", lw=1.5, ls='--', zorder=11)

    # --- Points: Using clip_on=False to prevent axis clipping ---
    ax.scatter(X_qq[-1], Y_qq[-1], s=60, color="#99CAF0", edgecolors="black", zorder=30, clip_on=False)
    ax.text(X_qq[-1]+0.02, Y_qq[-1]+0.01, "Max. pure entangled states", fontsize=13, color="#1B71B3")
    ax.scatter(X_qq[0], Y_qq[0], s=60, color="#5CD68F", edgecolors="black", zorder=20, clip_on=False)
    ax.text(0.02, Y_qq[0], "Max. mixed\nentangled states", fontsize=13, color="#0E6333", va='top')
    ax.scatter(np.sqrt(0.6), np.sqrt(0.4), s=60, color="red", edgecolors="black", zorder=30, clip_on=False)
    ax.text(np.sqrt(0.6)+0.02, np.sqrt(0.4), "Pure prod. states", fontsize=13, color="red", va='center')
    ax.scatter(X_root[-1], Y_root[-1], s=60, color="deeppink", edgecolors="black", zorder=30, clip_on=False)
    ax.scatter(np.sqrt(0.6), 0, s=60, color="#BD7F55", edgecolors="black", zorder=20, clip_on=False, label="Pure qubit $\otimes$ max-mixed ")
    ax.scatter(np.sqrt(0.8), 0, s=60, color="orange", edgecolors="black", zorder=20, clip_on=False, label="Max-mixed $\otimes$ pure qutrit ")
    ax.scatter(0, 0, s=60, color="#FFEE99", edgecolors="black", zorder=20, clip_on=False)
    ax.text(0.015, 0.015, "Max.\nmixed\nstate", fontsize=16, color="#49616E", zorder=20)

    # Labels
    ax.text(0.30, 0.80, "Guaranteed entanglement", fontsize=18, color="#49616E", ha='center', zorder=20)
    ax.text(0.37, 0.73, "QC envelope", fontsize=16, color="#49616E", ha='center', zorder=20)
    ax.text(0.18, 0.15, "Separable", fontsize=18, color="#49616E", ha='center', zorder=20)
    ax.text(0.1, 0.95, "Unfeasible", fontsize=16, color="#49616E", ha='center', zorder=200)
    ax.text(0.94, 0.13, "$Q<0$", fontsize=16, color="#49616E", ha='center', zorder=20)
    ax.text(0.8, 0.26, r"$10X^2+5Y^2= 8$", fontsize=16, color="#49616E", rotation=-78, ha='center', zorder=20)
    ax.text(0.1, 0.66, "C envelope", fontsize=16, color="#49616E", rotation=23, ha='center', zorder=20)

    ax.set_xlabel(r"$X$", fontsize=20); ax.set_ylabel(r"$Y$", fontsize=20)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.grid(alpha=0.25)
    for spine in ax.spines.values(): spine.set_linewidth(1.8)
    plt.tight_layout(); plt.savefig("Fig_3A.pdf", dpi=300)

# =============================
#  4. Plot: Qubit-Qutrit Budget Space
# =============================
def plot_qubit_qutrit_budget_space():
    X = np.linspace(0, 1.0, 5000)
    X_max_ent, X_prod = np.sqrt(0.1), np.sqrt(0.6)
    Y_R1 = np.sqrt(np.maximum(0, 1 - X**2))
    w = np.linspace(2/3, 1, 500); w2 = w**2
    P_qq = (3*w2 - 2*w + 1) / 2.0
    BL_qq_param, BNL_qq_param = 4.5*w2 - 6*w + 2, 4.5*w2
    X_qq, Y_qq = np.sqrt(BL_qq_param / (5 * P_qq)), np.sqrt(BNL_qq_param / (5 * P_qq))
    Y_ent_fill = np.interp(X, X_qq, Y_qq, left=0, right=0); Y_ent_fill[X > X_max_ent] = 0
    Y_qcurve_raw = np.sqrt(np.maximum(0, 1.6 - 2*X**2))
    Y_sep_ball = np.sqrt(np.maximum(0, 0.2 - X**2))

    def create_transformed_polygon(x_arr, y_bottom, y_top, mask):
        xm, yb, yt = x_arr[mask], y_bottom[mask], y_top[mask]
        x_poly = np.concatenate([xm, xm[::-1]])
        y_poly = np.concatenate([yt, yb[::-1]])
        return xy_to_blbnl(x_poly, y_poly)

    QC_BL_Exact, QC_BNL_Exact = [0, 0.5, 3.0], [1.0, 1.5, 2.0]
    bl_cc_grid = np.linspace(0, 3, 500)
    bnl_cc_top = np.interp(bl_cc_grid, QC_BL_Exact, QC_BNL_Exact)
    
    bl_cc = np.concatenate([bl_cc_grid, bl_cc_grid[::-1]])
    bnl_cc = np.concatenate([bnl_cc_top, np.zeros_like(bl_cc_grid)])

    mask_ent_total = (X <= np.sqrt(0.6))
    bl_ent_top, bnl_ent_top = xy_to_blbnl(X[mask_ent_total], np.where(X <= np.sqrt(0.1), Y_ent_fill, Y_R1)[mask_ent_total])
    bnl_ent_bot = np.interp(bl_ent_top, QC_BL_Exact, QC_BNL_Exact)
    bl_ent_merged = np.concatenate([bl_ent_top, bl_ent_top[::-1]])
    bnl_ent_merged = np.concatenate([bnl_ent_top, bnl_ent_bot[::-1]])

    bl_sep, bnl_sep = create_transformed_polygon(X, np.zeros_like(X), Y_sep_ball, (X <= np.sqrt(0.2)))
    bl_gray_1, bnl_gray_1 = create_transformed_polygon(X, Y_ent_fill, Y_R1, (X <= X_max_ent))
    bl_gray_2, bnl_gray_2 = create_transformed_polygon(X, Y_qcurve_raw, Y_R1, (X > X_prod))
    bl_cc_tail, bnl_cc_tail = create_transformed_polygon(X, np.zeros_like(X), Y_qcurve_raw, (X > X_prod))
    
    bl_root = np.linspace(0, 0.125, 300)
    bnl_root = 2/3 + (11 * np.sqrt(2) / 12) * np.sqrt(bl_root)
    bnl_root_top = np.interp(bl_root, QC_BL_Exact, QC_BNL_Exact)

    fig, ax = plt.subplots(figsize=(8, 8))
    col_Blue, col_Sep, col_Gray, col_CC, col_UPunfeas = "#5AA9E6", "#FFEE99", "lightgray", "#FFEBEB", "#B8A597"
    line_purple, line_red = "#975AE6", "#D62728"

    ax.fill(bl_cc, bnl_cc, color=col_CC, zorder=3); ax.fill(bl_cc_tail, bnl_cc_tail, color=col_CC, zorder=3)
    ax.fill(bl_sep, bnl_sep, color=col_Sep, zorder=8, alpha=0.65)
    ax.fill(bl_ent_merged, bnl_ent_merged, color=col_Blue, alpha=0.45, zorder=1)
    ax.fill(bl_gray_1, bnl_gray_1, color=col_UPunfeas, zorder=6); ax.fill(bl_gray_2, bnl_gray_2, color=col_Gray, zorder=6)
    ax.fill_between(bl_root, bnl_root, bnl_root_top, color="#F38989", alpha=0.6, zorder=4)

    def plot_tx(x, y, **kwargs):
        bl, bnl = xy_to_blbnl(x, y); ax.plot(bl, bnl, **kwargs)

    # Boundaries
    plot_tx(X[X < X_max_ent], Y_R1[X < X_max_ent], color="#49616E", lw=2.5, ls='--', zorder=10)
    plot_tx(X[(X >= X_max_ent) & (X <= X_prod)], Y_R1[(X >= X_max_ent) & (X <= X_prod)], color="#274C77", lw=2.5, zorder=10)
    plot_tx(X[X > X_prod], Y_R1[X > X_prod], color="#49616E", lw=2.5, ls='--', zorder=10)
    plot_tx(X_qq, Y_qq, color="black", lw=2.0, zorder=9)
    ax.plot(QC_BL_Exact, QC_BNL_Exact, color=line_purple, lw=2.5, zorder=8, label="QC boundary")
    ax.plot(bl_root, bnl_root, color="deeppink", lw=2.5, zorder=12, label="C boundary")
    plot_tx(X[X > X_prod], Y_qcurve_raw[X > X_prod], color="black", lw=2.0, zorder=8)
    plot_tx(X[X <= np.sqrt(0.2)], Y_sep_ball[X <= np.sqrt(0.2)], color=line_red, lw=2.0, ls='--', zorder=15, label="Separable boundary")
    plot_tx(np.linspace(0, np.sqrt(0.8), 100), np.zeros(100), color="red", lw=1.5, ls='--', zorder=11)

    # --- Points: Added clip_on=False to all scatter calls ---
    def sc_tx(x, y, **kwargs):
        bl, bnl = xy_to_blbnl(x, y); ax.scatter(bl, bnl, clip_on=False, **kwargs)
    
    sc_tx(X_qq[-1], Y_qq[-1], s=60, color="#99CAF0", edgecolors="black", zorder=2000)
    ax.text(0.5, 4.57, "Max. pure entangled states", fontsize=16, color="#1B71B3", zorder=2000)
    sc_tx(X_qq[0], Y_qq[0], s=60, color="#5CD68F", edgecolors="black", zorder=2000)
    ax.text(0.02, 2.01, "Max. mixed\nentangled states", fontsize=16, color="#0E6333", va='bottom', zorder=2000)
    sc_tx(np.sqrt(0.6), np.sqrt(0.4), s=60, color="red", edgecolors="black", zorder=2000)
    sc_tx(np.sqrt(0.6), 0, s=60, color="#BD7F55", edgecolors="black", zorder=2000, label=r"Pure qubit $\otimes$ Max-mixed")
    sc_tx(np.sqrt(0.8), 0, s=60, color="orange", edgecolors="black", zorder=2000, label=r"Max-mixed $\otimes$ Pure qutrit")
    sc_tx(0, 0, s=60, color="#FFEE99", edgecolors="black", zorder=2000)
    ax.scatter(0.125, 1.125, s=60, color="deeppink", edgecolors="black", zorder=2000, clip_on=False)

    # Text
    ax.text(0.015, 0.02, "Max.\nmixed\nstate", fontsize=16, color="#49616E", zorder=2000)
    ax.text(2.60, 2, "Pure prod. states", fontsize=16, color="red", ha='center', zorder=2000)
    ax.text(1, 3, "Guaranteed entanglement", fontsize=18, color="#49616E", ha='center', zorder=2000)
    ax.text(2.7, 0.5, "$Q<0$ \n Unfeasible", fontsize=18, color="#49616E", ha='center', zorder=2000)
    ax.text(1.2, 1.42, "QC envelope", fontsize=18, color="#49616E", rotation=8, ha='center', zorder=2000)
    ax.text(0.18, 4.15, "Unfeasible", fontsize=14, color="#49616E", ha='center', zorder=2000)
    ax.text(0.12, .61, "C envelope", fontsize=12, color="#49616E", rotation=70, ha='center', zorder=2000)

    ax.set_xlabel(r"$B_\mathrm{L}$", fontsize=20); ax.set_ylabel(r"$B_\mathrm{NL}$", fontsize=20)
    ax.set_xlim(0, 3); ax.set_ylim(0, 5.0)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    ax.grid(alpha=0.25)
    for spine in ax.spines.values(): spine.set_linewidth(1.8)
    plt.tight_layout(); plt.savefig("Fig_6A.pdf", dpi=300)

# Run Both
plot_qubit_qutrit_phase_space()
plot_qubit_qutrit_budget_space()
plt.show()
