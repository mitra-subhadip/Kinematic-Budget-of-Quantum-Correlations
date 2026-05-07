import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import interp1d
import os

# =============================
#  Font configuration
# =============================
font_path = "./times.ttf"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_family = "Times New Roman"
else:
    font_family = "DejaVu Serif"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.8,
})

# =============================
#  Construct X,Y curves
# =============================
X = np.linspace(0, 1, 2000)

# Purity boundary R = 1
Y_R1 = np.sqrt(np.maximum(0, 1 - X**2))

# PPT (separable) boundary R = 1/3
Y_R13 = np.sqrt(np.maximum(0, 1/3 - X**2))

# CC / QC / CQ boundary
x_max_cc = np.sqrt(2/3)
mask_cc = X <= x_max_cc
X_cc = X[mask_cc]
Y_cc = np.sqrt(2/3 - 0.5 * X_cc**2)
Y_cc_interp = interp1d(X_cc, Y_cc, bounds_error=False, fill_value=0)(X)

# CHSH guaranteed ellipse
Y_chsh_raw = np.sqrt(np.maximum(0, (4 - 3*X**2)/5))
Y_chsh = np.minimum(Y_chsh_raw, Y_R1)

# =============================
#  Correct feasibility boundary X_max(Y)
# =============================
Y = np.linspace(0, 1, 2000)

X_max = np.zeros_like(Y)
mask_lowY  = Y**2 < 1/3
mask_highY = ~mask_lowY

# Theorem 1 upper boundary
X_max[mask_lowY]  = np.sqrt(2/3)
X_max[mask_highY] = np.sqrt(1 - Y[mask_highY]**2)

# Maximum possible X inside purity circle
X_R1 = np.sqrt(np.maximum(0, 1 - Y**2))

# =============================
#  Colours
# =============================
col_R1         = "#5AA9E6"   # CHSH guaranteed
col_CHSH_below = "#7FC8F8"   # Below CHSH
col_CC         = "#FFEBEB"   # Classical region
col_R1_outer   = "#FFEE99"   # Discord wedge
orange         = "#ff9900"
ppt_color      = "#FF6392"

# =============================
#  Begin plotting
# =============================
fig, ax = plt.subplots(figsize=(7, 7))

# Entire purity disk
ax.fill_between(X, 0, Y_R1, alpha=0.4, color=col_R1, zorder=1)

# Below CHSH boundary
ax.fill_between(X, 0, Y_chsh, alpha=0.45, color=col_CHSH_below, zorder=2)

# Classical region
ax.fill_between(X, 0, Y_cc_interp, alpha=1, color=col_CC, zorder=3)

# CHSH ellipse boundary
CHSH_line, = ax.plot(X, Y_chsh, "-.", color="#274C77", linewidth=2,
                     label="CHSH/steerable boundary", zorder=6)

# CC / QC / CQ boundary
CC_line_top, = ax.plot(X_cc, Y_cc, color="#975AE6", linewidth=2.5,
                       label="C/QC boundary", zorder=7)
ax.plot([x_max_cc, x_max_cc], [0, Y_cc[-1]], color="#975AE6", linewidth=2.5, zorder=7)

# PPT region
ax.fill_between(X, 0, Y_R13, alpha=0.65, color="#FFEE99", zorder=8) ##F9F9F9

# PPT boundary
PPT_line, = ax.plot(X, Y_R13, "--", color=ppt_color, linewidth=2.5,
                    label="Separable boundary", zorder=9)

# horizontal PPT extension for visibility
epsilon = 0.004
x0 = np.sqrt(1/3)
ax.plot(np.linspace(x0,1,400), epsilon*np.ones(400),
        "--", color=ppt_color, linewidth=2.5, zorder=9)

# ============================================
#  *** UNFEASIBLE REGION: X > X_max(Y) ***
# ============================================
ax.fill_betweenx(
    Y,
    X_max,
    X_R1,
    where=(X_R1 > np.sqrt(2.0/3)),
    color="lightgray",
    zorder=20,
)

# Boundary outline
ax.plot(X_max, Y, color="black", linewidth=2.5, zorder=21)  ##CD6B1F Orange

# Restrict purity boundary to X ≤ sqrt(2/3)
mask_purity = X <= x_max_cc
MAX_Xline, = ax.plot(X[mask_purity], Y_R1[mask_purity],
        color="#274C77", linewidth=2.5, zorder=22)

# Grey continuation of the purity boundary for X > sqrt(2/3)
mask_purity_gray = X > x_max_cc
ax.plot(
    X[mask_purity_gray], Y_R1[mask_purity_gray],"--",
    color="#49616E",
    linewidth=2.5,
    zorder=22
)

# ====================================
# Radial arrow from r=1/sqrt(3) to r=1
# ====================================
x_outer = np.sqrt(2/3)-0.005
y_outer = np.sqrt(1 - (x_outer+0.005)**2)-0.005
theta = np.arctan2(y_outer+0.005, x_outer+0.005)
ux, uy = np.cos(theta), np.sin(theta)

r_inner = 1/np.sqrt(3)
x_inner, y_inner = r_inner * ux, r_inner * uy

ax.annotate(
    "",
    xy=(x_outer, y_outer),
    xytext=(x_inner, y_inner),
    arrowprops=dict(arrowstyle="->", color=ppt_color, linewidth=2.0),
    zorder=40
)

ax.text(
    x_inner + 0.02*ux,
    y_inner + 0.02*uy + 0.02,
    r"Entanglement possible   ",
    fontsize=16, color="#49616E",
    rotation=np.degrees(theta),
    rotation_mode='anchor',
    va='center',
    zorder=40
)

x_pt = np.sqrt(2/3)
y_pt = np.sqrt(1/3)

# =============================
#  Region labels
# =============================
ax.text(0.040, 0.930, "CHSH", fontsize=17, color="#49616E")
ax.text(0.220, 0.800, "Steerable", fontsize=17, color="#49616E", rotation=-10)
ax.text(0.400, 0.680, "Classical", fontsize=17, color="#49616E", rotation=-19)
ax.text(0.550, 0.620, "envelope", fontsize=17, color="#49616E", rotation=-22)
ax.text(0.200, 0.200, "Separable", fontsize=17, color="#49616E",zorder=100)
ax.text(0.850, 0.165, "$Q<0$", fontsize=17, color="#49616E", zorder=100)
ax.text(0.840, 0.130, "Unfeasible", fontsize=12, color="#49616E", zorder=100)
ax.text(0.760, 0.250, "$X=\sqrt{2/3}$", fontsize=12, color="#49616E", rotation=-90)
ax.text(0.830, 0.570, "Pure prod.\n states", fontsize=14, color="red")
ax.text(0.005, 1.020, "Bell states", fontsize=14, color="#1B71B3")
ax.text(0.825, 0.010, "Pure $\otimes$ max.\nmixed states", fontsize=13, color="#CD6B1F", zorder=100)
ax.text(0.010, 0.010, "Max.\nmixed\nstate", fontsize=14, color="#49616E", zorder=100)

# =============================
#  Axes & Legend
# =============================
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_xlabel("$X$", fontsize=20)
ax.set_ylabel("$Y$", fontsize=20)
ax.tick_params(labelsize=16)
ax.grid(alpha=0.25)

ax.legend(
    handles=[CHSH_line, CC_line_top, PPT_line],
    loc="upper right",
    fontsize=14,
    frameon=False
)

# Thicker spines
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
    spine.set_zorder(50)

# ==========================================
#  SPECIAL POINTS – always on top + no clip
# ==========================================

# Small filled red circle at (sqrt(2/3), sqrt(1/3))
ax.scatter(
    x_pt, y_pt,
    s=50,
    color="red",
    edgecolors="black",
    linewidth=1.0,
    zorder=1000,
    clip_on=False
)

# Small filled blueish circle at (0, 1)  (Bell states)
ax.scatter(
    0.0, 1.0,
    s=50,
    color="#99CAF0",
    edgecolors="black",
    linewidth=1.0,
    zorder=1000,
    clip_on=False
)

# Small filled orange circle at (sqrt(2/3), 0)  (pure ⊗ max mixed)
ax.scatter(
    np.sqrt(2/3), 0.0,
    s=50,
    color="orange",
    edgecolors="black",
    linewidth=1.0,
    zorder=1000,
    clip_on=False
)

# Small filled yellow circle at (0, 0)  (max mixed)
ax.scatter(
    0.0, 0.0,
    s=50,
    color="#FFEE99",
    edgecolors="black",
    linewidth=1.0,
    zorder=1000,
    clip_on=False
)

# Export
plt.tight_layout()
plt.savefig("Fig_1A.pdf", dpi=300, bbox_inches="tight")

