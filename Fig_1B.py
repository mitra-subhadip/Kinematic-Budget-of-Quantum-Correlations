"""
Monte Carlo exploration of the two-qubit budget geometry.

This script:
  - Generates random two-qubit states of several types:
        * product_pure      : random product pure states
        * classical_CC      : random diagonal (classical) states
        * pure_entangled    : random pure 2-qubit states
        * werner_like       : random Werner-like states
        * mixed_entangled   : generic mixed entangled (PPT entangled) states
        * mixed_separable   : generic mixed separable (PPT separable, impure) states
  - Converts each density matrix rho into Bloch-Fano data (r,s,t)
  - Computes BL, BNL, B, P, X, Y, R
  - Plots the points on top of analytic resource-region boundaries.
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from dataclasses import dataclass
from scipy.interpolate import interp1d


# =============================
#  Font configuration (as in budgetplot4.py)
# =============================
font_path = "./times.ttf"
fm.fontManager.addfont(font_path)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.8,
})

# =============================
#  Pauli / basic linear algebra
# =============================
def dagger(M):
    return np.conjugate(M).T

sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0],
                    [0, -1]], dtype=complex)
identity_2 = np.eye(2, dtype=complex)
paulis = [sigma_x, sigma_y, sigma_z]

def tensor(a, b):
    return np.kron(a, b)


# =============================
#  Random state generators
# =============================
def random_complex_vector(d):
    v = np.random.normal(size=(d,)) + 1j * np.random.normal(size=(d,))
    v /= np.linalg.norm(v)
    return v

def random_pure_state_2qubit():
    v = random_complex_vector(4)
    return np.outer(v, np.conjugate(v))

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

def bell_state_phi_plus():
    v = np.zeros(4, dtype=complex)
    v[0] = 1.0 / np.sqrt(2.0)
    v[3] = 1.0 / np.sqrt(2.0)
    return np.outer(v, np.conjugate(v))

def werner_like_state():
    p = np.random.rand()
    rho_bell = bell_state_phi_plus()
    rho_mix = np.eye(4, dtype=complex) / 4.0
    return p * rho_bell + (1.0 - p) * rho_mix

def random_mixed_state_2qubit():
    G = np.random.normal(size=(4, 4)) + 1j * np.random.normal(size=(4, 4))
    M = G @ dagger(G)
    return M / np.trace(M)


# =============================
#  PPT / negativity
# =============================
def partial_transpose(rho, sys='A'):
    rho = np.asarray(rho, dtype=complex).reshape(4, 4)
    rho_pt = np.zeros_like(rho, dtype=complex)

    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    i = 2 * a + b
                    j = 2 * c + d
                    if sys == 'A':
                        i_pt = 2 * c + b
                        j_pt = 2 * a + d
                    else:
                        i_pt = 2 * a + d
                        j_pt = 2 * c + b
                    rho_pt[i_pt, j_pt] = rho[i, j]
    return rho_pt

def negativity(rho):
    rho_pt = partial_transpose(rho, sys='A')
    evals = np.linalg.eigvalsh(rho_pt)
    neg_evals = evals[evals < 0]
    return float(-np.sum(neg_evals))

def random_mixed_entangled_state(max_tries=5000):
    for _ in range(max_tries):
        rho = random_mixed_state_2qubit()
        if negativity(rho) > 1e-8:
            return rho
    raise RuntimeError("Failed to find entangled mixed state within max_tries")

def random_mixed_separable_state(max_tries=5000):
    for _ in range(max_tries):
        rho = random_mixed_state_2qubit()
        neg = negativity(rho)
        P = np.real_if_close(np.trace(rho @ rho))
        if neg < 1e-8 and P < 0.999:
            return rho
    raise RuntimeError("Failed to find mixed separable state within max_tries")

# =============================
#  Fano decomposition
# =============================
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


# =============================
#  Budget point
# =============================
@dataclass
class BudgetPoint:
    label: str
    X: float
    Y: float
    R: float

def compute_budget_from_r_s_t(r, s, t, label=""):
    r = np.asarray(r, dtype=float).reshape(3)
    s = np.asarray(s, dtype=float).reshape(3)
    t = np.asarray(t, dtype=float).reshape(3, 3)

    BL = float(np.dot(r, r) + np.dot(s, s))
    BNL = float(np.sum(t**2))
    B = BL + BNL
    P = (B + 1.0) / 4.0

    eps = 1e-12
    P_safe = max(P, eps)

    X_sq = BL / (3.0 * P_safe)
    Y_sq = BNL / (3.0 * P_safe)

    X_sq = max(X_sq, 0.0)
    Y_sq = max(Y_sq, 0.0)

    X = float(np.sqrt(X_sq))
    Y = float(np.sqrt(Y_sq))
    R = X_sq + Y_sq

    return BudgetPoint(label=label, X=X, Y=Y, R=R)


# =============================
#  Generate Monte Carlo points
# =============================
def generate_budget_points(N_PER_TYPE=50000, seed=int(time.time())):
    np.random.seed(seed)

    points_by_type = {
        "Pure_product": [],
        "CC": [],
        "Pure_entangled": [],
        "Werner-like": [],
        "Mixed_entangled": [],
        "Mixed_separable": [],
    }

    # 1) Product pure
    for _ in range(N_PER_TYPE):
        rho = random_product_pure_state()
        r, s, t = rho_to_fano(rho)
        bp = compute_budget_from_r_s_t(r, s, t, label="Pure_product")
        points_by_type["Pure_product"].append(bp)

    # 2) Classical CC (diagonal)
    for _ in range(N_PER_TYPE):
        rho = random_classical_CC_state()
        r, s, t = rho_to_fano(rho)
        bp = compute_budget_from_r_s_t(r, s, t, label="CC")
        points_by_type["CC"].append(bp)

    # 3) Pure 2-qubit (typically entangled)
    for _ in range(N_PER_TYPE):
        rho = random_pure_state_2qubit()
        r, s, t = rho_to_fano(rho)
        bp = compute_budget_from_r_s_t(r, s, t, label="Pure_entangled")
        points_by_type["Pure_entangled"].append(bp)

    # 4) Werner-like
    for _ in range(N_PER_TYPE):
        rho = werner_like_state()
        r, s, t = rho_to_fano(rho)
        bp = compute_budget_from_r_s_t(r, s, t, label="Werner-like")
        points_by_type["Werner-like"].append(bp)

    # 5) Mixed entangled (NPT)
    for _ in range(10*N_PER_TYPE):
        rho = random_mixed_entangled_state()
        r, s, t = rho_to_fano(rho)
        bp = compute_budget_from_r_s_t(r, s, t, label="Mixed_entangled")
        points_by_type["Mixed_entangled"].append(bp)

    # 6) Mixed separable (PPT & impure)
    for _ in range(2*N_PER_TYPE):
        rho = random_mixed_separable_state()
        r, s, t = rho_to_fano(rho)
        bp = compute_budget_from_r_s_t(r, s, t, label="Mixed_separable")
        points_by_type["Mixed_separable"].append(bp)

    return points_by_type



# =============================
#  Plotting
# =============================
def plot_budget_with_monte_carlo(points_by_type, show=True, save_prefix="Budget-XY-MC"):
    # ===== Analytic background from budgetplot4.py =====
    X = np.linspace(0, 1, 2000)

    # Purity boundary R = 1
    Y_R1 = np.sqrt(np.maximum(0, 1 - X**2))

    # PPT boundary R = 1/3
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

    # Correct feasibility boundary X_max(Y)
    Y = np.linspace(0, 1, 2000)
    X_max = np.zeros_like(Y)
    mask_lowY  = Y**2 < 1/3
    mask_highY = ~mask_lowY
    X_max[mask_lowY]  = np.sqrt(2/3)
    X_max[mask_highY] = np.sqrt(1 - Y[mask_highY]**2)
    X_R1 = np.sqrt(np.maximum(0, 1 - Y**2))

    # Colours (as in budgetplot4.py)
    col_R1         = "#5AA9E6"   # CHSH guaranteed
    col_CHSH_below = "#7FC8F8"   # Below CHSH
    col_CC         = "#FFEBEB"   # Classical region
    col_R1_outer   = "#FFEE99"   # Discord wedge
    orange         = "#FF5E00"
    ppt_color      = "#FF6392"   # Pink

    fig, ax = plt.subplots(figsize=(7, 7))

    # --- region fills (low zorder) ---
    ax.fill_between(X, 0, Y_R1, alpha=0.4, color=col_R1, zorder=1)          # Purity disk
    ax.fill_between(X, 0, Y_chsh, alpha=0.45, color=col_CHSH_below, zorder=2)
    ax.fill_between(X, 0, Y_cc_interp, alpha=1, color=col_CC, zorder=3)
    ax.fill_between(X, 0, Y_R13, alpha=0.65, color="#FFEE99", zorder=8)      # PPT region

    # CHSH ellipse boundary
    CHSH_line, = ax.plot(X, Y_chsh, "-.", color="#274C77", linewidth=2,zorder=6)

    # CC / QC / CQ boundary
    CC_line_top, = ax.plot(X_cc, Y_cc, color="#975AE6", linewidth=2.5, zorder=7)
    ax.plot([x_max_cc, x_max_cc], [0, Y_cc[-1]], color="#975AE6", linewidth=2.5, zorder=7)

    # PPT boundary
    PPT_line, = ax.plot(X, Y_R13, "--", color=ppt_color, linewidth=2.5,zorder=9)

    # horizontal PPT extension
    epsilon = 0.004
    x0 = np.sqrt(1/3)
    ax.plot(np.linspace(x0, 1, 400), epsilon*np.ones(400),
            "--", color=ppt_color, linewidth=2.5, zorder=9)

    # UNFEASIBLE REGION: X > X_max(Y)
    ax.fill_betweenx(
        Y, X_max, X_R1,
        where=(X_R1 > np.sqrt(2.0/3)),
        color="lightgray",
        zorder=20,
    )
    ax.plot(X_max, Y, color="black", linewidth=2.5, zorder=21)  # feasibility boundary

    # Restrict purity boundary to X ≤ sqrt(2/3)
    mask_purity = X <= x_max_cc
    MAX_Xline, = ax.plot(X[mask_purity], Y_R1[mask_purity],
                         color="#274C77", linewidth=2.5, zorder=22)

    # Grey continuation of purity boundary
    mask_purity_gray = X > x_max_cc
    ax.plot(X[mask_purity_gray], Y_R1[mask_purity_gray], "--",
            color="#49616E", linewidth=2.5, zorder=22)



    x_pt = np.sqrt(2/3)
    y_pt = np.sqrt(1/3)

    # ====================================
    # Monte Carlo scatter
    # ====================================
    type_styles = {
    "Pure_product":   dict(marker="o", s=30, alpha=1.00, facecolors="red", edgecolors="black"),
    "CC":             dict(marker="o", s=20, alpha=1.00, linewidth=0.5, facecolors="none", edgecolors="#FF6392"),
    "Pure_entangled": dict(marker="o", s=10, alpha=1.00, linewidth=0.5, facecolors="none", edgecolors="#71AEF5"),
    "Werner-like":    dict(marker="P", s=20, alpha=1.00, linewidth=0.5, facecolors="none", edgecolors="#7A5195"),
    "Mixed_entangled":dict(marker="*", s=20, alpha=1.00, linewidth=0.5, facecolors="none", edgecolors="#274C77"),
    "Mixed_separable":dict(marker="p", s=20, alpha=1.00, linewidth=0.5, facecolors="#FFEE99", edgecolors="black"),
    }


    scatter_handles = []
    for t_label, plist in points_by_type.items():
        if not plist:
            continue
        Xs = [p.X for p in plist]
        Ys = [p.Y for p in plist]
        style = type_styles.get(t_label, dict(marker=".", s=8, alpha=0.4))
        sc = ax.scatter(Xs, Ys, zorder=80, label=t_label.replace("_", " "), **style)
        scatter_handles.append(sc)

    # ====================================
    # Special points (still below text)
    # ====================================
    ax.scatter(
        x_pt, y_pt,
        s=50,
        color="red",
        edgecolors="black",
        linewidth=0.5,
        zorder=90,
        clip_on=False
    )
   
    # ====================================
    # Axes & Legend
    # ====================================
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("$X$", fontsize=20)
    ax.set_ylabel("$Y$", fontsize=20)
    ax.tick_params(labelsize=16)
   
    ax.grid(alpha=0.25)

    # combine analytic boundaries + MC types in legend
    handles = scatter_handles #[CHSH_line, CC_line_top, PPT_line] + 
    labels = [h.get_label() for h in handles]
    ax.legend(handles=handles, labels=labels,
              loc="upper right", fontsize=15, frameon=False)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_zorder(50)

   
    plt.tight_layout()
   
    plt.savefig("Fig_1B.pdf", dpi=300, bbox_inches="tight")
 
    

# =============================
#  Main
# =============================
def main():
    N_PER_TYPE = 5000  
    points_by_type = generate_budget_points(N_PER_TYPE=N_PER_TYPE, seed=1234)
    plot_budget_with_monte_carlo(points_by_type, show=True, save_prefix="Fig_1B")

if __name__ == "__main__":
    main()

