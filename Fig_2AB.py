import time
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import matplotlib.font_manager as fm
import os 

# ============================================================
#  1. Font & Plot Configuration
# ============================================================
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

# ============================================================
#  2. JAX Setup: State Generation & XYR Kinematics
# ============================================================
# Pauli operators for JAX
sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
I2 = jnp.eye(2, dtype=jnp.complex64)

paulis = jnp.stack([sigma_x, sigma_y, sigma_z])
PA = jnp.stack([jnp.kron(p, I2) for p in paulis])
PB = jnp.stack([jnp.kron(I2, p) for p in paulis])
PAB = jnp.stack([
    jnp.stack([jnp.kron(paulis[i], paulis[j]) for j in range(3)], axis=0)
    for i in range(3)
], axis=0)

@partial(jax.jit, static_argnums=1)
def random_mixed_states(key, batch):
    key_r, key_i = jax.random.split(key)
    G = (
        jax.random.normal(key_r, (batch, 4, 4)) +
        1j * jax.random.normal(key_i, (batch, 4, 4))
    )
    M = G @ jnp.conj(jnp.transpose(G, (0, 2, 1)))
    tr = jnp.trace(M, axis1=1, axis2=2)
    return M / tr[:, None, None]

@jax.jit
def rho_to_fano_batch(rho):
    r = jnp.real(jnp.einsum("bij,kji->bk", rho, PA))
    s = jnp.real(jnp.einsum("bij,kji->bk", rho, PB))
    t = jnp.real(jnp.einsum("bij,klji->bkl", rho, PAB))
    return r, s, t

@jax.jit
def compute_XYR_batch(rho):
    r, s, t = rho_to_fano_batch(rho)
    BL = jnp.sum(r * r, axis=1) + jnp.sum(s * s, axis=1)
    BNL = jnp.sum(t * t, axis=(1, 2))
    P = (BL + BNL + 1.0) / 4.0
    X = jnp.sqrt(BL / (3.0 * P))
    Y = jnp.sqrt(BNL / (3.0 * P))
    R = X * X + Y * Y
    return X, Y, R

def make_sampler(batch):
    @jax.jit
    def rejection_step(state, _):
        key, count, rho_out, X_out, Y_out, R_target, delta_R = state
        key, subkey = jax.random.split(key)
        rho = random_mixed_states(subkey, batch)
        X, Y, R = compute_XYR_batch(rho)

        mask = (jnp.abs(R - R_target) < delta_R)
        idx = jnp.nonzero(mask, size=batch, fill_value=0)[0]
        rho_c, X_c, Y_c = rho[idx], X[idx], Y[idx]
        n_new = jnp.sum(mask)

        space = rho_out.shape[0] - count
        n_take = jnp.minimum(space, n_new)

        write_mask = jnp.arange(batch) < n_take
        rho_write = jnp.where(write_mask[:, None, None], rho_c, jnp.zeros_like(rho_c))
        X_write = jnp.where(write_mask, X_c, 0.0)
        Y_write = jnp.where(write_mask, Y_c, 0.0)

        rho_out = lax.dynamic_update_slice(rho_out, rho_write, (count, 0, 0))
        X_out = lax.dynamic_update_slice(X_out, X_write, (count,))
        Y_out = lax.dynamic_update_slice(Y_out, Y_write, (count,))
        
        count = count + n_take
        return (key, count, rho_out, X_out, Y_out, R_target, delta_R), None

    def sample(R_target, n_accept, delta_R, max_iters):
        rho_out = jnp.zeros((n_accept, 4, 4), dtype=jnp.complex64)
        X_out = jnp.zeros((n_accept,), dtype=jnp.float32)
        Y_out = jnp.zeros((n_accept,), dtype=jnp.float32)

        init_state = (
            jax.random.PRNGKey(0), jnp.array(0, dtype=jnp.int32),
            rho_out, X_out, Y_out,
            jnp.array(R_target, dtype=jnp.float32), jnp.array(delta_R, dtype=jnp.float32)
        )
        final_state, _ = lax.scan(rejection_step, init_state, None, length=max_iters)
        _, count, rho_out, X_out, Y_out, *_ = final_state
        return rho_out[:count], X_out[:count], Y_out[:count]

    return sample

# ============================================================
#  3. Numba Setup: High-Speed Metrics Processing
# ============================================================
sigma_x_nb = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y_nb = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z_nb = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2_nb      = np.eye(2, dtype=np.complex128)

paulis_nb = np.stack([sigma_x_nb, sigma_y_nb, sigma_z_nb])
PA_nb  = np.empty((3, 4, 4), dtype=np.complex128)
PB_nb  = np.empty((3, 4, 4), dtype=np.complex128)
PAB_nb = np.empty((3, 3, 4, 4), dtype=np.complex128)

for i in range(3):
    PA_nb[i] = np.kron(paulis_nb[i], I2_nb)
    PB_nb[i] = np.kron(I2_nb, paulis_nb[i])
    for j in range(3):
        PAB_nb[i, j] = np.kron(paulis_nb[i], paulis_nb[j])

@njit(parallel=True, fastmath=True)
def compute_XY_magic_negativity(rho):
    N = rho.shape[0]
    X_vals, Y_vals = np.empty(N), np.empty(N)
    Magic_vals, Neg_vals = np.empty(N), np.empty(N)

    for n in prange(N):
        r, s, t = np.zeros(3), np.zeros(3), np.zeros((3, 3))
        for i in range(3):
            r[i] = np.real(np.trace(rho[n] @ PA_nb[i]))
            s[i] = np.real(np.trace(rho[n] @ PB_nb[i]))
            for j in range(3):
                t[i, j] = np.real(np.trace(rho[n] @ PAB_nb[i, j]))

        BL = np.dot(r, r) + np.dot(s, s)
        BNL = np.sum(t * t)
        P = (BL + BNL + 1.0) / 4.0

        X_vals[n] = np.sqrt(BL / (3.0 * P))
        Y_vals[n] = np.sqrt(BNL / (3.0 * P))

        num = 1.0 + np.sum(r**4) + np.sum(s**4) + np.sum(t**4)
        den = 1.0 + np.sum(r**2) + np.sum(s**2) + np.sum(t**2)
        Magic_vals[n] = -np.log2(num / den)

        rho_PT = np.empty((4, 4), dtype=np.complex128)
        for iA in range(2):
            for jB in range(2):
                for kA in range(2):
                    for lB in range(2):
                        rho_PT[2 * iA + jB, 2 * kA + lB] = rho[n, 2 * kA + jB, 2 * iA + lB]

        evals = np.linalg.eigvalsh(rho_PT)
        neg = 0.0
        for e in evals:
            if e < 0: neg -= e
        Neg_vals[n] = neg

    return X_vals, Y_vals, Magic_vals, Neg_vals

# ============================================================
#  4. Analytical Bounds & Binning
# ============================================================
def magic_upper_bound(R):
    S2 = 3.0 * R / (4.0 - 3.0 * R)
    gamma = 1.0 / (4.0 - 3.0 * R)
    return -np.log2((1.0 + S2**2 / 15.0) / (4.0 * gamma))

def negativity_upper_bound(R):
    if R <= 1.0 / 3.0: return 0.0
    val = np.sqrt(9.0 * R / (64.0 - 48.0 * R)) - 0.25
    return max(0.0, val)

def bin_arc_max(X, Y, Z, n_bins=10000):
    theta = np.arctan2(Y, X)
    bins = np.linspace(theta.min(), theta.max(), n_bins + 1)
    idx = np.digitize(theta, bins) - 1

    Xb, Yb, Zb = [], [], []
    for k in range(n_bins):
        mask = idx == k
        if not np.any(mask): continue
        j = np.argmax(Z[mask])
        Xb.append(X[mask][j])
        Yb.append(Y[mask][j])
        Zb.append(Z[mask][j])
    return np.array(Xb), np.array(Yb), np.array(Zb)

# ============================================================
#  5. Plotting Routines
# ============================================================
R_colors = {"R_0.400000": "#A25D5F", "R_0.600000": "#5FA25D", "R_0.800000": "#5D5FA2"}

def plot_theta_magic(XY_by_R, Magic_by_R, R_list, filename=None):
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    for key in R_list:
        X, Y = XY_by_R[key]
        Z = Magic_by_R[key]
        mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        X, Y, Z = X[mask], Y[mask], Z[mask]

        Xb, Yb, Zb = bin_arc_max(X, Y, Z, n_bins=10000)
        theta = np.arctan2(Yb, Xb) / (np.pi / 2)
        color = R_colors.get(key, "black")
        R_val = float(key.split("_")[1])

        ax.plot(theta, Zb, marker="o", linestyle="none", markersize=4,
                alpha=0.9, color=color, label=rf"$R={R_val:.2f}$")

        Mmax = magic_upper_bound(R_val)
        ax.hlines(Mmax, 0, 1, colors=color, linestyles="--", linewidth=2.2, alpha=0.9)
        ax.text(0.50, Mmax, rf"$\widetilde{{M}}_2={Mmax:.3f}$", fontsize=12,
                color=color, ha="center", va="bottom")

    ax.set_xlim(0,1)
    ax.set_ylim(0,1.2)
    ax.set_xlabel(r"$2\theta/\pi$", fontsize=16)
    ax.set_ylabel(r"$\widetilde{M}_2$", fontsize=16)

    ticks = ax.yaxis.get_major_ticks()
    if len(ticks)>0: ticks[0].label1.set_visible(False)

    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.03,0.82), frameon=False,
              fontsize=12, handletextpad=0.02)
    ax.set_box_aspect(1)
    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

def plot_theta_negativity(XY_by_R, Neg_by_R, R_list, filename=None):
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    for key in R_list:
        X, Y = XY_by_R[key]
        Z = Neg_by_R[key]
        mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        X, Y, Z = X[mask], Y[mask], Z[mask]

        Xb, Yb, Zb = bin_arc_max(X, Y, Z, n_bins=10000)
        theta = np.arctan2(Yb, Xb) / (np.pi / 2)
        color = R_colors.get(key, "black")
        R_val = float(key.split("_")[1])

        ax.plot(theta, Zb, marker="o", linestyle="none", markersize=4,
                alpha=0.9, color=color, label=rf"$R={R_val:.2f}$")

        Nmax = negativity_upper_bound(R_val)
        ax.hlines(Nmax, 0, 1, colors=color, linestyles="--", linewidth=2.2, alpha=0.9)
        if Nmax > 0:
            ax.text(0.50, Nmax + 0.005, rf"$\mathcal{{N}}_{{max}}={Nmax:.3f}$",
                    fontsize=12, color=color, ha="center", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel(r"$2\theta/\pi$", fontsize=16)
    ax.set_ylabel(r"$\mathcal{N}$", fontsize=16)

    ticks = ax.yaxis.get_major_ticks()
    if len(ticks)>0: ticks[0].label1.set_visible(False)

    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.03,0.82), frameon=False,
              fontsize=12, handletextpad=0.02)
    ax.set_box_aspect(1)
    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
#  6. Master Execution Block
# ============================================================
if __name__ == "__main__":
    batch = 8192
    sampler = make_sampler(batch)
    R_values = [0.4, 0.6, 0.8]
    
    XY_by_R = {}
    Magic_by_R = {}
    Neg_by_R = {}
    target_Rs = []

    print("--- Starting Integrated Generation & Processing Pipeline ---")
    
    for R in R_values:
        key_str = f"R_{R:.6f}"
        target_Rs.append(key_str)
        print(f"\n[1/2] JAX Sampling: R = {R:.3f}...")
        
        t0 = time.time()
        if R != 0.8:
            n_accept = 1000000
            delta_R = 3e-4
            max_iters = 300_000
        else:
            n_accept = 20000000
            delta_R = 3e-4
            max_iters = 5000_000
            
        rho_jax, _, _ = sampler(R_target=R, n_accept=n_accept, delta_R=delta_R, max_iters=max_iters)
        rho_jax.block_until_ready()
        print(f"      Generated {rho_jax.shape[0]} states in {time.time() - t0:.1f}s")
        
        # Pull from Device (GPU/TPU) to Host (CPU) memory
        rho_host = np.asarray(rho_jax, dtype=np.complex128)
        
        print(f"[2/2] Numba Processing: R = {R:.3f}...")
        t1 = time.time()
        X, Y, M_val, N_val = compute_XY_magic_negativity(rho_host)
        print(f"      Processed metrics in {time.time() - t1:.1f}s")
        
        # Store for plotting
        XY_by_R[key_str] = (X, Y)
        Magic_by_R[key_str] = M_val
        Neg_by_R[key_str] = N_val
        
        # Free up memory before the next heavy loop (especially for the 20M batch)
        del rho_jax
        del rho_host

    print("\n--- Generating Plots ---")
    plot_theta_magic(
        XY_by_R, Magic_by_R, target_Rs,
        filename="Fig_2A.pdf"
    )

    plot_theta_negativity(
        XY_by_R, Neg_by_R, target_Rs,
        filename="Fig_2B.pdf"
    )
