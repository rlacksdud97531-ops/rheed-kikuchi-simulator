"""
RHEED Kikuchi Line Analysis Tool
Based on:
  - Mitura et al., Acta Cryst. A80, 104-111 (2024)
  - Pawlak, Przybylski & Mitura, Materials 14, 7077 (2021)

Coordinate system (physically correct RHEED geometry):
  X : along beam direction (toward screen), horizontal
  Y : horizontal, perpendicular to beam (along surface)
  Z : perpendicular to surface (upward, positive = vacuum)

  Surface plane = XY plane.
  Screen at X = L (vertical screen, L = sample-to-screen distance along beam).
  Screen coords: Y_s = (K_fY/K_fX)*L,  Z_s = (K_fZ/K_fX)*L

  Incident beam at glancing angle theta (measured from surface):
    K_iX =  |Ki| cos(theta)  (large, along surface toward screen)
    K_iY =  0
    K_iZ = -|Ki| sin(theta)  (small, downward into surface)
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Physical constants
hbar  = 1.054571817e-34   # J*s
m0    = 9.1093837015e-31  # kg
qe    = 1.602176634e-19   # C
c     = 2.99792458e8      # m/s

# Crystal presets
CRYSTAL_PRESETS = {
    "SrTiO3": {
        "a": 3.905e-10,    # cubic lattice constant [m]
        "V_I": 15.08,      # mean inner potential [eV] (Pawlak 2021 Eq.7)
    },
    "LSMO": {
        "a": 3.876e-10,
        "V_I": 14.0,
    },
    "Cu": {
        "a": 3.615e-10,
        "V_I": 12.0,
    },
}


# ============================================================================
#  PHYSICS HELPERS
# ============================================================================

def ki_magnitude(energy_keV: float) -> float:
    """Relativistic |Ki| in m^-1."""
    E_J = energy_keV * 1e3 * qe
    gamma_corr = 1.0 + E_J / (2.0 * m0 * c**2)
    p = np.sqrt(2.0 * m0 * E_J * gamma_corr)
    return p / hbar


def reduced_potential(V_I_eV: float, energy_keV: float) -> float:
    """
    v_tilde < 0  (Eq.6 Pawlak / Eq.2 Mitura).
    v_tilde = -(1 + |qe|U/(m0 c^2)) * (2m0/hbar^2) * V_I   [m^-2]
    V_I is the mean inner potential in eV (positive number).
    """
    U   = energy_keV * 1e3
    rel = 1.0 + (qe * U) / (m0 * c**2)
    return -rel * (2.0 * m0 / hbar**2) * (V_I_eV * qe)


def ki_vec(Ki: float, theta_deg: float, azimuth_deg: float = 0.0):
    """
    Incident wave-vector in physical (X,Y,Z) coordinates.
    azimuth_deg: rotation of crystal about Z around the specular direction.
    Returns (K_iX, K_iY, K_iZ).
    """
    th  = np.radians(theta_deg)
    phi = np.radians(azimuth_deg)
    K_iX_0 = Ki * np.cos(th)
    K_iY_0 = 0.0
    K_iZ   = -Ki * np.sin(th)
    K_iX = K_iX_0 * np.cos(phi) - K_iY_0 * np.sin(phi)
    K_iY = K_iX_0 * np.sin(phi) + K_iY_0 * np.cos(phi)
    return K_iX, K_iY, K_iZ


def surface_g_vecs(a: float, hmax: int) -> np.ndarray:
    """
    2D surface reciprocal lattice vectors g = (gX, gY) = (h,k)*2pi/a
    for the (001) face of a cubic crystal. Excludes (0,0). Shape (N,2).
    """
    b = 2.0 * np.pi / a
    rows = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            if h == 0 and k == 0:
                continue
            rows.append([h * b, k * b])
    return np.array(rows)


def bulk_G_vecs(a: float, hmax: int) -> np.ndarray:
    """
    3D reciprocal lattice vectors G = (GX,GY,GZ) = (h,k,l)*2pi/a.
    Excludes (0,0,0). Shape (N,3).
    """
    b = 2.0 * np.pi / a
    rows = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                rows.append([h * b, k * b, l * b])
    return np.array(rows)


# ============================================================================
#  1. BRAGG SPOTS
# ============================================================================

def compute_bragg_spots(Ki, theta_deg, a, azimuth_deg=0.0, hmax=4, L=0.2):
    """
    Elastic RHEED spots. Returns list of (Y_mm, Z_mm) screen coordinates.
    L: sample-to-screen distance in metres.
    """
    K_iX, K_iY, K_iZ = ki_vec(Ki, theta_deg, azimuth_deg)
    b = 2.0 * np.pi / a
    spots = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            K_fX = K_iX + h * b
            K_fY = K_iY + k * b
            val  = Ki**2 - K_fX**2 - K_fY**2
            if val <= 0 or K_fX <= 0:
                continue
            K_fZ = np.sqrt(val)
            Y_s = (K_fY / K_fX) * L * 1e3  # mm
            Z_s = (K_fZ / K_fX) * L * 1e3
            spots.append((Y_s, Z_s))
    return spots


# ============================================================================
#  2. BRAGG REFLECTION KIKUCHI LINES  (Eq.5 Pawlak / Eq.1 Mitura)
# ============================================================================

def _quadratic_K_fy(R, GX, GY, q):
    """
    Solve for K_fY given:
      K_fX = (R - 2*K_fY*GY) / (2*GX)
      K_fX^2 + K_fY^2 = q   (q = Ki^2 - K_fZ^2)
    Returns up to two real K_fY solutions.
    """
    Gperp2 = GX**2 + GY**2
    if Gperp2 == 0:
        return []
    a_coef = Gperp2
    b_coef = -R * GY
    c_coef = R**2 / 4.0 - GX**2 * q
    disc   = b_coef**2 - 4.0 * a_coef * c_coef
    if disc < 0:
        return []
    sq = np.sqrt(disc)
    return [(-b_coef + sq) / (2.0 * a_coef),
            (-b_coef - sq) / (2.0 * a_coef)]


def bragg_kikuchi_line(G, Ki, theta_deg, v_tilde, azimuth_deg=0.0,
                       L=0.2, y_range=(-80, 80), npts=500):
    """
    Bragg reflection Kikuchi line for one G=(GX,GY,GZ).
    Handles three cases:
      * GX=GY=0 (pure Laue zone line): K_fZ fixed, sweep K_fY
      * GX=0, GY!=0: K_fY from Bragg, K_fX from sphere
      * GX!=0: quadratic in K_fY
    Returns (Y_pts_mm, Z_pts_mm).
    """
    GX, GY, GZ = G
    G2   = GX**2 + GY**2 + GZ**2
    pts_Y, pts_Z = [], []

    # Case 1: G = (0, 0, GZ) -> Laue-zone horizontal curve
    if abs(GX) < 1.0 and abs(GY) < 1.0:
        if abs(GZ) < 1.0:
            return pts_Y, pts_Z
        K_fZ2 = (GZ / 2.0)**2 + v_tilde   # v_tilde < 0
        if K_fZ2 <= 0:
            return pts_Y, pts_Z
        K_fZ_fixed = np.sqrt(K_fZ2)
        q_max = Ki**2 - K_fZ_fixed**2
        if q_max <= 0:
            return pts_Y, pts_Z
        K_fY_max = np.sqrt(q_max)
        K_fY_arr = np.linspace(-K_fY_max * 0.999, K_fY_max * 0.999, npts)
        for K_fY in K_fY_arr:
            K_fX2 = q_max - K_fY**2
            if K_fX2 <= 0:
                continue
            K_fX = np.sqrt(K_fX2)
            _add_point(K_fX, K_fY, K_fZ_fixed, L, y_range, pts_Y, pts_Z)
        return pts_Y, pts_Z

    # Cases 2 & 3: sweep K_fZ, solve for K_fY
    K_fZ_arr = np.linspace(1e7, Ki * 0.98, npts)
    for K_fZ in K_fZ_arr:
        inner = K_fZ**2 - v_tilde   # > 0 since v_tilde < 0
        if inner <= 0:
            continue
        R = G2 - 2.0 * np.sqrt(inner) * GZ
        q = Ki**2 - K_fZ**2
        if q <= 0:
            continue

        if abs(GX) < 1.0:          # GX=0, GY!=0
            K_fY_val = R / (2.0 * GY)
            K_fX2    = q - K_fY_val**2
            if K_fX2 <= 0:
                continue
            K_fX = np.sqrt(K_fX2)
            _add_point(K_fX, K_fY_val, K_fZ, L, y_range, pts_Y, pts_Z)
        else:                       # GX!=0: quadratic
            for K_fY in _quadratic_K_fy(R, GX, GY, q):
                K_fX = (R - 2.0 * K_fY * GY) / (2.0 * GX)
                _add_point(K_fX, K_fY, K_fZ, L, y_range, pts_Y, pts_Z)

    return pts_Y, pts_Z


# ============================================================================
#  3. RESONANCE SCATTERING LINES  (Eq.8 Pawlak / Eq.3 Mitura)
# ============================================================================

def resonance_line(g, Ki, theta_deg, v_tilde, alpha=1.0, azimuth_deg=0.0,
                   L=0.2, y_range=(-80, 80), npts=500):
    """
    Resonance Kikuchi line for one g=(gX,gY).
    Eq.8: 2K_fX*gX + 2K_fY*gY + K_fZ^2 - alpha*v_tilde = |g|^2
    """
    gX, gY = g
    g2 = gX**2 + gY**2
    pts_Y, pts_Z = [], []

    K_fZ_arr = np.linspace(1e7, Ki * 0.98, npts)
    for K_fZ in K_fZ_arr:
        R = g2 - K_fZ**2 + alpha * v_tilde
        q = Ki**2 - K_fZ**2
        if q <= 0:
            continue

        if abs(gX) < 1.0:
            if abs(gY) < 1.0:
                continue
            K_fY_val = R / (2.0 * gY)
            K_fX2    = q - K_fY_val**2
            if K_fX2 <= 0:
                continue
            K_fX = np.sqrt(K_fX2)
            _add_point(K_fX, K_fY_val, K_fZ, L, y_range, pts_Y, pts_Z)
            continue

        for K_fY in _quadratic_K_fy(R, gX, gY, q):
            K_fX = (R - 2.0 * K_fY * gY) / (2.0 * gX)
            _add_point(K_fX, K_fY, K_fZ, L, y_range, pts_Y, pts_Z)

    return pts_Y, pts_Z


def _add_point(K_fX, K_fY, K_fZ, L, y_range, pts_Y, pts_Z):
    """Add a valid screen point if K_fX > 0 and Y is in range."""
    if K_fX <= 0 or K_fZ <= 0:
        return
    Y_s = (K_fY / K_fX) * L * 1e3
    Z_s = (K_fZ / K_fX) * L * 1e3
    if y_range[0] <= Y_s <= y_range[1]:
        pts_Y.append(Y_s)
        pts_Z.append(Z_s)


# ============================================================================
#  4. IMAGE FILTER  (RDB + AHE)
# ============================================================================

def filter_rheed_image(image_path: str, kernel_size: int = 15):
    """
    Applies RDB (remove dynamic background via Gaussian blur) followed by
    AHE (adaptive histogram equalization) as in Mitura et al. 2024 Sec.3.
    Returns filtered numpy array (float32, 0-1).
    Requires: pip install scikit-image
    """
    try:
        import skimage.io as skio
        import skimage.exposure as skexp
        import skimage.filters as skfilt
    except ImportError:
        raise ImportError("Install: pip install scikit-image")

    raw = skio.imread(image_path).astype(np.float32)
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    raw /= raw.max() + 1e-9

    # RDB: subtract Gaussian-blurred background
    sigma  = max(raw.shape) / 8.0
    bg     = skfilt.gaussian(raw, sigma=sigma)
    rdb    = raw - bg
    rdb   -= rdb.min()
    rdb   /= rdb.max() + 1e-9

    # AHE: adaptive histogram equalization
    ahe = skexp.equalize_adapthist(rdb, kernel_size=kernel_size, clip_limit=0.03)
    return ahe.astype(np.float32)


# ============================================================================
#  5. MAIN PLOT
# ============================================================================

def plot_rheed(
    crystal_name      : str   = "SrTiO3",
    energy_keV        : float = 20.0,
    theta_deg         : float = 2.9,
    azimuth_deg       : float = 0.0,
    L_mm              : float = 200.0,
    hmax_3d           : int   = 4,
    hmax_2d           : int   = 4,
    alpha_res         : float = 1.0,
    image_path        : str   = None,
    kernel_size       : int   = 15,
    y_range_mm        : tuple = (-60, 60),
    z_range_mm        : tuple = (0, 80),
    show_spots        : bool  = True,
    show_bragg_lines  : bool  = True,
    show_resonance    : bool  = True,
    save_path         : str   = None,
):
    preset = CRYSTAL_PRESETS[crystal_name]
    a   = preset["a"]
    V_I = preset["V_I"]
    Ki  = ki_magnitude(energy_keV)
    vt  = reduced_potential(V_I, energy_keV)
    L   = L_mm * 1e-3

    th  = np.radians(theta_deg)
    z_spec = np.tan(th) * L_mm

    print(f"Crystal : {crystal_name}  a={a*1e10:.3f} A  V_I={V_I} eV")
    print(f"Energy  : {energy_keV} keV   |Ki|={Ki:.4e} m^-1   v_tilde={vt:.4e} m^-2")
    print(f"theta={theta_deg} deg  phi={azimuth_deg} deg  L={L_mm} mm")
    print(f"Specular spot expected at Z_s ~ {z_spec:.1f} mm")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("#111111")

    # background image
    if image_path and Path(image_path).exists():
        print(f"\nFiltering: {image_path}")
        try:
            filt = filter_rheed_image(image_path, kernel_size)
            ax.imshow(filt, cmap="gray",
                      extent=[y_range_mm[0], y_range_mm[1],
                              z_range_mm[0], z_range_mm[1]],
                      aspect="auto", origin="lower", alpha=0.85)
            print("  RDB + AHE done.")
        except Exception as e:
            print(f"  WARNING: {e}")

    # Bragg spots
    if show_spots:
        spots = compute_bragg_spots(Ki, theta_deg, a, azimuth_deg,
                                    hmax=hmax_3d, L=L)
        visible = [(y, z) for y, z in spots
                   if y_range_mm[0] <= y <= y_range_mm[1]
                   and z_range_mm[0] <= z <= z_range_mm[1]]
        if visible:
            ys, zs = zip(*visible)
            ax.scatter(ys, zs, color="lime", s=25, zorder=5, label="Bragg spots")
        print(f"  {len(visible)} Bragg spots visible")

    # Bragg Kikuchi lines
    if show_bragg_lines:
        G_vecs   = bulk_G_vecs(a, hmax=hmax_3d)
        n_plotted = 0
        for G in G_vecs:
            py, pz = bragg_kikuchi_line(G, Ki, theta_deg, vt,
                                        azimuth_deg=azimuth_deg, L=L,
                                        y_range=y_range_mm)
            mask = [z_range_mm[0] <= z <= z_range_mm[1] for z in pz]
            py2  = [y for y, m in zip(py, mask) if m]
            pz2  = [z for z, m in zip(pz, mask) if m]
            if len(py2) > 3:
                ord_ = np.argsort(py2)
                ax.plot(np.array(py2)[ord_], np.array(pz2)[ord_],
                        color="dodgerblue", lw=0.7, alpha=0.75, zorder=3)
                n_plotted += 1
        print(f"  {n_plotted} Bragg Kikuchi lines")

    # Resonance lines
    if show_resonance:
        g_vecs   = surface_g_vecs(a, hmax=hmax_2d)
        n_plotted = 0
        for g in g_vecs:
            py, pz = resonance_line(g, Ki, theta_deg, vt,
                                    alpha=alpha_res,
                                    azimuth_deg=azimuth_deg, L=L,
                                    y_range=y_range_mm)
            mask = [z_range_mm[0] <= z <= z_range_mm[1] for z in pz]
            py2  = [y for y, m in zip(py, mask) if m]
            pz2  = [z for z, m in zip(pz, mask) if m]
            if len(py2) > 3:
                ord_ = np.argsort(py2)
                ax.plot(np.array(py2)[ord_], np.array(pz2)[ord_],
                        color="tomato", lw=0.7, alpha=0.75, zorder=3)
                n_plotted += 1
        print(f"  {n_plotted} resonance lines")

    # legend & labels
    handles = []
    if show_spots:
        handles.append(mpatches.Patch(color="lime",       label="Bragg spots"))
    if show_bragg_lines:
        handles.append(mpatches.Patch(color="dodgerblue", label="Bragg Kikuchi lines"))
    if show_resonance:
        handles.append(mpatches.Patch(color="tomato",     label="Resonance lines"))

    ax.legend(handles=handles, loc="upper right", fontsize=9,
              facecolor="#222222", labelcolor="white")
    ax.set_xlim(y_range_mm)
    ax.set_ylim(z_range_mm)
    ax.invert_yaxis()   # shadow edge at top, matching paper convention
    ax.set_xlabel("Y  [mm]  (horizontal, perp beam)", color="white")
    ax.set_ylabel("Z  [mm]  (distance from shadow edge)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.axhline(0, color="#555555", lw=0.5, ls="--")
    ax.text(y_range_mm[0] + 1, 0.5, "shadow edge", color="#888888",
            fontsize=7, va="top")

    title = (f"RHEED Kikuchi -- {crystal_name} (001)  |  "
             f"E={energy_keV} keV  theta={theta_deg} deg  phi={azimuth_deg} deg  L={L_mm} mm")
    ax.set_title(title, color="white", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nSaved to: {save_path}")
    else:
        plt.show()
    return fig, ax


# ============================================================================
#  CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RHEED Kikuchi simulator + image filter")
    p.add_argument("--crystal",  default="SrTiO3", choices=list(CRYSTAL_PRESETS))
    p.add_argument("--energy",   type=float, default=20.0,  help="keV")
    p.add_argument("--theta",    type=float, default=2.9,   help="glancing angle [deg]")
    p.add_argument("--azimuth",  type=float, default=0.0,   help="azimuth offset [deg]")
    p.add_argument("--screen",   type=float, default=200.0, help="sample-to-screen [mm]")
    p.add_argument("--image",    default=None, help="path to .tif RHEED image")
    p.add_argument("--kernel",   type=int, default=15,      help="AHE kernel size [px]")
    p.add_argument("--save",     default=None, help="save figure here instead of showing")
    p.add_argument("--no-spots",     action="store_true")
    p.add_argument("--no-bragg",     action="store_true")
    p.add_argument("--no-resonance", action="store_true")
    args = p.parse_args()

    plot_rheed(
        crystal_name     = args.crystal,
        energy_keV       = args.energy,
        theta_deg        = args.theta,
        azimuth_deg      = args.azimuth,
        L_mm             = args.screen,
        image_path       = args.image,
        kernel_size      = args.kernel,
        save_path        = args.save,
        show_spots       = not args.no_spots,
        show_bragg_lines = not args.no_bragg,
        show_resonance   = not args.no_resonance,
    )
