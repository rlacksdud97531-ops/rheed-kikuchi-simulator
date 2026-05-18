"""
RHEED Kikuchi Line Simulator — Streamlit Web App
Based on:
  - Mitura et al., Acta Cryst. A80, 104-111 (2024)
  - Pawlak, Przybylski & Mitura, Materials 14, 7077 (2021)

Run:
    streamlit run rheed_kikuchi_app.py
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from pathlib import Path

# ============================================================================
#  Physical constants
# ============================================================================
hbar = 1.054571817e-34
m0   = 9.1093837015e-31
qe   = 1.602176634e-19
c    = 2.99792458e8

CRYSTAL_PRESETS = {
    "SrTiO3": {"a": 3.905e-10, "V_I": 15.08},
    "LSMO":   {"a": 3.876e-10, "V_I": 14.0},
    "Cu":     {"a": 3.615e-10, "V_I": 12.0},
}

# ============================================================================
#  Physics helpers
# ============================================================================

def ki_magnitude(energy_keV):
    E_J = energy_keV * 1e3 * qe
    gamma_corr = 1.0 + E_J / (2.0 * m0 * c**2)
    p = np.sqrt(2.0 * m0 * E_J * gamma_corr)
    return p / hbar

def reduced_potential(V_I_eV, energy_keV):
    U   = energy_keV * 1e3
    rel = 1.0 + (qe * U) / (m0 * c**2)
    return -rel * (2.0 * m0 / hbar**2) * (V_I_eV * qe)

def ki_vec(Ki, theta_deg, azimuth_deg=0.0):
    th  = np.radians(theta_deg)
    phi = np.radians(azimuth_deg)
    K_iX_0 = Ki * np.cos(th)
    K_iZ   = -Ki * np.sin(th)
    K_iX = K_iX_0 * np.cos(phi)
    K_iY = K_iX_0 * np.sin(phi)
    return K_iX, K_iY, K_iZ

def surface_g_vecs(a, hmax):
    b = 2.0 * np.pi / a
    return np.array([[h*b, k*b]
                     for h in range(-hmax, hmax+1)
                     for k in range(-hmax, hmax+1)
                     if not (h==0 and k==0)])

def bulk_G_vecs(a, hmax):
    b = 2.0 * np.pi / a
    return np.array([[h*b, k*b, l*b]
                     for h in range(-hmax, hmax+1)
                     for k in range(-hmax, hmax+1)
                     for l in range(-hmax, hmax+1)
                     if not (h==0 and k==0 and l==0)])

def _add_point(K_fX, K_fY, K_fZ, L, y_range, pts_Y, pts_Z):
    if K_fX <= 0 or K_fZ <= 0:
        return
    Y_s = (K_fY / K_fX) * L * 1e3
    Z_s = (K_fZ / K_fX) * L * 1e3
    if y_range[0] <= Y_s <= y_range[1]:
        pts_Y.append(Y_s)
        pts_Z.append(Z_s)

def _quadratic_K_fy(R, GX, GY, q):
    Gp2 = GX**2 + GY**2
    if Gp2 == 0:
        return []
    a_c = Gp2
    b_c = -R * GY
    c_c = R**2 / 4.0 - GX**2 * q
    disc = b_c**2 - 4.0 * a_c * c_c
    if disc < 0:
        return []
    sq = np.sqrt(disc)
    return [(-b_c + sq) / (2.0*a_c), (-b_c - sq) / (2.0*a_c)]

# ============================================================================
#  Bragg spots
# ============================================================================

def compute_bragg_spots(Ki, theta_deg, a, azimuth_deg=0.0, hmax=4, L=0.2):
    K_iX, K_iY, K_iZ = ki_vec(Ki, theta_deg, azimuth_deg)
    b = 2.0 * np.pi / a
    spots = []
    for h in range(-hmax, hmax+1):
        for k in range(-hmax, hmax+1):
            K_fX = K_iX + h * b
            K_fY = K_iY + k * b
            val  = Ki**2 - K_fX**2 - K_fY**2
            if val <= 0 or K_fX <= 0:
                continue
            K_fZ = np.sqrt(val)
            spots.append(((K_fY/K_fX)*L*1e3, (K_fZ/K_fX)*L*1e3))
    return spots

# ============================================================================
#  Bragg Kikuchi lines
# ============================================================================

def bragg_kikuchi_line(G, Ki, theta_deg, v_tilde, azimuth_deg=0.0,
                       L=0.2, y_range=(-80,80), npts=500):
    GX, GY, GZ = G
    G2 = GX**2 + GY**2 + GZ**2
    pts_Y, pts_Z = [], []

    if abs(GX) < 1.0 and abs(GY) < 1.0:
        if abs(GZ) < 1.0:
            return pts_Y, pts_Z
        K_fZ2 = (GZ/2.0)**2 + v_tilde
        if K_fZ2 <= 0:
            return pts_Y, pts_Z
        K_fZ_fixed = np.sqrt(K_fZ2)
        q_max = Ki**2 - K_fZ_fixed**2
        if q_max <= 0:
            return pts_Y, pts_Z
        K_fY_max = np.sqrt(q_max)
        for K_fY in np.linspace(-K_fY_max*0.999, K_fY_max*0.999, npts):
            K_fX2 = q_max - K_fY**2
            if K_fX2 <= 0:
                continue
            _add_point(np.sqrt(K_fX2), K_fY, K_fZ_fixed, L, y_range, pts_Y, pts_Z)
        return pts_Y, pts_Z

    for K_fZ in np.linspace(1e7, Ki*0.98, npts):
        inner = K_fZ**2 - v_tilde
        if inner <= 0:
            continue
        R = G2 - 2.0*np.sqrt(inner)*GZ
        q = Ki**2 - K_fZ**2
        if q <= 0:
            continue
        if abs(GX) < 1.0:
            K_fY_val = R / (2.0*GY)
            K_fX2 = q - K_fY_val**2
            if K_fX2 <= 0:
                continue
            _add_point(np.sqrt(K_fX2), K_fY_val, K_fZ, L, y_range, pts_Y, pts_Z)
        else:
            for K_fY in _quadratic_K_fy(R, GX, GY, q):
                K_fX = (R - 2.0*K_fY*GY) / (2.0*GX)
                _add_point(K_fX, K_fY, K_fZ, L, y_range, pts_Y, pts_Z)
    return pts_Y, pts_Z

# ============================================================================
#  Resonance lines
# ============================================================================

def resonance_line(g, Ki, theta_deg, v_tilde, alpha=1.0,
                   L=0.2, y_range=(-80,80), npts=500):
    gX, gY = g
    g2 = gX**2 + gY**2
    pts_Y, pts_Z = [], []
    for K_fZ in np.linspace(1e7, Ki*0.98, npts):
        R = g2 - K_fZ**2 + alpha*v_tilde
        q = Ki**2 - K_fZ**2
        if q <= 0:
            continue
        if abs(gX) < 1.0:
            if abs(gY) < 1.0:
                continue
            K_fY_val = R / (2.0*gY)
            K_fX2 = q - K_fY_val**2
            if K_fX2 <= 0:
                continue
            _add_point(np.sqrt(K_fX2), K_fY_val, K_fZ, L, y_range, pts_Y, pts_Z)
        else:
            for K_fY in _quadratic_K_fy(R, gX, gY, q):
                K_fX = (R - 2.0*K_fY*gY) / (2.0*gX)
                _add_point(K_fX, K_fY, K_fZ, L, y_range, pts_Y, pts_Z)
    return pts_Y, pts_Z

# ============================================================================
#  Image filter (RDB + AHE)
# ============================================================================

def filter_rheed_image(img_array, kernel_size=15):
    from skimage import filters, exposure
    raw = img_array.astype(np.float32)
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    raw /= raw.max() + 1e-9
    sigma = max(raw.shape) / 8.0
    bg    = filters.gaussian(raw, sigma=sigma)
    rdb   = raw - bg
    rdb  -= rdb.min()
    rdb  /= rdb.max() + 1e-9
    ahe   = exposure.equalize_adapthist(rdb, kernel_size=kernel_size, clip_limit=0.03)
    return ahe.astype(np.float32)

# ============================================================================
#  Core plotting function
# ============================================================================

@st.cache_data(show_spinner=False)
def build_figure(crystal_name, energy_keV, theta_deg, azimuth_deg, L_mm,
                 hmax_3d, hmax_2d, show_spots, show_bragg, show_resonance,
                 y_min, y_max, z_min, z_max, img_bytes, kernel_size):

    preset = CRYSTAL_PRESETS[crystal_name]
    a   = preset["a"]
    V_I = preset["V_I"]
    Ki  = ki_magnitude(energy_keV)
    vt  = reduced_potential(V_I, energy_keV)
    L   = L_mm * 1e-3
    y_range = (y_min, y_max)
    z_range = (z_min, z_max)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("#111111")

    stats = {}

    # Background RHEED image
    if img_bytes is not None:
        import skimage.io as skio
        raw = skio.imread(io.BytesIO(img_bytes))
        filt = filter_rheed_image(raw, kernel_size)
        ax.imshow(filt, cmap="gray",
                  extent=[y_min, y_max, z_min, z_max],
                  aspect="auto", origin="lower", alpha=0.85)

    # Bragg spots
    if show_spots:
        spots = compute_bragg_spots(Ki, theta_deg, a, azimuth_deg, hmax=hmax_3d, L=L)
        visible = [(y, z) for y, z in spots
                   if y_range[0]<=y<=y_range[1] and z_range[0]<=z<=z_range[1]]
        if visible:
            ys, zs = zip(*visible)
            ax.scatter(ys, zs, color="lime", s=30, zorder=5)
        stats["spots"] = len(visible)

    # Bragg Kikuchi lines
    if show_bragg:
        n = 0
        for G in bulk_G_vecs(a, hmax_3d):
            py, pz = bragg_kikuchi_line(G, Ki, theta_deg, vt,
                                        L=L, y_range=y_range)
            mask = [z_range[0]<=z<=z_range[1] for z in pz]
            py2  = [y for y,m in zip(py,mask) if m]
            pz2  = [z for z,m in zip(pz,mask) if m]
            if len(py2) > 3:
                o = np.argsort(py2)
                ax.plot(np.array(py2)[o], np.array(pz2)[o],
                        color="dodgerblue", lw=0.7, alpha=0.75, zorder=3)
                n += 1
        stats["bragg_lines"] = n

    # Resonance lines
    if show_resonance:
        n = 0
        for g in surface_g_vecs(a, hmax_2d):
            py, pz = resonance_line(g, Ki, theta_deg, vt,
                                    L=L, y_range=y_range)
            mask = [z_range[0]<=z<=z_range[1] for z in pz]
            py2  = [y for y,m in zip(py,mask) if m]
            pz2  = [z for z,m in zip(pz,mask) if m]
            if len(py2) > 3:
                o = np.argsort(py2)
                ax.plot(np.array(py2)[o], np.array(pz2)[o],
                        color="tomato", lw=0.7, alpha=0.75, zorder=3)
                n += 1
        stats["resonance_lines"] = n

    # Legend
    handles = []
    if show_spots:
        handles.append(mpatches.Patch(color="lime",       label="Bragg spots"))
    if show_bragg:
        handles.append(mpatches.Patch(color="dodgerblue", label="Bragg Kikuchi lines"))
    if show_resonance:
        handles.append(mpatches.Patch(color="tomato",     label="Resonance lines"))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=9,
                  facecolor="#222222", labelcolor="white")

    ax.set_xlim(y_range)
    ax.set_ylim(z_range)
    ax.set_xlabel("Y  [mm]  (horizontal, perp beam)", color="white", fontsize=11)
    ax.set_ylabel("Z  [mm]  (above surface)",         color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.axhline(0, color="#555555", lw=0.5, ls="--")
    ax.text(y_min + 1, 0.5, "shadow edge", color="#888888", fontsize=7, va="bottom")
    ax.set_title(
        f"RHEED Kikuchi  --  {crystal_name} (001)  |  "
        f"E={energy_keV} keV   theta={theta_deg} deg   phi={azimuth_deg} deg   L={L_mm} mm",
        color="white", fontsize=9
    )
    fig.tight_layout()
    return fig, stats, Ki, vt

# ============================================================================
#  Streamlit UI
# ============================================================================

st.set_page_config(
    page_title="RHEED Kikuchi Simulator",
    page_icon="🔬",
    layout="wide",
)

st.title("RHEED Kikuchi Line Simulator")
st.caption("Bragg reflection & resonance scattering lines | Pawlak 2021 · Mitura 2024")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    crystal = st.selectbox("Crystal", list(CRYSTAL_PRESETS.keys()))

    preset_info = CRYSTAL_PRESETS[crystal]
    st.caption(f"a = {preset_info['a']*1e10:.3f} Å   |   V_I = {preset_info['V_I']} eV")

    st.divider()

    energy  = st.slider("Beam energy (keV)", 5.0, 100.0, 20.0, 0.5)
    theta   = st.slider("Glancing angle theta (deg)", 0.5, 10.0, 2.9, 0.1)
    azimuth = st.slider("Azimuth phi (deg)", -45.0, 45.0, 0.0, 0.5)
    L_mm    = st.slider("Screen distance L (mm)", 50.0, 500.0, 200.0, 10.0)

    st.divider()
    st.subheader("Reciprocal lattice")
    hmax_3d = st.slider("hmax  (3D, Bragg/Kikuchi)", 1, 6, 4)
    hmax_2d = st.slider("hmax  (2D, resonance)",     1, 6, 4)

    st.divider()
    st.subheader("Screen window (mm)")
    col1, col2 = st.columns(2)
    y_min = col1.number_input("Y min", value=-60, step=10)
    y_max = col2.number_input("Y max", value=60,  step=10)
    z_min = col1.number_input("Z min", value=0,   step=5)
    z_max = col2.number_input("Z max", value=80,  step=5)

    st.divider()
    st.subheader("Overlays")
    show_spots     = st.checkbox("Bragg spots",         value=True)
    show_bragg     = st.checkbox("Bragg Kikuchi lines", value=True)
    show_resonance = st.checkbox("Resonance lines",     value=True)

    st.divider()
    st.subheader("RHEED image (optional)")
    uploaded = st.file_uploader("Upload .tif / .png image", type=["tif","tiff","png","jpg"])
    kernel_size = st.slider("AHE kernel size (px)", 5, 50, 15, step=5,
                            help="Adaptive histogram equalization window")

# ── Main panel ───────────────────────────────────────────────────────────────
img_bytes = uploaded.read() if uploaded else None

z_spec = np.tan(np.radians(theta)) * L_mm

# Physics info bar
c1, c2, c3, c4 = st.columns(4)
Ki_val = ki_magnitude(energy)
vt_val = reduced_potential(preset_info["V_I"], energy)
c1.metric("Beam energy",    f"{energy} keV")
c2.metric("|Ki|",           f"{Ki_val:.3e} m⁻¹")
c3.metric("v_tilde",        f"{vt_val:.3e} m⁻²")
c4.metric("Specular Z_s",   f"{z_spec:.1f} mm")

st.divider()

with st.spinner("Computing Kikuchi pattern..."):
    fig, stats, Ki_val, vt_val = build_figure(
        crystal, energy, theta, azimuth, L_mm,
        hmax_3d, hmax_2d,
        show_spots, show_bragg, show_resonance,
        float(y_min), float(y_max), float(z_min), float(z_max),
        img_bytes, kernel_size,
    )

st.pyplot(fig, use_container_width=True)

# Stats row
if stats:
    s1, s2, s3 = st.columns(3)
    if "spots"         in stats: s1.info(f"**{stats['spots']}** Bragg spots")
    if "bragg_lines"   in stats: s2.info(f"**{stats['bragg_lines']}** Bragg Kikuchi lines")
    if "resonance_lines" in stats: s3.info(f"**{stats['resonance_lines']}** resonance lines")

# Download button
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
buf.seek(0)
st.download_button(
    label="Download PNG",
    data=buf,
    file_name=f"rheed_kikuchi_{crystal}_E{energy}keV_th{theta}deg.png",
    mime="image/png",
)

plt.close(fig)

st.divider()
with st.expander("About"):
    st.markdown("""
**RHEED Kikuchi Line Simulator**

Computes three types of features on the RHEED screen:

| Color | Feature | Equation |
|-------|---------|----------|
| 🟢 Green | Bragg diffraction spots | Ewald sphere + 2D periodicity |
| 🔵 Blue | Bragg reflection Kikuchi lines | Eq. 5 (Pawlak 2021) / Eq. 1 (Mitura 2024) |
| 🔴 Red | Resonance scattering lines | Eq. 8 (Pawlak 2021) / Eq. 3 (Mitura 2024) |

**Coordinate system:**  X = beam direction, Y = horizontal (⊥ beam), Z = surface normal (up).
The screen is vertical at distance L from the sample.

**Image filter** (when a RHEED image is uploaded):
- RDB: Remove Dynamic Background via Gaussian subtraction
- AHE: Adaptive Histogram Equalization (scikit-image)

**References:**
- Pawlak, Przybylski & Mitura, *Materials* **14**, 7077 (2021)
- Mitura, Pawlak & Przybylski, *Acta Cryst. A* **80**, 104–111 (2024)
""")
