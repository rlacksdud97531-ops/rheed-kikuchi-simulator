"""
RHEED Kikuchi Line Simulator — Streamlit Web App
Layout:
  ① Theoretical Kikuchi pattern (top)
  ② Experimental RHEED image + line overlay (bottom)

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

# ============================================================================
#  Physical constants
# ============================================================================
hbar = 1.054571817e-34
m0   = 9.1093837015e-31
qe   = 1.602176634e-19
c    = 2.99792458e8

CRYSTAL_PRESETS = {
    "SrTiO3":  {"a": 3.905, "c": 3.905, "V_I": 15.08, "nu": 0.23, "group": "Substrate"},
    "LaAlO3":  {"a": 3.787, "c": 3.787, "V_I": 13.5,  "nu": 0.24, "group": "Substrate"},
    "MgO":     {"a": 4.211, "c": 4.211, "V_I": 13.0,  "nu": 0.18, "group": "Substrate"},
    "LSAT":    {"a": 3.868, "c": 3.868, "V_I": 13.8,  "nu": 0.24, "group": "Substrate"},
    "LSMO":    {"a": 3.876, "c": 3.876, "V_I": 14.0,  "nu": 0.25, "group": "Oxide film"},
    "BaTiO3":  {"a": 3.994, "c": 4.038, "V_I": 16.0,  "nu": 0.25, "group": "Oxide film"},
    "LaNiO3":  {"a": 3.838, "c": 3.838, "V_I": 14.5,  "nu": 0.25, "group": "Oxide film"},
    "SrRuO3":  {"a": 3.930, "c": 3.930, "V_I": 15.5,  "nu": 0.25, "group": "Oxide film"},
    "BiFeO3":  {"a": 3.965, "c": 3.965, "V_I": 15.0,  "nu": 0.25, "group": "Oxide film"},
    "Cu":      {"a": 3.615, "c": 3.615, "V_I": 12.0,  "nu": 0.34, "group": "Metal"},
    "Pt":      {"a": 3.924, "c": 3.924, "V_I": 18.0,  "nu": 0.38, "group": "Metal"},
    "Fe":      {"a": 2.870, "c": 2.870, "V_I": 11.5,  "nu": 0.29, "group": "Metal"},
}

SUBSTRATE_LIST = [k for k,v in CRYSTAL_PRESETS.items() if v["group"] == "Substrate"]
FILM_LIST      = [k for k,v in CRYSTAL_PRESETS.items() if v["group"] != "Substrate"]

# ============================================================================
#  Physics helpers
# ============================================================================

def ki_magnitude(energy_keV):
    E_J = energy_keV * 1e3 * qe
    p   = np.sqrt(2.0 * m0 * E_J * (1.0 + E_J / (2.0 * m0 * c**2)))
    return p / hbar

def reduced_potential(V_I_eV, energy_keV):
    U   = energy_keV * 1e3
    rel = 1.0 + (qe * U) / (m0 * c**2)
    return -rel * (2.0 * m0 / hbar**2) * (V_I_eV * qe)

def ki_vec(Ki, theta_deg, azimuth_deg=0.0):
    th  = np.radians(theta_deg)
    phi = np.radians(azimuth_deg)
    K_iX = Ki * np.cos(th) * np.cos(phi)
    K_iY = Ki * np.cos(th) * np.sin(phi)
    K_iZ = -Ki * np.sin(th)
    return K_iX, K_iY, K_iZ

def strained_c(a_bulk, c_bulk, a_in, nu):
    eps = (a_in - a_bulk) / a_bulk
    return c_bulk * (1.0 - 2.0 * nu / (1.0 - nu) * eps)

def surface_g_vecs(a_in_m, hmax):
    b = 2.0 * np.pi / a_in_m
    return np.array([[h*b, k*b]
                     for h in range(-hmax, hmax+1)
                     for k in range(-hmax, hmax+1)
                     if not (h==0 and k==0)])

def bulk_G_vecs(a_in_m, c_out_m, hmax):
    b_in  = 2.0 * np.pi / a_in_m
    b_out = 2.0 * np.pi / c_out_m
    return np.array([[h*b_in, k*b_in, l*b_out]
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
    if Gp2 == 0: return []
    disc = (R*GY)**2 - 4*Gp2*(R**2/4.0 - GX**2*q)
    if disc < 0:  return []
    sq = np.sqrt(disc)
    return [(R*GY + sq) / (2*Gp2), (R*GY - sq) / (2*Gp2)]

def compute_bragg_spots(Ki, theta_deg, a_in_m, azimuth_deg, hmax, L):
    K_iX, K_iY, _ = ki_vec(Ki, theta_deg, azimuth_deg)
    b = 2.0 * np.pi / a_in_m
    spots = []
    for h in range(-hmax, hmax+1):
        for k in range(-hmax, hmax+1):
            K_fX = K_iX + h*b;  K_fY = K_iY + k*b
            val  = Ki**2 - K_fX**2 - K_fY**2
            if val <= 0 or K_fX <= 0: continue
            spots.append(((K_fY/K_fX)*L*1e3, (np.sqrt(val)/K_fX)*L*1e3))
    return spots

def bragg_kikuchi_line(G, Ki, vt, L, y_range, npts=500):
    GX, GY, GZ = G
    G2 = GX**2 + GY**2 + GZ**2
    py, pz = [], []
    if abs(GX) < 1 and abs(GY) < 1:
        if abs(GZ) < 1: return py, pz
        K_fZ2 = (GZ/2)**2 + vt
        if K_fZ2 <= 0: return py, pz
        K_fZ_f = np.sqrt(K_fZ2)
        qm = Ki**2 - K_fZ_f**2
        if qm <= 0: return py, pz
        Km = np.sqrt(qm)
        for K_fY in np.linspace(-Km*0.999, Km*0.999, npts):
            K_fX2 = qm - K_fY**2
            if K_fX2 > 0: _add_point(np.sqrt(K_fX2), K_fY, K_fZ_f, L, y_range, py, pz)
        return py, pz
    for K_fZ in np.linspace(1e7, Ki*0.98, npts):
        inner = K_fZ**2 - vt
        if inner <= 0: continue
        R = G2 - 2*np.sqrt(inner)*GZ
        q = Ki**2 - K_fZ**2
        if q <= 0: continue
        if abs(GX) < 1:
            K_fY_v = R / (2*GY)
            K_fX2  = q - K_fY_v**2
            if K_fX2 > 0: _add_point(np.sqrt(K_fX2), K_fY_v, K_fZ, L, y_range, py, pz)
        else:
            for K_fY in _quadratic_K_fy(R, GX, GY, q):
                _add_point((R - 2*K_fY*GY)/(2*GX), K_fY, K_fZ, L, y_range, py, pz)
    return py, pz

def resonance_line(g, Ki, vt, L, y_range, npts=500):
    gX, gY = g
    g2 = gX**2 + gY**2
    py, pz = [], []
    for K_fZ in np.linspace(1e7, Ki*0.98, npts):
        R = g2 - K_fZ**2 + vt
        q = Ki**2 - K_fZ**2
        if q <= 0: continue
        if abs(gX) < 1:
            if abs(gY) < 1: continue
            K_fY_v = R / (2*gY)
            K_fX2  = q - K_fY_v**2
            if K_fX2 > 0: _add_point(np.sqrt(K_fX2), K_fY_v, K_fZ, L, y_range, py, pz)
        else:
            for K_fY in _quadratic_K_fy(R, gX, gY, q):
                _add_point((R - 2*K_fY*gY)/(2*gX), K_fY, K_fZ, L, y_range, py, pz)
    return py, pz

# ============================================================================
#  RDB + AHE filter
# ============================================================================

def filter_rheed_image(img_array, kernel_size=15):
    from skimage import filters, exposure
    raw = img_array.astype(np.float32)
    if raw.ndim == 3: raw = raw.mean(axis=2)
    raw /= raw.max() + 1e-9
    rdb  = raw - filters.gaussian(raw, sigma=max(raw.shape)/8.0)
    rdb -= rdb.min();  rdb /= rdb.max() + 1e-9
    return exposure.equalize_adapthist(rdb, kernel_size=kernel_size,
                                       clip_limit=0.03).astype(np.float32)

# ============================================================================
#  Draw Kikuchi lines onto an axes object (shared helper)
# ============================================================================

def _draw_lines(ax, Ki, vt, a_in, c_out, L, y_range, z_range,
                hmax_3d, hmax_2d, show_spots, show_bragg, show_resonance,
                theta_deg, azimuth_deg,
                spot_color="lime", bragg_color="dodgerblue", res_color="tomato",
                lw=0.7, alpha=0.75):
    stats = {}

    if show_spots:
        spots   = compute_bragg_spots(Ki, theta_deg, a_in, azimuth_deg, hmax_3d, L)
        visible = [(y,z) for y,z in spots
                   if y_range[0]<=y<=y_range[1] and z_range[0]<=z<=z_range[1]]
        if visible:
            ys, zs = zip(*visible)
            ax.scatter(ys, zs, color=spot_color, s=30, zorder=5)
        stats["spots"] = len(visible)

    if show_bragg:
        n = 0
        for G in bulk_G_vecs(a_in, c_out, hmax_3d):
            py, pz = bragg_kikuchi_line(G, Ki, vt, L, y_range)
            mask = [z_range[0]<=z<=z_range[1] for z in pz]
            py2  = [y for y,m in zip(py,mask) if m]
            pz2  = [z for z,m in zip(pz,mask) if m]
            if len(py2) > 3:
                o = np.argsort(py2)
                ax.plot(np.array(py2)[o], np.array(pz2)[o],
                        color=bragg_color, lw=lw, alpha=alpha, zorder=3)
                n += 1
        stats["bragg_lines"] = n

    if show_resonance:
        n = 0
        for g in surface_g_vecs(a_in, hmax_2d):
            py, pz = resonance_line(g, Ki, vt, L, y_range)
            mask = [z_range[0]<=z<=z_range[1] for z in pz]
            py2  = [y for y,m in zip(py,mask) if m]
            pz2  = [z for z,m in zip(pz,mask) if m]
            if len(py2) > 3:
                o = np.argsort(py2)
                ax.plot(np.array(py2)[o], np.array(pz2)[o],
                        color=res_color, lw=lw, alpha=alpha, zorder=3)
                n += 1
        stats["resonance_lines"] = n

    return stats

def _finish_axes(ax, y_range, z_range, title, y_min):
    handles = [
        mpatches.Patch(color="lime",       label="Bragg spots"),
        mpatches.Patch(color="dodgerblue", label="Bragg Kikuchi lines"),
        mpatches.Patch(color="tomato",     label="Resonance lines"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              facecolor="#222222", labelcolor="white")
    ax.set_xlim(y_range);  ax.set_ylim(z_range);  ax.invert_yaxis()
    ax.set_xlabel("Y  [mm]  (horizontal, perp beam)",     color="white", fontsize=10)
    ax.set_ylabel("Z  [mm]  (distance from shadow edge)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#555555")
    ax.axhline(0, color="#555555", lw=0.5, ls="--")
    ax.text(y_min+1, 0.5, "shadow edge", color="#888888", fontsize=7, va="top")
    ax.set_title(title, color="white", fontsize=9)

# ============================================================================
#  Figure builders  (cached)
# ============================================================================

@st.cache_data(show_spinner=False)
def build_theory_figure(a_in_A, c_out_A, V_I, label,
                        energy_keV, theta_deg, azimuth_deg, L_mm,
                        hmax_3d, hmax_2d,
                        show_spots, show_bragg, show_resonance,
                        y_min, y_max, z_min, z_max):
    a_in  = a_in_A  * 1e-10
    c_out = c_out_A * 1e-10
    Ki    = ki_magnitude(energy_keV)
    vt    = reduced_potential(V_I, energy_keV)
    L     = L_mm * 1e-3

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("black");  fig.patch.set_facecolor("#111111")

    stats = _draw_lines(ax, Ki, vt, a_in, c_out, L,
                        (y_min, y_max), (z_min, z_max),
                        hmax_3d, hmax_2d,
                        show_spots, show_bragg, show_resonance,
                        theta_deg, azimuth_deg)

    _finish_axes(ax, (y_min,y_max), (z_min,z_max),
                 f"Theoretical  |  {label}  |  E={energy_keV} keV  "
                 f"theta={theta_deg}deg  phi={azimuth_deg}deg  L={L_mm}mm",
                 y_min)
    fig.tight_layout()
    return fig, stats


@st.cache_data(show_spinner=False)
def build_overlay_figure(img_bytes, kernel_size,
                         a_in_A, c_out_A, V_I,
                         energy_keV, theta_deg, azimuth_deg, L_mm,
                         hmax_3d, hmax_2d,
                         show_spots, show_bragg, show_resonance,
                         y_min, y_max, z_min, z_max):
    import skimage.io as skio

    a_in  = a_in_A  * 1e-10
    c_out = c_out_A * 1e-10
    Ki    = ki_magnitude(energy_keV)
    vt    = reduced_potential(V_I, energy_keV)
    L     = L_mm * 1e-3

    raw   = skio.imread(io.BytesIO(img_bytes))
    filt  = filter_rheed_image(raw, kernel_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#111111")

    # ── Left: filtered image only ────────────────────────────────────────────
    ax_img = axes[0]
    ax_img.set_facecolor("black")
    ax_img.imshow(filt, cmap="gray", aspect="auto", origin="upper")
    ax_img.set_title("Filtered RHEED image  (RDB + AHE)",
                     color="white", fontsize=9)
    ax_img.tick_params(colors="white")
    ax_img.set_xlabel("pixel X", color="white", fontsize=10)
    ax_img.set_ylabel("pixel Y", color="white", fontsize=10)
    for sp in ax_img.spines.values(): sp.set_edgecolor("#555555")

    # ── Right: filtered image + Kikuchi lines overlaid ───────────────────────
    ax_ov = axes[1]
    ax_ov.set_facecolor("black")
    ax_ov.imshow(filt, cmap="gray",
                 extent=[y_min, y_max, z_min, z_max],
                 aspect="auto", origin="lower", alpha=0.9)

    stats = _draw_lines(ax_ov, Ki, vt, a_in, c_out, L,
                        (y_min, y_max), (z_min, z_max),
                        hmax_3d, hmax_2d,
                        show_spots, show_bragg, show_resonance,
                        theta_deg, azimuth_deg,
                        lw=0.8, alpha=0.85)

    _finish_axes(ax_ov, (y_min,y_max), (z_min,z_max),
                 "Filtered image + Kikuchi line overlay", y_min)

    fig.tight_layout()
    return fig, stats

# ============================================================================
#  Streamlit UI
# ============================================================================

st.set_page_config(page_title="RHEED Kikuchi Simulator",
                   page_icon="🔬", layout="wide")

st.title("🔬 RHEED Kikuchi Line Simulator")
st.caption("Pawlak 2021 · Mitura 2024  |  Bragg spots · Bragg Kikuchi lines · Resonance lines")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Crystal")
    mode = st.radio("Mode", ["Substrate", "Epitaxial Film"], horizontal=True)

    if mode == "Substrate":
        substrate = st.selectbox("Substrate", list(CRYSTAL_PRESETS.keys()),
                                 index=0)
        p       = CRYSTAL_PRESETS[substrate]
        a_in_A  = p["a"];  c_out_A = p["c"];  V_I = p["V_I"]
        label   = f"{substrate} (001)"
        st.caption(f"a = {a_in_A:.3f} Å  |  c = {c_out_A:.3f} Å  |  V_I = {V_I} eV")
    else:
        c1, c2  = st.columns(2)
        substrate = c1.selectbox("Substrate", SUBSTRATE_LIST)
        film      = c2.selectbox("Film",      FILM_LIST)
        p_sub   = CRYSTAL_PRESETS[substrate]
        p_film  = CRYSTAL_PRESETS[film]
        a_in_A  = p_sub["a"]
        c_auto  = strained_c(p_film["a"], p_film["c"], a_in_A, p_film["nu"])
        V_I     = p_film["V_I"]
        label   = f"{film}/{substrate} (001)"
        strain_pct = (a_in_A - p_film["a"]) / p_film["a"] * 100
        st.info(f"in-plane a = **{a_in_A:.3f} Å** (substrate)\n\n"
                f"Film bulk a = {p_film['a']:.3f} Å  →  strain **{strain_pct:+.2f}%**")
        s_type = st.radio("Strain", ["Fully strained", "Fully relaxed", "Manual c"])
        if   s_type == "Fully strained": c_out_A = c_auto
        elif s_type == "Fully relaxed":  c_out_A = p_film["c"]
        else: c_out_A = st.slider("c out-of-plane (Å)", 3.5, 4.5,
                                   float(round(c_auto,3)), 0.001, format="%.3f")
        st.caption(f"c_out = {c_out_A:.3f} Å  |  V_I = {V_I} eV")

    st.divider()
    st.subheader("Beam & Geometry")
    energy  = st.slider("Energy (keV)",      5.0, 100.0, 20.0, 0.5)
    theta   = st.slider("theta (deg)",       0.5,  10.0,  2.9, 0.1)
    azimuth = st.slider("phi (deg)",       -45.0,  45.0,  0.0, 0.5)
    L_mm    = st.slider("Screen dist L (mm)", 50.0, 500.0, 200.0, 10.0)

    st.divider()
    st.subheader("Reciprocal lattice")
    hmax_3d = st.slider("hmax 3D (Bragg/Kikuchi)", 1, 6, 4)
    hmax_2d = st.slider("hmax 2D (resonance)",     1, 6, 4)

    st.divider()
    st.subheader("Screen window (mm)")
    col1, col2 = st.columns(2)
    y_min = col1.number_input("Y min", value=-60, step=10)
    y_max = col2.number_input("Y max", value= 60, step=10)
    z_min = col1.number_input("Z min", value=  0, step= 5)
    z_max = col2.number_input("Z max", value= 80, step= 5)

    st.divider()
    st.subheader("Overlays")
    show_spots     = st.checkbox("Bragg spots",         value=True)
    show_bragg     = st.checkbox("Bragg Kikuchi lines", value=True)
    show_resonance = st.checkbox("Resonance lines",     value=True)

# ============================================================================
#  ① THEORETICAL PATTERN
# ============================================================================
st.subheader("① Theoretical Kikuchi Pattern")

Ki_val = ki_magnitude(energy)
vt_val = reduced_potential(V_I, energy)
z_spec = np.tan(np.radians(theta)) * L_mm

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Energy",       f"{energy} keV")
m2.metric("|Ki|",         f"{Ki_val:.3e} m⁻¹")
m3.metric("v_tilde",      f"{vt_val:.3e} m⁻²")
m4.metric("Specular Z_s", f"{z_spec:.1f} mm")
m5.metric("a / c",        f"{a_in_A:.3f} / {c_out_A:.3f} Å")

with st.spinner("Computing theoretical pattern..."):
    fig_th, stats_th = build_theory_figure(
        a_in_A, c_out_A, V_I, label,
        energy, theta, azimuth, L_mm,
        hmax_3d, hmax_2d,
        show_spots, show_bragg, show_resonance,
        float(y_min), float(y_max), float(z_min), float(z_max),
    )

st.pyplot(fig_th, use_container_width=True)

# stats
if stats_th:
    s1, s2, s3 = st.columns(3)
    if "spots"           in stats_th: s1.info(f"**{stats_th['spots']}** Bragg spots")
    if "bragg_lines"     in stats_th: s2.info(f"**{stats_th['bragg_lines']}** Bragg Kikuchi lines")
    if "resonance_lines" in stats_th: s3.info(f"**{stats_th['resonance_lines']}** resonance lines")

# download
buf = io.BytesIO()
fig_th.savefig(buf, format="png", dpi=150, bbox_inches="tight",
               facecolor=fig_th.get_facecolor())
buf.seek(0)
st.download_button("Download theoretical PNG", buf,
                   file_name=f"theory_{label.replace('/','_')}_E{energy}keV.png",
                   mime="image/png")
plt.close(fig_th)

# ============================================================================
#  ② EXPERIMENTAL IMAGE  +  OVERLAY
# ============================================================================
st.divider()
st.subheader("② Experimental RHEED Image + Kikuchi Line Overlay")

col_up, col_kern = st.columns([3, 1])
uploaded    = col_up.file_uploader(
    "Upload RHEED image (.tif / .png / .jpg)",
    type=["tif","tiff","png","jpg"])
kernel_size = col_kern.slider("AHE kernel (px)", 5, 50, 15, step=5,
                              help="Adaptive histogram equalization window size")

if uploaded is None:
    st.info("Upload a RHEED image above to see the filtered result and Kikuchi line overlay.")
else:
    img_bytes = uploaded.read()
    with st.spinner("Filtering image and computing overlay..."):
        fig_ov, stats_ov = build_overlay_figure(
            img_bytes, kernel_size,
            a_in_A, c_out_A, V_I,
            energy, theta, azimuth, L_mm,
            hmax_3d, hmax_2d,
            show_spots, show_bragg, show_resonance,
            float(y_min), float(y_max), float(z_min), float(z_max),
        )

    st.pyplot(fig_ov, use_container_width=True)

    st.caption(
        "**Left**: filtered image (RDB = remove dynamic background, AHE = adaptive histogram equalization)  "
        "**Right**: same image with Kikuchi lines overlaid using the same mm coordinate window as the theoretical plot."
    )

    if stats_ov:
        s1, s2, s3 = st.columns(3)
        if "spots"           in stats_ov: s1.info(f"**{stats_ov['spots']}** Bragg spots")
        if "bragg_lines"     in stats_ov: s2.info(f"**{stats_ov['bragg_lines']}** Bragg Kikuchi lines")
        if "resonance_lines" in stats_ov: s3.info(f"**{stats_ov['resonance_lines']}** resonance lines")

    buf2 = io.BytesIO()
    fig_ov.savefig(buf2, format="png", dpi=150, bbox_inches="tight",
                   facecolor=fig_ov.get_facecolor())
    buf2.seek(0)
    st.download_button("Download overlay PNG", buf2,
                       file_name=f"overlay_{label.replace('/','_')}_E{energy}keV.png",
                       mime="image/png")
    plt.close(fig_ov)

# ============================================================================
#  About
# ============================================================================
st.divider()
with st.expander("About"):
    st.markdown("""
**RHEED Kikuchi Line Simulator**

| Color | Feature | Equation |
|-------|---------|----------|
| 🟢 Green | Bragg diffraction spots | Ewald sphere + 2D periodicity |
| 🔵 Blue | Bragg Kikuchi lines | Eq. 5 (Pawlak 2021) / Eq. 1 (Mitura 2024) |
| 🔴 Red | Resonance scattering lines | Eq. 8 (Pawlak 2021) / Eq. 3 (Mitura 2024) |

**Crystal support:** Cubic (a=c) and tetragonal (a≠c).
**Film mode:** in-plane a locked to substrate; c_out from Poisson ratio or manual.

**References:**
- Pawlak, Przybylski & Mitura, *Materials* **14**, 7077 (2021)
- Mitura, Pawlak & Przybylski, *Acta Cryst. A* **80**, 104–111 (2024)
""")
