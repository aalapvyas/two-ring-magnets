import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff

# -----------------------------
# Physics helpers (same model as before)
# -----------------------------
mu0 = 4e-7 * np.pi  # H/m

def grade_to_Br_T(grade: str) -> float:
    g = grade.strip().upper().replace(' ', '')
    if g in ('N35','N35M','N35H'):   return 1.17
    if g in ('N38','N38H'):          return 1.23
    if g in ('N42','N42H','N42SH'):  return 1.32
    if g in ('N45','N45H'):          return 1.38
    if g in ('N48','N48H'):          return 1.42
    if g in ('N52','N52H'):          return 1.45
    return 1.30  # conservative fallback

def magnetization_from_grade(grade: str) -> float:
    Br = grade_to_Br_T(grade)
    return Br / mu0  # A/m

def loop_field_dB(r_obs, a, I, y0, nphi=160):
    """Field (Tesla) of one circular current loop centered at (0,y0,0) in x–z plane."""
    r_obs = np.atleast_2d(r_obs)  # (N,3)
    phis = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    dphi = 2*np.pi / nphi
    xs = a*np.cos(phis); zs = a*np.sin(phis); ys = np.full_like(xs, y0)
    xs2 = a*np.cos(phis + dphi); zs2 = a*np.sin(phis + dphi)
    dl = np.stack([xs2 - xs, np.zeros_like(xs), zs2 - zs], axis=1)  # (nphi,3)
    Idl = I * dl
    r_src = np.stack([xs, ys, zs], axis=1)

    B = np.zeros((r_obs.shape[0], 3))
    for k in range(nphi):
        R = r_obs - r_src[k]                          # (N,3)
        R2 = np.einsum('ij,ij->i', R, R)              # (N,)
        mask = R2 > 1e-20
        if not np.any(mask): 
            continue
        Rm = np.sqrt(R2[mask])
        Rhat = R[mask] / Rm[:,None]
        contrib = np.zeros_like(R)
        contrib[mask] = np.cross(Idl[k], Rhat) / (Rm**2)[:,None]
        B += (mu0/(4*np.pi)) * contrib
    return B  # Tesla

def ring_magnet_loops(M, rin, rout, thickness, y_center, nz_slices=24):
    """Return list of (radius a, current I, y0) for one axially-magnetized ring."""
    if nz_slices <= 1:
        ys = np.array([y_center]); dz = thickness
    else:
        ys = np.linspace(y_center - thickness/2, y_center + thickness/2, nz_slices)
        dz = thickness / (nz_slices - 1)
    K = M
    loops = []
    for y0 in ys:
        if rout > 0: loops.append((rout,  +K*dz, y0))
        if rin  > 0: loops.append((rin,   -K*dz, y0))
    return loops

def magnet_pair_loops(grade, rin, rout, thickness, gap_clear, nz_slices=24):
    """Build loops for two identical rings, both magnetized +y.
    gap_clear = face-to-face air gap between rings (not center-to-center).
    Centers are at y=±(gap_clear/2 + thickness/2)."""
    M = magnetization_from_grade(grade)
    y1 = -(gap_clear/2 + thickness/2)
    y2 = +(gap_clear/2 + thickness/2)
    return ring_magnet_loops(M, rin, rout, thickness, y1, nz_slices) + \
           ring_magnet_loops(M, rin, rout, thickness, y2, nz_slices)

def field_from_loops(points, loops, nphi=160):
    points = np.atleast_2d(points)
    B = np.zeros_like(points)
    for (a, I, y0) in loops:
        B += loop_field_dB(points, a, I, y0, nphi=nphi)
    return B

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Two-Ring Magnet Field (By)", layout="wide")
st.title("Two-Ring Neodymium Magnet Field Viewer — v0.4")
st.caption("Axially magnetized rings • air gap only • Biot–Savart with equivalent surface currents • 2D & 3D with hover readouts")

with st.sidebar:
    st.header("Controls")
    grade = st.selectbox("Magnet grade", ["N35", "N42", "N52"], index=2)

    col_r = st.columns(2)
    rin_mm  = col_r[0].number_input("Inner radius (mm)", min_value=0.0, value=20.0, step=1.0, format="%.1f")
    rout_mm = col_r[1].number_input("Outer radius (mm)", min_value=0.1, value=35.0, step=1.0, format="%.1f")

    thickness_mm = st.number_input("Thickness (mm)", min_value=0.1, value=10.0, step=0.5, format="%.1f")
    gap_clear_mm = st.number_input("Clear air gap between faces (mm)", min_value=0.1, value=30.0, step=1.0, format="%.1f")

    vectors_on = st.toggle("Show vectors (2D slice)", value=False)

    quality = st.select_slider("Quality / Speed", options=["Fast", "Balanced", "High"])
    if quality == "Fast":
        nphi, nz_slices, nx, ny = 64, 12, 81, 101
        nx3d = ny3d = nz3d = 9
    elif quality == "Balanced":
        nphi, nz_slices, nx, ny = 120, 24, 121, 141
        nx3d = ny3d = 11; nz3d = 9
    else:  # High
        nphi, nz_slices, nx, ny = 200, 36, 161, 181
        nx3d = ny3d = 13; nz3d = 11

    span_mm = st.slider("Plot span (half-width, mm)", min_value=30, max_value=80, value=60, step=5)

    st.markdown("---")
    st.subheader("3D Vector Options")
    vec3d_size = st.slider("Cone size (scene units)", min_value=0.05, max_value=0.80, value=0.18, step=0.01, help="Smaller = shorter arrows")
    vec3d_density = st.select_slider("Density", options=["Sparse", "Normal", "Dense"], value="Normal")

    st.subheader("2D Vector Options")
    vec2d_len_mm = st.slider("Arrow length (mm)", min_value=1.0, max_value=15.0, value=6.0, step=0.5)

    compute = st.button("Compute / Update", type="primary")

# Validate inputs
if rin_mm >= rout_mm:
    st.error("Inner radius must be **less** than outer radius.")
    st.stop()

# Convert to meters
rin = rin_mm * 1e-3
rout = rout_mm * 1e-3
thickness = thickness_mm * 1e-3
gap_clear = gap_clear_mm * 1e-3
span = span_mm * 1e-3

# Build model
loops = magnet_pair_loops(grade, rin, rout, thickness, gap_clear, nz_slices=nz_slices)

# Tabs for 2D and 3D
tab2d, tab3d = st.tabs(["🔹 2D Slice (z=0)", "🟣 3D View"])

# ---------- 2D (Plotly Heatmap + optional quiver) ----------

def compute_2d_and_plot():
    xs = np.linspace(-span, span, nx)
    ys = np.linspace(-span, span, ny)  # y is the axis
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Z = np.zeros_like(X)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    B = field_from_loops(pts, loops, nphi=nphi)
    By = B[:,1].reshape(Y.shape) * 1e4  # Gauss

    x_mm = xs*1e3; y_mm = ys*1e3
    heat = go.Heatmap(
        x=x_mm, y=y_mm, z=By,
        coloraxis='coloraxis',
        hovertemplate="x: %{x:.2f} mm\ny: %{y:.2f} mm\nB<sub>y</sub>: %{z:.2f} G<extra></extra>",
    )

    fig = go.Figure(data=[heat])
    fig.update_layout(
        coloraxis={'colorscale':'Viridis', 'colorbar':{'title':'B<sub>y</sub> (G)'}},
        xaxis_title='x (mm)', yaxis_title='y (mm) [axial]',
        yaxis_scaleanchor='x', yaxis_scaleratio=1,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"By (z=0) | Grade={grade}, t={thickness_mm:.1f} mm, rin={rin_mm:.1f} mm, rout={rout_mm:.1f} mm, gap(clear)={gap_clear_mm:.1f} mm | nphi={nphi}, nz={nz_slices}"
    )

    # Magnet outlines in this plane (as dashed lines)
    y1 = -(gap_clear/2 + thickness/2); y2 = +(gap_clear/2 + thickness/2)
    for ycen in (y1, y2):
        for r in (rin, rout):
            fig.add_shape(type='line', x0= r*1e3, x1= r*1e3, y0=(ycen-thickness/2)*1e3, y1=(ycen+thickness/2)*1e3,
                          line=dict(color='black', width=1, dash='dash'))
            fig.add_shape(type='line', x0=-r*1e3, x1=-r*1e3, y0=(ycen-thickness/2)*1e3, y1=(ycen+thickness/2)*1e3,
                          line=dict(color='black', width=1, dash='dash'))

    if vectors_on:
        # Downsample and build normalized arrows with fixed screen-length (in mm units)
        step = max(1, int(min(nx, ny)//18))
        Xs = X[::step, ::step]; Ys = Y[::step, ::step]
        pts_s = np.stack([Xs.ravel(), Ys.ravel(), np.zeros_like(Xs).ravel()], axis=1)
        Bs = field_from_loops(pts_s, loops, nphi=max(64, nphi//2)) * 1e4  # Gauss
        Bxs = Bs[:,0].reshape(Xs.shape)
        Bys = Bs[:,1].reshape(Xs.shape)
        norm = np.hypot(Bxs, Bys); norm = np.where(norm==0, 1.0, norm)
        U = (Bxs / norm) * (vec2d_len_mm / 1000.0)  # meters
        V = (Bys / norm) * (vec2d_len_mm / 1000.0)
        Xs_mm = Xs*1e3; Ys_mm = Ys*1e3
        qfig = ff.create_quiver(Xs_mm, Ys_mm, U*1e3, V*1e3, scale=1, arrow_scale=0.3, name='Vectors')
        # Suppress hover for arrows so heatmap hover shows By values
        for tr in qfig.data:
            tr.update(hoverinfo='skip', line=dict(width=1.2, color='rgba(255,255,255,0.9)'))
            fig.add_trace(tr)

    st.plotly_chart(fig, use_container_width=True)

    # On-axis readout in 20–43 mm window
    y_probe = np.linspace(- (gap_clear/2 - 1e-6), + (gap_clear/2 - 1e-6), 401)
    pts_line = np.stack([np.zeros_like(y_probe), y_probe, np.zeros_like(y_probe)], axis=1)
    B_line = field_from_loops(pts_line, loops, nphi=nphi)
    By_line_G = B_line[:,1]*1e4
    dmm = np.abs(y_probe)*1e3
    mask = (dmm >= 20) & (dmm <= 43)
    if np.any(mask):
        st.write("**On-axis (|y| = 20–43 mm) — B<sub>y</sub> (Gauss):**")
        rows = [f"|y| = {d:5.1f} mm → {b:9.2f} G" for d,b in zip(dmm[mask], By_line_G[mask])]
        st.code("\n".join(rows))

# ---------- 3D (Plotly Cones + ring surfaces) ----------

def ring_mesh(r_in, r_out, y_center, t, n_theta=64, n_rad=8):
    theta = np.linspace(0, 2*np.pi, n_theta)
    r = np.linspace(r_in, r_out, n_rad)
    Th, R = np.meshgrid(theta, r, indexing='xy')
    X = R*np.cos(Th); Z = R*np.sin(Th)
    Y_top = np.full_like(X, y_center + t/2)
    Y_bot = np.full_like(X, y_center - t/2)
    return (X, Y_top, Z), (X, Y_bot, Z)

def compute_3d_and_plot():
    # Grid for cones
    nx3 = nx3d; ny3 = ny3d; nz3 = nz3d
    xs = np.linspace(-span, span, nx3)
    ys = np.linspace(-span, span, ny3)
    zs = np.linspace(-span, span, nz3)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='xy')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    B = field_from_loops(pts, loops, nphi=nphi) * 1e4  # Gauss
    Bx = B[:,0].reshape(X.shape)
    By = B[:,1].reshape(X.shape)
    Bz = B[:,2].reshape(X.shape)

    fig = go.Figure()

    # Magnet translucent surfaces (top/bottom annular caps)
    y1 = -(gap_clear/2 + thickness/2); y2 = +(gap_clear/2 + thickness/2)
    for yc, color in [(y1, 'rgba(200,60,60,0.30)'), (y2, 'rgba(60,60,200,0.30)')]:
        (Xt, Yt, Zt), (Xb, Yb, Zb) = ring_mesh(rin, rout, yc, thickness, n_theta=100, n_rad=10)
        fig.add_surface(x=Xt, y=Yt, z=Zt, showscale=False, opacity=0.35, colorscale=[[0, color],[1, color]], hoverinfo='skip')
        fig.add_surface(x=Xb, y=Yb, z=Zb, showscale=False, opacity=0.35, colorscale=[[0, color],[1, color]], hoverinfo='skip')
        th = np.linspace(0, 2*np.pi, 120)
        for r, colr in [(rin, color), (rout, color)]:
            xc = r*np.cos(th); zc = r*np.sin(th)
            fig.add_trace(go.Scatter3d(x=xc, y=np.full_like(xc, yc - thickness/2), z=zc,
                                       mode='lines', line=dict(color=colr, width=3), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter3d(x=xc, y=np.full_like(xc, yc + thickness/2), z=zc,
                                       mode='lines', line=dict(color=colr, width=3), showlegend=False, hoverinfo='skip'))

    # Field cones — adjustable density & size
    stride = {"Sparse": 3, "Normal": 2, "Dense": 1}[vec3d_density]
    Xq = X[::stride, ::stride, ::stride]
    Yq = Y[::stride, ::stride, ::stride]
    Zq = Z[::stride, ::stride, ::stride]
    Bxq = Bx[::stride, ::stride, ::stride]
    Byq = By[::stride, ::stride, ::stride]
    Bzq = Bz[::stride, ::stride, ::stride]
    mag = np.sqrt(Bxq**2 + Byq**2 + Bzq**2) + 1e-12

    # Cones use direction (u,v,w). We'll normalize to unit and use absolute sizeref for length.
    u = (Bxq / mag).flatten(); v = (Byq / mag).flatten(); w = (Bzq / mag).flatten()
    x = Xq.flatten(); y = Yq.flatten(); z = Zq.flatten()
    custom = np.stack([x*1e3, y*1e3, z*1e3, mag.flatten()], axis=1)  # mm, Gauss

    fig.add_trace(go.Cone(
        x=x, y=y, z=z, u=u, v=v, w=w,
        sizemode="absolute", sizeref=vec3d_size, anchor="tail",
        colorscale="Blues", showscale=False, opacity=0.85,
        customdata=custom,
        hovertemplate=(
            "x: %{customdata[0]:.2f} mm\n" +
            "y: %{customdata[1]:.2f} mm\n" +
            "z: %{customdata[2]:.2f} mm\n" +
            "|B|: %{customdata[3]:.2f} G<extra></extra>"
        )
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m) [axial]", zaxis_title="z (m)",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"3D Field & Magnets | Grade={grade}, t={thickness_mm:.1f} mm, rin={rin_mm:.1f}, rout={rout_mm:.1f}, gap(clear)={gap_clear_mm:.1f} mm"
    )
    st.plotly_chart(fig, use_container_width=True)

# Only (re)compute on button press to keep it snappy
if compute:
    with tab2d:
        compute_2d_and_plot()
    with tab3d:
        compute_3d_and_plot()
else:
    st.info("Set your knobs in the sidebar, then click **Compute / Update**.")
