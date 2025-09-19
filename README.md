# Two-Ring Magnets — B-field Viewer

A Streamlit app that visualizes the magnetic field of two coaxial ring magnets in air.
- 2D By heatmap with hover readouts (x, y, Gauss)
- 3D field cones with hover (x, y, z, |B| in Gauss)
- Adjustable geometry (inner/outer radius, thickness, clear gap) and quality settings

## Run locally
```bash
pip install streamlit numpy plotly
streamlit run app.py
