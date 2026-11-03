"""The Visual showcase mode: a black hole's shadow and lensing, an orbit on the embedding surface, and the Ising critical point
(modules/showcase.py). Each picture is shown next to the number that says it is right. No language model."""
import io
import math

import streamlit as st

from modules import gpu
from modules import showcase as sc
from modules import showcase_plots as sp


def _png(rgb) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _lens_tab() -> None:
    st.caption("Light rays are traced backwards from the observer through the Schwarzschild geometry, one per pixel. A ray that falls "
               "into the hole leaves a black pixel (the shadow); one that escapes shows the part of a procedural starfield it came "
               "from, bent by the hole; the gold ring is light that has orbited the photon sphere. Units G = c = 1, lengths in M.")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Image size (px)", 120, 480, 240, 40, key="showcase_lens_n")
    fov = c2.slider("Field of view (degrees)", 20, 90, 36, 2, key="showcase_lens_fov")
    dist = c3.slider("Observer distance (M)", 8, 200, 30, 2, key="showcase_lens_d")
    c4, c5 = st.columns(2)
    edge = c4.checkbox("Mark the exact shadow edge", True, key="showcase_lens_edge")
    compare = c5.checkbox("Also show the same sky with no black hole", False, key="showcase_lens_compare")
    if st.button("🕳️ Render", type="primary", key="showcase_lens_go"):
        try:
            with st.spinner("Tracing rays…"):
                st.session_state["_showcase_lens"] = sc.render_lens_image(n, float(fov), 1.0, float(dist))
        except sc.ShowcaseError as exc:
            st.session_state.pop("_showcase_lens", None)
            st.error(str(exc))
            return
    img = st.session_state.get("_showcase_lens")
    if img is None:
        st.info("Choose a view and press Render.")
        return
    st.plotly_chart(sp.lens_figure(img, True, edge), width="stretch")
    if compare:
        st.plotly_chart(sp.lens_figure(img, False), width="stretch")
    if img.shadow_px_measured is not None:
        err = img.shadow_error or 0.0
        (st.success if err < 0.02 else st.warning)(
            f"Check: the shadow's radius is {img.shadow_px_measured:.1f} px; the exact value, from the critical impact parameter "
            f"b_c = 3√3 M, is {img.shadow_px_expected:.1f} px ({err:.2%} apart).")
    for note in img.notes:
        st.warning(note)
    st.caption(f"Computed on: {img.backend}. {int((img.status == sc.WINDING).sum()):,} pixels are light still orbiting the photon sphere.")
    st.download_button("⬇️ Download the image (PNG)", data=_png(img.rgb), file_name="black_hole.png", mime="image/png", key="showcase_lens_dl")


def _orbit_tab() -> None:
    st.caption("A body orbiting the hole, drawn on Flamm's paraboloid: the surface whose geometry is the hole's spatial geometry in "
               "the equatorial plane (the orbit is lifted onto it; height is not time). The orbit is not a closed ellipse: its closest "
               "point advances each lap.")
    c1, c2, c3 = st.columns(3)
    p = c1.slider("Semi-latus rectum p (in M)", 7.0, 200.0, 40.0, 1.0, key="showcase_orbit_p",
                  help="Size of the orbit. Stable bound orbits need p > 6 + 2e.")
    e = c2.slider("Eccentricity e", 0.0, 0.9, 0.5, 0.05, key="showcase_orbit_e")
    laps = c3.slider("Orbits", 1, 8, 3, key="showcase_orbit_n")
    try:
        orb = sc.orbit(p, e, laps)
    except sc.ShowcaseError as exc:
        st.error(str(exc))
        return
    st.plotly_chart(sp.orbit_figure(orb), width="stretch")
    m1, m2, m3 = st.columns(3)
    second = orb.precession_weak_field * (1 + (18 + e * e) / (4 * p))
    m1.metric("Periapsis advance per orbit", f"{orb.precession:.4f} rad" if orb.precession is not None else "—",
              help="Measured from the integrated orbit.")
    m2.metric("Leading-order prediction 6π/p", f"{orb.precession_weak_field:.4f} rad")
    m3.metric("With the next-order term", f"{second:.4f} rad")
    (st.success if orb.drift < 1e-8 else st.warning)(
        f"Check: the conserved quantity of the orbit equation changes by {orb.drift:.1e} (relative) over the whole run.")


def _critical_tab() -> None:
    st.caption("The 2-D Ising model at 0.85, 1.0 and 1.25 times its exact critical temperature T_c = 2 / ln(1 + √2) ≈ 2.269, all started "
               "from the same random state. Below T_c one colour takes over; at T_c domains of every size appear and keep changing; "
               "above it the picture is noise. Large lattices run on the GPU when one is available.")
    c1, c2, c3 = st.columns(3)
    size = c1.slider("Lattice side L", 32, 128, 64, 16, key="showcase_crit_L")
    sweeps = c2.slider("Sweeps", 120, 600, 240, 60, key="showcase_crit_sweeps")
    seed = int(c3.number_input("Seed", 0, 10 ** 6, 7, key="showcase_crit_seed"))
    if st.button("🧲 Run", type="primary", key="showcase_crit_go"):
        try:
            with st.spinner("Running three lattices…"):
                st.session_state["_showcase_movie"] = sc.critical_movie(size, sweeps, max(1, sweeps // 60), seed)
        except sc.ShowcaseError as exc:
            st.session_state.pop("_showcase_movie", None)
            st.error(str(exc))
            return
    movie = st.session_state.get("_showcase_movie")
    if movie is None:
        st.info("Choose a size and press Run.")
        return
    st.plotly_chart(sp.critical_figure(movie), width="stretch")
    st.plotly_chart(sp.magnetisation_figure(movie), width="stretch")
    cold, hot = movie.runs[0].order[-30:].mean(), movie.runs[2].order[-30:].mean()
    st.caption(f"Check: T_c = {movie.tc:.4f} is the exact (Onsager) value. Mean |m| over the last 30 sweeps: {cold:.2f} below T_c, "
               f"{hot:.2f} above. (On a finite lattice |m| above T_c is a small positive number of order 1/L, not exactly zero.)")


def render_showcase_tab() -> None:
    st.subheader("🎬 Visual showcase")
    st.caption("Three things worth looking at, each shown with the number that says the picture is right. "
               f"Compute: {gpu.describe()}")
    tabs = st.tabs(["🕳️ Black hole shadow & lensing", "🪐 Orbit & spacetime", "🧲 Critical point"])
    with tabs[0]:
        _lens_tab()
    with tabs[1]:
        _orbit_tab()
    with tabs[2]:
        _critical_tab()
