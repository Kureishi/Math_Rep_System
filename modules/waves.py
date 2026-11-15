"""
Waves and optics: Fraunhofer diffraction by FFT, a ripple tank, and the Doppler effect -- Streamlit-free, no LLM.

Every result is checked against something the method did NOT use to produce it:
  * FRAUNHOFER DIFFRACTION is computed by an FFT of the sampled aperture (single slit, double slit, N-slit grating, and in 2-D a
    circular, rectangular or double-slit aperture). It is then compared with the closed forms: the fringe spacing lambda*L/d, the
    sinc^2 envelope (fitted from the measured fringe heights, so the slit width is RECOVERED from the pattern rather than assumed), the
    first single-slit zero lambda*L/a, and for a round aperture the Airy zero 1.2197 lambda*L/D and the 83.78 % encircled energy.
  * THE RIPPLE TANK integrates the 2-D wave equation with a leapfrog finite-difference scheme (a sponge layer absorbs the boundary
    reflections). Two coherent sources must produce interference maxima where the path difference r2 - r1 is a whole number of
    wavelengths (plus the phase offset); one source must decay as 1/sqrt(r) and have the wavelength it was given. The wavelength
    used in these comparisons is the NUMERICAL one, from the scheme's own dispersion relation, so grid dispersion is not mistaken
    for a physics error (and is reported).
  * THE DOPPLER EFFECT is simulated from first principles: the source emits a wavefront every 1/f0 seconds, and the time each one
    reaches the observer is found by solving |observer(T) - emission point| = c (T - t_emit) exactly. The received frequency measured
    from those arrival times is compared with the textbook formula, and for a supersonic source the Mach-cone angle measured from
    where successive wavefronts intersect is compared with arcsin(c/v).

PARAXIAL APPROXIMATION, STATED. The diffraction formulas use the screen coordinate x = lambda * L * f (small angles, x << L), which
is what the FFT gives directly. At wide angles the exact pattern differs; the lab says so rather than hiding it.

Units: SI in the engines (metres, seconds, hertz); the page converts nm / micrometres / mm for display. The ripple tank works in grid
cells and time steps (wave speed in cells per step), which is dimensionless by construction.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from modules import gpu
from modules.quantum_common import CheckList


class WavesError(ValueError):
    """Bad input, with a reason a student can act on."""


def _positive(name: str, value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise WavesError(f"{name} must be a positive number.")
    return float(value)


def _parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample position of the extremum near index i, from a parabola through three samples."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = a - 2 * b + c
    return float(i) if denom == 0 else float(i) + 0.5 * (a - c) / denom


# ------------------------------------------------------------------------------------------ 1-D Fraunhofer diffraction

KINDS_1D = ("single slit", "double slit", "grating")
SAMPLES_PER_SLIT = 64
MAX_SAMPLES = 2 ** 20


@dataclass
class DiffractionPattern:
    kind: str
    wavelength: float
    distance: float
    slit_width: float
    separation: float
    n_slits: int
    x: np.ndarray                  # screen positions (m), the displayed window
    intensity: np.ndarray          # from the FFT, central maximum = 1
    analytic: np.ndarray           # closed form, same normalisation
    aperture_x: np.ndarray         # a window of the aperture function, for display
    aperture: np.ndarray
    fringe_spacing_theory: float | None
    fringe_spacing_measured: float | None
    envelope_zero_theory: float
    envelope_zero_measured: float | None
    features_x: np.ndarray         # the fringe features that were measured (m)
    features_kind: str             # "dark fringes" (double slit) or "principal maxima" (grating)
    checks: CheckList = field(default_factory=CheckList)
    backend: str = "CPU (NumPy)"
    n_samples: int = 0
    notes: list[str] = field(default_factory=list)


def _slit_centres(kind: str, separation: float, n_slits: int) -> np.ndarray:
    n = {"single slit": 1, "double slit": 2}.get(kind, n_slits)
    return (np.arange(n) - (n - 1) / 2.0) * separation


def _coverage(grid_edges_lo: np.ndarray, dx: float, centre: float, width: float) -> np.ndarray:
    """Fraction of each sample cell [lo, lo + dx] covered by the slit [centre - width/2, centre + width/2]: the aperture is placed
    with sub-sample accuracy, so a slit width or separation that is not a multiple of dx is still exact."""
    lo = np.maximum(grid_edges_lo, centre - width / 2)
    hi = np.minimum(grid_edges_lo + dx, centre + width / 2)
    return np.clip(hi - lo, 0.0, dx) / dx


def interference_factor(x: np.ndarray, wavelength: float, distance: float, separation: float, n_slits: int) -> np.ndarray:
    """[sin(N pi d x / lambda L) / (N sin(pi d x / lambda L))]^2, with its limit N^2/N^2 = 1 at the principal maxima."""
    if n_slits == 1:
        return np.ones_like(x)
    u = np.pi * separation * x / (wavelength * distance)
    s = np.sin(u)
    near = np.abs(s) < 1e-12
    safe = np.where(near, 1.0, s)
    return np.where(near, 1.0, (np.sin(n_slits * u) / (n_slits * safe)) ** 2)


def analytic_intensity(x: np.ndarray, wavelength: float, distance: float, slit_width: float, separation: float, n_slits: int) -> np.ndarray:
    """The Fraunhofer intensity (paraxial), central maximum 1: sinc^2(a x / lambda L) times the N-slit interference factor."""
    return np.sinc(slit_width * x / (wavelength * distance)) ** 2 * interference_factor(x, wavelength, distance, separation, n_slits)


def _principal_peaks(x: np.ndarray, intensity: np.ndarray, envelope: np.ndarray, x_max: float) -> np.ndarray:
    """Positions (sub-sample) of the principal interference maxima inside |x| <= x_max. A peak counts as principal when it is at least
    half as tall as the single-slit envelope at its position -- that only SELECTS which peaks to measure (it rejects the weak secondary
    maxima of a grating); the positions themselves come from the FFT result alone."""
    window = np.abs(x) <= x_max
    idx_all, _ = find_peaks(intensity)
    keep = [i for i in idx_all if window[i] and intensity[i] >= 0.5 * envelope[i] and envelope[i] > 1e-4]
    if not keep:
        return np.array([])
    dx = float(x[1] - x[0])
    return np.array([x[0] + _parabolic_peak(intensity, i) * dx for i in keep])


def _dark_fringes(x: np.ndarray, intensity: np.ndarray, x_max: float) -> np.ndarray:
    """Positions (sub-sample) of the intensity minima inside |x| <= x_max. For two slits the minima are the zeros of the cos^2 factor,
    which the single-slit envelope does not move (it only scales them), so they give the fringe spacing without the bias that the
    envelope puts on the maxima."""
    idx, _ = find_peaks(-intensity, prominence=5e-4)
    dx = float(x[1] - x[0])
    keep = [i for i in idx if abs(x[i]) <= x_max]
    return np.array([x[0] + _parabolic_peak(intensity, i) * dx for i in keep])


def diffraction_1d(kind: str, wavelength: float, distance: float, slit_width: float, separation: float = 0.0, n_slits: int = 2,
                   lobes: float = 6.0, backend: gpu.Backend | None = None) -> DiffractionPattern:
    """Fraunhofer diffraction of a slit, a double slit or an N-slit grating, computed by an FFT of the aperture.

    The aperture is sampled at `slit_width / 64` and padded so the screen is resolved to a small fraction of one fringe; the window
    returned covers +-`lobes` diffraction lobes (+-lobes * lambda L / a)."""
    if kind not in KINDS_1D:
        raise WavesError(f"Choose one of: {', '.join(KINDS_1D)}.")
    lam, L, a = _positive("The wavelength", wavelength), _positive("The distance to the screen", distance), _positive("The slit width", slit_width)
    n = 1 if kind == "single slit" else 2 if kind == "double slit" else int(n_slits)
    if kind == "grating" and not 3 <= n <= 20:
        raise WavesError("A grating here has between 3 and 20 slits (use 'double slit' for two).")
    d = float(separation) if n > 1 else 0.0
    if n > 1 and d <= 1.2 * a:
        raise WavesError("The slit separation must be larger than 1.2 times the slit width (the slits would nearly touch).")
    backend = backend or gpu.CPU
    dx = a / SAMPLES_PER_SLIT
    extent = (n - 1) * d + a
    n_samples = int(2 ** math.ceil(math.log2(max(2 ** 14, SAMPLES_PER_SLIT * extent / dx))))
    if n_samples > MAX_SAMPLES:
        raise WavesError("The separation is too large compared with the slit width (the sampling would need more than a million points). "
                         "Use a separation under about 150 slit widths.")
    xs = (np.arange(n_samples) - n_samples // 2) * dx
    edges = xs - dx / 2
    aperture = np.zeros(n_samples)
    for c in _slit_centres(kind, d, n):
        aperture += _coverage(edges, dx, float(c), a)
    aperture = np.minimum(aperture, 1.0)

    def on_gpu():
        xp = backend.xp
        return gpu.to_host(xp.abs(xp.fft.fft(xp.asarray(aperture))) ** 2)

    power = gpu.attempt(backend, on_gpu, lambda: np.abs(np.fft.fft(aperture)) ** 2)
    used = gpu.last_used() or gpu.CPU.name
    freqs = np.fft.fftfreq(n_samples, d=dx)
    order = np.argsort(freqs)
    screen_x = lam * L * freqs[order]
    intensity = power[order] / power[0]                          # the zero-frequency term is the central maximum (sum of the aperture)^2
    half = lobes * lam * L / a
    window = np.abs(screen_x) <= half
    sx, si = screen_x[window], intensity[window]
    analytic = analytic_intensity(sx, lam, L, a, d, n)
    envelope = np.sinc(a * sx / (lam * L)) ** 2

    zero_theory = lam * L / a
    spacing_theory = lam * L / d if n > 1 else None
    if n == 2:
        peaks, features_kind = _dark_fringes(sx, si, 0.9 * zero_theory), "dark fringes"
    elif n > 2:
        peaks, features_kind = _principal_peaks(sx, si, envelope, 0.9 * zero_theory), "principal maxima"
    else:
        peaks, features_kind = np.array([]), ""
    spacing_measured = float(np.mean(np.diff(np.sort(peaks)))) if len(peaks) >= 2 else None

    # the first diffraction zero: for a single slit it is the first minimum; otherwise it is recovered by fitting the envelope to the heights
    # of the measured principal maxima (the slit width is then an OUTPUT of the pattern)
    zero_measured: float | None = None
    notes: list[str] = []
    if n == 1:
        pos = sx > 0
        i_min, _ = find_peaks(-si[pos])
        if len(i_min):
            step = float(sx[1] - sx[0])
            zero_measured = float(sx[pos][0] + _parabolic_peak(si[pos], int(i_min[0])) * step)
    else:
        if n == 2:
            # a bright fringe's centre is midway between two neighbouring dark fringes, where the interference factor is exactly 1, so the
            # intensity there IS the envelope; skip fringes that straddle one of the envelope's own zeros, where the dark fringes merge
            dark = np.sort(_dark_fringes(sx, si, 2.9 * zero_theory))
            mids = 0.5 * (dark[:-1] + dark[1:]) if len(dark) >= 2 else np.array([])
            gap = np.diff(dark) if len(dark) >= 2 else np.array([])
            ok_mid = np.abs(gap - lam * L / d) < 0.25 * lam * L / d if len(mids) else np.array([], dtype=bool)
            near_zero = np.abs(mids / zero_theory - np.round(mids / zero_theory)) < 0.08 if len(mids) else np.array([], dtype=bool)
            wide = mids[ok_mid & ~near_zero] if len(mids) else mids
        else:
            wide = _principal_peaks(sx, si, envelope, 0.97 * 3 * zero_theory)
        heights = np.array([float(np.interp(p, sx, si)) for p in wide])
        keep = heights > 1e-3
        if keep.sum() >= 3:
            px, ph = wide[keep], heights[keep]
            fit = least_squares(lambda p: np.sinc(p[0] * px / (lam * L)) ** 2 - ph, x0=[a], x_scale=[a], bounds=(0.2 * a, 5 * a))
            zero_measured = lam * L / float(fit.x[0])
        else:
            notes.append("The sinc^2 envelope was not fitted: fewer than three bright fringes fall inside the first lobes. Increase the separation "
                         "relative to the slit width to see it.")

    ck = CheckList()
    centre_lobes = np.abs(sx) <= min(half, 3 * zero_theory)
    err = float(np.max(np.abs(si[centre_lobes] - analytic[centre_lobes])))
    ck.add("The FFT agrees with the closed-form Fraunhofer pattern", err < 5e-3,
           f"largest difference in normalised intensity over the central lobes: {err:.1e} (tolerance 5e-3)")
    if n > 1:
        if spacing_measured is not None and spacing_theory is not None:
            rel = abs(spacing_measured - spacing_theory) / spacing_theory
            tol = 0.02
            ck.add("Fringe spacing = lambda L / d", rel < tol,
                   f"measured {spacing_measured * 1e3:.4g} mm from {len(peaks)} {features_kind}, formula {spacing_theory * 1e3:.4g} mm "
                   f"(difference {rel:.2%}, tolerance {tol:.0%})")
        else:
            ck.add("Fringe spacing = lambda L / d", False, "fewer than two fringes fall inside the central lobe, so the spacing cannot be measured: "
                   "increase the separation relative to the slit width")
        if zero_measured is not None:
            rel = abs(zero_measured - zero_theory) / zero_theory
            ck.add("sinc^2 envelope: slit width recovered from the fringe heights", rel < 0.03,
                   f"fitting sinc^2 to the measured fringe heights gives a first zero at {zero_measured * 1e3:.4g} mm; lambda L / a = "
                   f"{zero_theory * 1e3:.4g} mm (difference {rel:.2%})")
    else:
        if zero_measured is not None:
            rel = abs(zero_measured - zero_theory) / zero_theory
            ck.add("First minimum at lambda L / a", rel < 0.01, f"measured {zero_measured * 1e3:.4g} mm, formula {zero_theory * 1e3:.4g} mm (difference {rel:.2%})")
        else:
            ck.add("First minimum at lambda L / a", False, "no minimum found in the window")
    power_total = float(np.sum(aperture ** 2))
    parseval = float(np.sum(power)) / (n_samples * power_total)
    ck.add("Energy is conserved by the transform (Parseval)", abs(parseval - 1.0) < 1e-9, f"sum of the pattern / sum of the aperture = {parseval:.12f}")

    shown = np.abs(xs) <= max(extent, 2 * a) * 0.75
    return DiffractionPattern(kind, lam, L, a, d, n, sx, si, analytic, xs[shown], aperture[shown], spacing_theory, spacing_measured,
                              zero_theory, zero_measured, peaks, features_kind, ck, used, n_samples, notes)


# ------------------------------------------------------------------------------------------ 2-D apertures

KINDS_2D = ("circular", "rectangular", "double slit")
AIRY_FIRST_ZERO = 1.2196698912665045          # j_{1,1} / pi
AIRY_ENCIRCLED_FIRST_RING = 0.8377            # the fraction of the power inside the first dark ring (1 - J0(j11)^2 - J1(j11)^2)


@dataclass
class DiffractionPattern2D:
    kind: str
    wavelength: float
    distance: float
    size: float                      # D for a circle, the width for a rectangle, a for slits (m)
    size2: float                     # the height for a rectangle / slit length (m); equals size for a circle
    separation: float
    x: np.ndarray                    # screen axis (m), the displayed window
    intensity: np.ndarray            # (ny, nx), central maximum 1, the displayed window
    aperture: np.ndarray             # the aperture mask, displayed window
    aperture_extent: float           # half-width of the displayed aperture window (m)
    row: np.ndarray                  # the profile through the centre along x (same window)
    first_zero_theory: float
    first_zero_measured: float | None
    checks: CheckList = field(default_factory=CheckList)
    backend: str = "CPU (NumPy)"
    n_pixels: int = 0


def _disk_coverage(rho: np.ndarray, radius: float, dx: float) -> np.ndarray:
    """Anti-aliased disk: the pixel's covered area is approximated from its distance to the edge (exact for a straight edge)."""
    return np.clip(0.5 - (rho - radius) / dx, 0.0, 1.0)


def diffraction_2d(kind: str, wavelength: float, distance: float, size: float, size2: float | None = None, separation: float = 0.0,
                   n_pixels: int = 1024, view_lobes: float = 6.0, backend: gpu.Backend | None = None) -> DiffractionPattern2D:
    """Far-field pattern of a circular, rectangular or double-slit aperture by a 2-D FFT."""
    if kind not in KINDS_2D:
        raise WavesError(f"Choose one of: {', '.join(KINDS_2D)}.")
    lam, L, w = _positive("The wavelength", wavelength), _positive("The distance to the screen", distance), _positive("The aperture size", size)
    h = _positive("The aperture height", size2) if size2 is not None else (w if kind != "double slit" else 6 * w)
    d = float(separation)
    if kind == "double slit" and d <= 1.2 * w:
        raise WavesError("The slit separation must be larger than 1.2 times the slit width.")
    n_pixels = int(n_pixels)
    if not 256 <= n_pixels <= 8192:
        raise WavesError("Use between 256 and 8192 pixels per side.")
    backend = backend or gpu.CPU
    ref = {"circular": w, "rectangular": min(w, h), "double slit": w}[kind]       # the feature that must be resolved
    dx = ref / 48.0
    extent_needed = {"circular": w, "rectangular": max(w, h), "double slit": max(d + w, h)}[kind]
    if n_pixels * dx < 3 * extent_needed:
        raise WavesError("The aperture is too large for this many pixels: raise the pixel count or make the aperture smaller / closer together.")
    idx = (np.arange(n_pixels) - n_pixels // 2) * dx
    X, Y = np.meshgrid(idx, idx)
    if kind == "circular":
        mask = _disk_coverage(np.hypot(X, Y), w / 2, dx)
    elif kind == "rectangular":
        mask = np.clip(0.5 + (w / 2 - np.abs(X)) / dx, 0, 1) * np.clip(0.5 + (h / 2 - np.abs(Y)) / dx, 0, 1)
    else:
        mask = sum(np.clip(0.5 + (w / 2 - np.abs(X - s)) / dx, 0, 1) for s in (-d / 2, d / 2)) * np.clip(0.5 + (h / 2 - np.abs(Y)) / dx, 0, 1)
        mask = np.minimum(mask, 1.0)

    def on_gpu():
        xp = backend.xp
        return gpu.to_host(xp.abs(xp.fft.fft2(xp.asarray(mask))) ** 2)

    power = gpu.attempt(backend, on_gpu, lambda: np.abs(np.fft.fft2(mask)) ** 2)
    used = gpu.last_used() or gpu.CPU.name
    power = np.fft.fftshift(power)
    c = n_pixels // 2
    central = float(power[c, c])
    normalised = power / central
    f = np.fft.fftshift(np.fft.fftfreq(n_pixels, d=dx))
    screen = lam * L * f
    zero_theory = {"circular": AIRY_FIRST_ZERO * lam * L / w, "rectangular": lam * L / w, "double slit": lam * L / h}[kind]

    row = normalised[c, :]
    col = normalised[:, c]
    step = float(screen[1] - screen[0])

    def first_minimum(profile: np.ndarray) -> float | None:
        right = profile[c:]
        i_min, _ = find_peaks(-right[: max(5, int(4 * zero_theory / step))])
        return float(_parabolic_peak(right, int(i_min[0])) * step) if len(i_min) else None

    ck = CheckList()
    measured: float | None
    if kind == "circular":
        measured = first_minimum(row)
        if measured is not None:
            rel = abs(measured - zero_theory) / zero_theory
            ck.add("First dark ring at 1.2197 lambda L / D (the Airy zero)", rel < 0.02,
                   f"measured {measured * 1e3:.4g} mm, formula {zero_theory * 1e3:.4g} mm (difference {rel:.2%})")
        else:
            ck.add("First dark ring at 1.2197 lambda L / D (the Airy zero)", False, "no minimum found: raise the pixel count")
        rr = np.hypot(*np.meshgrid(np.arange(n_pixels) - c, np.arange(n_pixels) - c)) * step
        # the power that passes a binary aperture is its open area (a pixel's anti-aliased coverage counts once, not squared)
        inside = float(np.sum(power[rr <= zero_theory])) / (n_pixels ** 2 * float(np.sum(mask)))
        ck.add("83.8 % of the light falls inside the first dark ring", abs(inside - AIRY_ENCIRCLED_FIRST_RING) < 0.01,
               f"measured {inside:.2%} of the power through the aperture, theory {AIRY_ENCIRCLED_FIRST_RING:.2%}")
    elif kind == "rectangular":
        mx, my = first_minimum(row), first_minimum(col)
        measured = mx
        if mx is not None and my is not None:
            ex, ey = abs(mx - lam * L / w) / (lam * L / w), abs(my - lam * L / h) / (lam * L / h)
            ck.add("First zeros at lambda L / width and lambda L / height", max(ex, ey) < 0.02,
                   f"along x: measured {mx * 1e3:.4g} mm, formula {lam * L / w * 1e3:.4g} mm ({ex:.2%}); along y: measured {my * 1e3:.4g} mm, "
                   f"formula {lam * L / h * 1e3:.4g} mm ({ey:.2%})")
        else:
            ck.add("First zeros at lambda L / width and lambda L / height", False, "no minimum found: raise the pixel count")
    else:
        in_lobe = np.abs(screen) <= 0.9 * lam * L / w
        idx_p, _ = find_peaks(-row[in_lobe], prominence=5e-4)       # dark fringes: unbiased by the single-slit envelope
        px = screen[in_lobe]
        pos = np.array([float(px[0] + _parabolic_peak(row[in_lobe], int(i)) * step) for i in idx_p])
        theory = lam * L / d
        measured = float(np.mean(np.diff(np.sort(pos)))) if len(pos) >= 2 else None
        if measured is not None:
            rel = abs(measured - theory) / theory
            ck.add("Fringe spacing = lambda L / d", rel < 0.03, f"measured {measured * 1e3:.4g} mm from {len(pos)} dark fringes, formula {theory * 1e3:.4g} mm (difference {rel:.2%}, tolerance 3%)")
        else:
            ck.add("Fringe spacing = lambda L / d", False, "fewer than two fringes in the central lobe")
        zero_theory = lam * L / w
    parseval = float(np.sum(power)) / (n_pixels ** 2 * float(np.sum(mask ** 2)))
    ck.add("Energy is conserved by the transform (Parseval)", abs(parseval - 1.0) < 1e-9, f"sum of the pattern / sum of the aperture = {parseval:.12f}")

    half = view_lobes * (zero_theory if kind != "double slit" else lam * L / w)
    half = min(half, float(screen[-1]))
    sel = np.abs(screen) <= half
    win = np.ix_(sel, sel)
    ap_half = min(extent_needed * 0.8, float(idx[-1]))
    ap_sel = np.abs(idx) <= ap_half
    return DiffractionPattern2D(kind, lam, L, w, h, d, screen[sel], normalised[win], mask[np.ix_(ap_sel, ap_sel)], ap_half, row[sel], zero_theory, measured, ck, used, n_pixels)


# ------------------------------------------------------------------------------------------ ripple tank

@dataclass
class Source:
    x: float                 # grid cells from the left edge
    y: float                 # grid cells from the top edge
    phase: float = 0.0       # radians
    amplitude: float = 1.0


@dataclass
class RippleTank:
    n: int
    wavelength_cells: float
    courant: float
    steps: int
    sources: list[Source]
    frames: np.ndarray               # (n_frames, n, n) float32
    frame_steps: np.ndarray
    amplitude: np.ndarray            # rms amplitude over the last whole periods, (n, n)
    final_field: np.ndarray          # (n, n)
    numerical_wavelength: float
    period_steps: int
    probe_x: np.ndarray | None       # a ray from a single source: x (cells)
    probe_u: np.ndarray | None
    maxima: list[tuple[float, float]]    # (measured angle, predicted angle) in degrees, two-source runs
    checks: CheckList = field(default_factory=CheckList)
    backend: str = "CPU (NumPy)"
    seed_free: bool = True


def numerical_wavelength(wavelength_cells: float, courant: float) -> float:
    """The wavelength the leapfrog scheme actually propagates at the nominal one, averaged over directions. The scheme obeys
    sin(w dt / 2) = C sqrt(sin^2(kx/2) + sin^2(ky/2)) (grid units); solving for |k| at each direction and averaging gives the
    effective wavelength -- about half a percent SHORTER than nominal at 12 cells per wavelength (the grid makes the wave a little slower, so more of it fits in a given frequency)."""
    k_nominal = 2 * math.pi / wavelength_cells
    omega_dt = k_nominal * courant                       # w dt for the nominal wave (c = 1 cell per unit time, dt = C)
    s = math.sin(omega_dt / 2)

    def k_at(psi: float) -> float:
        lo, hi = 1e-6, math.pi
        f = lambda k: courant * math.sqrt(math.sin(k * math.cos(psi) / 2) ** 2 + math.sin(k * math.sin(psi) / 2) ** 2) - s
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return 0.5 * (lo + hi)

    ks = [k_at(psi) for psi in np.linspace(0.0, math.pi / 4, 9)]
    return 2 * math.pi / float(np.mean(ks))


def _sponge(n: int, width: int, strength: float = 0.06) -> np.ndarray:
    """Damping factor per cell: 1 in the interior, falling quadratically to 1 - strength at the edge."""
    ramp = np.zeros(n)
    edge = np.arange(width)
    ramp[:width] = ((width - edge) / width) ** 2
    ramp[n - width:] = ramp[:width][::-1]
    return 1.0 - strength * np.maximum.outer(ramp, ramp)


def ripple_tank(n: int = 281, wavelength_cells: float = 12.0, sources: list[Source] | None = None, courant: float = 0.5,
                n_frames: int = 36, backend: gpu.Backend | None = None) -> RippleTank:
    """Integrate u_tt = c^2 (u_xx + u_yy) from rest with soft sinusoidal sources, then measure the steady interference pattern.

    Leapfrog scheme, Courant number `courant` (the 2-D stability limit is 1/sqrt 2); a sponge layer 30 cells wide absorbs the outgoing
    waves. The run lasts long enough for the wave to reach the edge of the measuring region and then 6 further periods are averaged."""
    n = int(n)
    if not 121 <= n <= 1025 or n % 2 == 0:
        raise WavesError("Use an odd grid size between 121 and 1025.")
    wl = float(wavelength_cells)
    if not 8 <= wl <= 40:
        raise WavesError("Use between 8 and 40 grid cells per wavelength (fewer is inaccurate, more is slow).")
    if not 0.05 <= courant <= 0.7:
        raise WavesError("The Courant number must be between 0.05 and 0.7 (the 2-D stability limit is 0.707).")
    sources = [Source(n // 2, n // 2)] if sources is None else sources
    if not 1 <= len(sources) <= 4:
        raise WavesError("Use between 1 and 4 sources.")
    for s in sources:
        if not (35 <= s.x <= n - 36 and 35 <= s.y <= n - 36):
            raise WavesError("Keep each source at least 35 cells from the edge (the outer 30 cells absorb waves).")
    backend = backend or gpu.CPU
    period = int(round(wl / courant))
    omega_dt = 2 * math.pi / period
    c2 = courant ** 2
    reach = 0.5 * n - 30
    settle = int(reach / courant) + 3 * period
    measure = 6 * period
    steps = settle + measure
    frame_every = max(1, steps // n_frames)
    damp = _sponge(n, 30)

    def run(xp, to_host):
        u = xp.zeros((n, n))
        u_prev = xp.zeros((n, n))
        damp_d = xp.asarray(damp)
        acc = xp.zeros((n, n))
        frames, frame_steps = [], []
        probe_row = None
        sx = [int(round(s.x)) for s in sources]
        sy = [int(round(s.y)) for s in sources]
        for t in range(steps):
            lap = xp.zeros_like(u)
            lap[1:-1, 1:-1] = u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
            u_next = (2 * u - u_prev + c2 * lap) * damp_d
            ramp = min(1.0, t / (3.0 * period))
            for s, ix, iy in zip(sources, sx, sy):
                u_next[iy, ix] += s.amplitude * ramp * math.sin(omega_dt * t - s.phase)
            u_prev, u = u, u_next
            if t >= settle:
                acc += u * u
            if t % frame_every == 0:
                frames.append(to_host(u).astype(np.float32))
                frame_steps.append(t)
        return to_host(u), to_host(acc), frames, frame_steps

    def on_gpu():
        return run(backend.xp, gpu.to_host)

    final, acc, frames, frame_steps = gpu.attempt(backend, on_gpu, lambda: run(np, lambda a: a))
    used = gpu.last_used() or gpu.CPU.name
    amp = np.sqrt(acc / measure)
    lam_num = numerical_wavelength(wl, courant)

    ck = CheckList()
    maxima: list[tuple[float, float]] = []
    probe_x = probe_u = None
    if len(sources) == 1:
        s = sources[0]
        iy, ix0 = int(round(s.y)), int(round(s.x))
        radii = np.arange(ix0 + 6, n - 36) - ix0
        a_r = amp[iy, ix0 + radii]
        sel = (radii >= 2.0 * wl) & (radii <= reach - 8)
        slope = float(np.polyfit(np.log(radii[sel]), np.log(a_r[sel]), 1)[0]) if sel.sum() > 5 else float("nan")
        ck.add("A circular wave decays as 1 / sqrt(r)", abs(slope + 0.5) < 0.06, f"log-log slope of the amplitude against distance: {slope:.3f} (theory -0.5, tolerance 0.06)")
        probe_x = np.arange(ix0, n - 30)
        probe_u = final[iy, ix0:n - 30]
        sign_changes = np.nonzero(np.diff(np.sign(probe_u[int(2 * wl):int(reach)])) != 0)[0]
        if len(sign_changes) >= 6:
            lam_meas = 2.0 * float(np.mean(np.diff(sign_changes)))
            err_num = abs(lam_meas - lam_num) / lam_num
            ck.add("The wavelength is the one the scheme propagates", err_num < 0.03,
                   f"measured {lam_meas:.2f} cells between zero crossings; nominal {wl:g}, the scheme's own dispersion relation gives {lam_num:.2f} ({err_num:.1%} from it)")
        else:
            ck.add("The wavelength is the one the scheme propagates", False, "too few zero crossings were found along the ray")
    else:
        p = sources[:2]
        d_cells = math.hypot(p[0].x - p[1].x, p[0].y - p[1].y)
        mid = ((p[0].x + p[1].x) / 2, (p[0].y + p[1].y) / 2)
        axis = math.atan2(p[1].y - p[0].y, p[1].x - p[0].x)               # the line through the sources
        dphi = p[1].phase - p[0].phase
        radius = min(0.5 * n - 45, 95)
        angles = np.linspace(-math.radians(80), math.radians(80), 1601)
        # angle measured from the perpendicular bisector (the broadside direction)
        broad = axis + math.pi / 2
        px = mid[0] + radius * np.cos(broad + angles)
        py = mid[1] + radius * np.sin(broad + angles)
        from scipy.ndimage import map_coordinates
        around = map_coordinates(amp, [py, px], order=3)
        pk, _ = find_peaks(around, prominence=0.15 * float(np.ptp(around)))
        measured_deg = np.degrees(angles[pk] + 0 * pk)
        r1 = np.hypot(px - p[0].x, py - p[0].y)
        r2 = np.hypot(px - p[1].x, py - p[1].y)
        path = (r2 - r1) / lam_num - dphi / (2 * math.pi)
        # a maximum is predicted wherever `path` crosses an integer; locate those crossings
        cross = []
        for m in range(int(math.floor(path.min())) - 1, int(math.ceil(path.max())) + 2):
            sign = np.sign(path - m)
            j = np.nonzero(np.diff(sign) != 0)[0]
            for jj in j:
                frac = (m - path[jj]) / (path[jj + 1] - path[jj])
                cross.append(math.degrees(angles[jj] + frac * (angles[jj + 1] - angles[jj])))
        cross = sorted(cross)
        used_pairs = []
        for mdeg in measured_deg:
            if not cross:
                break
            nearest = min(cross, key=lambda c: abs(c - mdeg))
            used_pairs.append((float(mdeg), float(nearest)))
        maxima = used_pairs
        if len(cross) >= 1 and used_pairs:
            spacing = float(np.median(np.diff(sorted(cross)))) if len(cross) > 1 else 20.0
            worst = max(abs(a - b) for a, b in used_pairs)
            tol = max(1.0, 0.12 * spacing)
            missing = sum(1 for c in cross if abs(c) < 70 and all(abs(c - m) > tol for m, _ in used_pairs))
            ck.add("Interference maxima sit where the path difference is a whole number of wavelengths", worst < tol and missing == 0,
                   f"{len(used_pairs)} maxima measured on an arc of radius {radius:.0f} cells; the largest offset from r2 - r1 = (m + phase/2pi) lambda is "
                   f"{worst:.2f} degrees (tolerance {tol:.1f}); {missing} predicted maxima within 70 degrees not found")
        else:
            ck.add("Interference maxima sit where the path difference is a whole number of wavelengths", False,
                   "no interference maxima were found: separate the sources by at least two wavelengths")
        ck.add("The sources are far enough apart to interfere", d_cells >= 1.5 * lam_num, f"separation {d_cells:.1f} cells = {d_cells / lam_num:.2f} wavelengths (at least 1.5 needed for two or more maxima)")
    ck.add("The scheme is stable and finite", bool(np.isfinite(final).all()) and float(np.max(np.abs(final))) < 50.0,
           f"largest |u| at the end: {float(np.max(np.abs(final))):.3g}; Courant number {courant:g} (limit 0.707)")
    return RippleTank(n, wl, courant, steps, sources, np.array(frames), np.array(frame_steps), amp, final, lam_num, period, probe_x, probe_u, maxima, ck, used)


# ------------------------------------------------------------------------------------------ Doppler effect

@dataclass
class DopplerRun:
    c: float
    f0: float
    source_velocity: np.ndarray        # (2,)
    source_start: np.ndarray
    observer_start: np.ndarray
    observer_velocity: np.ndarray
    emit_times: np.ndarray
    arrival_times: np.ndarray
    emit_positions: np.ndarray         # (K, 2)
    inst_frequency: np.ndarray         # 1 / (T_{k+1} - T_k), at the midpoints
    inst_theory: np.ndarray
    inst_times: np.ndarray             # midpoint arrival times
    mach_angle_theory: float | None    # degrees (half-angle of the cone)
    mach_angle_measured: float | None
    cone_points: np.ndarray            # intersections of successive wavefronts at the snapshot time
    snapshot_time: float
    checks: CheckList = field(default_factory=CheckList)
    notes: list[str] = field(default_factory=list)


def doppler_collinear(f0: float, c: float, v_source_toward: float, v_observer_toward: float) -> float:
    """The classical Doppler formula f' = f0 (c + v_o) / (c - v_s): v_s is the source speed TOWARD the observer, v_o the observer speed
    TOWARD the source, both measured in the frame of the medium."""
    if v_source_toward >= c:
        raise WavesError("A source at or above the wave speed has no single Doppler frequency: it outruns its own waves (see the Mach cone).")
    return f0 * (c + v_observer_toward) / (c - v_source_toward)


def _arrival_times(t_emit: np.ndarray, src: np.ndarray, obs0: np.ndarray, vobs: np.ndarray, c: float) -> np.ndarray:
    """Solve |obs0 + vobs T - src_k| = c (T - t_k), T >= t_k, for each emission (a quadratic in T). NaN where the wave never arrives."""
    r0 = obs0[None, :] - src
    a = float(vobs @ vobs) - c * c
    b = 2 * (r0 @ vobs) + 2 * c * c * t_emit
    cc = np.sum(r0 * r0, axis=1) - c * c * t_emit ** 2
    out = np.full(len(t_emit), np.nan)
    if abs(a) < 1e-12:
        ok = np.abs(b) > 1e-12
        t = -cc[ok] / b[ok]
        out[ok] = t
    else:
        disc = b * b - 4 * a * cc
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t1 = (-b - sq) / (2 * a)
        t2 = (-b + sq) / (2 * a)
        cand = np.where(t1 >= t_emit - 1e-12, t1, np.where(t2 >= t_emit - 1e-12, t2, np.nan))
        both = np.where((t1 >= t_emit - 1e-12) & (t2 >= t_emit - 1e-12), np.minimum(t1, t2), cand)
        out = np.where(ok, both, np.nan)
    # a root that is the observer catching up a wave that has not been emitted yet is not an arrival
    out[out < t_emit - 1e-9] = np.nan
    return out


def doppler(c: float, f0: float, source_speed: float, source_angle_deg: float = 0.0, source_start: tuple[float, float] = (-200.0, 0.0),
            observer_start: tuple[float, float] = (0.0, 30.0), observer_velocity: tuple[float, float] = (0.0, 0.0), n_waves: int = 400,
            snapshot_fraction: float = 0.5) -> DopplerRun:
    """A source moving at constant velocity past (or toward) an observer: emission times, exact arrival times, the measured frequency
    against time, and for v >= c the Mach cone."""
    c, f0 = _positive("The wave speed", c), _positive("The source frequency", f0)
    if source_speed < 0:
        raise WavesError("The source speed cannot be negative (use the direction angle to turn it around).")
    if not 20 <= n_waves <= 20000:
        raise WavesError("Use between 20 and 20000 emitted waves.")
    ang = math.radians(source_angle_deg)
    vs = source_speed * np.array([math.cos(ang), math.sin(ang)])
    s0, o0, vo = np.array(source_start, float), np.array(observer_start, float), np.array(observer_velocity, float)
    if float(vo @ vo) >= c * c:
        raise WavesError("The observer must move slower than the wave.")
    k = np.arange(n_waves)
    t_emit = k / f0
    emit_pos = s0[None, :] + np.outer(t_emit, vs)
    arrive = _arrival_times(t_emit, emit_pos, o0, vo, c)
    ok = np.isfinite(arrive)
    ck = CheckList()
    theory_mid = np.full(max(n_waves - 1, 0), np.nan)
    inst = np.full(max(n_waves - 1, 0), np.nan)
    mid_t = np.full(max(n_waves - 1, 0), np.nan)
    both = ok[:-1] & ok[1:]
    inst[both] = 1.0 / np.diff(arrive)[both]
    mid_t[both] = 0.5 * (arrive[:-1] + arrive[1:])[both]
    if both.any():
        # exact instantaneous frequency for any constant velocities: the rate at which arrival time advances, at the midpoint emission
        t_mid_emit = 0.5 * (t_emit[:-1] + t_emit[1:])
        # dT/dt_e by a symmetric difference of the exact solution around the midpoint emission
        eps = 0.25 / f0
        t_hi, t_lo = _arrival_times(t_mid_emit + eps, s0[None, :] + np.outer(t_mid_emit + eps, vs), o0, vo, c), _arrival_times(
            t_mid_emit - eps, s0[None, :] + np.outer(t_mid_emit - eps, vs), o0, vo, c)
        with np.errstate(invalid="ignore", divide="ignore"):
            theory_mid = f0 / ((t_hi - t_lo) / (2 * eps))
    collinear = abs((o0 - s0)[0] * vs[1] - (o0 - s0)[1] * vs[0]) < 1e-9 * max(1.0, float(np.linalg.norm(o0 - s0)) * source_speed) \
        and abs((o0 - s0)[0] * vo[1] - (o0 - s0)[1] * vo[0]) < 1e-9 * max(1.0, float(np.linalg.norm(o0 - s0)) * float(np.linalg.norm(vo)))
    mach_theory = math.degrees(math.asin(c / source_speed)) if source_speed > c else None
    if collinear and source_speed < c and both.any():
        line = (o0 - s0) / float(np.linalg.norm(o0 - s0))
        v_s_toward, v_o_toward = float(vs @ line), float(-(vo @ line))
        formula = doppler_collinear(f0, c, v_s_toward, v_o_toward)
        measured = (ok.sum() - 1) / (arrive[ok][-1] - arrive[ok][0])
        rel = abs(measured - formula) / formula
        ck.add("Received frequency = f0 (c + v_o) / (c - v_s)", rel < 1e-6,
               f"measured {measured:.6g} Hz from {int(ok.sum())} arrival times, formula {formula:.6g} Hz (difference {rel:.1e})")
    elif both.any() and source_speed < c:
        good = np.isfinite(theory_mid) & np.isfinite(inst)
        if good.sum() >= 5:
            rel = float(np.max(np.abs(inst[good] - theory_mid[good]) / theory_mid[good]))
            tol = 0.02
            ck.add("Frequency heard at every moment = f0 / (dT/dt_emit)", rel < tol,
                   f"largest difference between 1/(arrival gap) and the exact Doppler rate over {int(good.sum())} waves: {rel:.2%} (tolerance {tol:.0%}; "
                   "the gap averages the rate over one period)")
        if float(np.linalg.norm(vo)) == 0.0 and source_speed < c and good.sum() >= 5:
            # for a stationary observer the closed form f0 / (1 - (v_s/c) cos psi) at the retarded source position is exact
            t_mid_emit = 0.5 * (t_emit[:-1] + t_emit[1:])
            src_mid = s0[None, :] + np.outer(t_mid_emit, vs)
            to_obs = o0[None, :] - src_mid
            cospsi = (to_obs @ vs) / (np.linalg.norm(to_obs, axis=1) * source_speed) if source_speed > 0 else np.zeros(len(src_mid))
            closed = f0 / (1.0 - source_speed * cospsi / c)
            rel2 = float(np.max(np.abs(inst[good] - closed[good]) / closed[good]))
            ck.add("Passing source: f = f0 / (1 - (v/c) cos(angle to the observer))", rel2 < tol,
                   f"largest difference over the pass: {rel2:.2%} (tolerance {tol:.0%})")
    snap_t = float(snapshot_fraction * t_emit[-1])
    cone_pts = np.zeros((0, 2))
    mach_meas: float | None = None
    if mach_theory is not None:
        # intersections of the circles of successive emissions at the snapshot time lie on the Mach cone
        live = np.nonzero(t_emit < snap_t)[0]
        pts = []
        for i in live[:-1]:
            c1, c2_ = emit_pos[i], emit_pos[i + 1]
            r1, r2 = c * (snap_t - t_emit[i]), c * (snap_t - t_emit[i + 1])
            dd = float(np.linalg.norm(c2_ - c1))
            if dd == 0 or dd > r1 + r2 or dd < abs(r1 - r2):
                continue
            aa = (r1 * r1 - r2 * r2 + dd * dd) / (2 * dd)
            hh = math.sqrt(max(r1 * r1 - aa * aa, 0.0))
            base = c1 + aa * (c2_ - c1) / dd
            perp = np.array([-(c2_ - c1)[1], (c2_ - c1)[0]]) / dd
            pts += [base + hh * perp, base - hh * perp]
        cone_pts = np.array(pts) if pts else np.zeros((0, 2))
        if len(cone_pts) >= 8:
            axis = vs / source_speed
            side = np.array([-axis[1], axis[0]])
            rel_pts = cone_pts - emit_pos[live[-1]][None, :]
            u, v = rel_pts @ axis, rel_pts @ side
            upper = v > 0
            if upper.sum() >= 4:
                slope = float(np.polyfit(-u[upper], v[upper], 1)[0])
                mach_meas = math.degrees(math.atan(abs(slope)))
                rel = abs(mach_meas - mach_theory) / mach_theory
                ck.add("Mach cone half-angle = arcsin(c / v)", rel < 0.01, f"measured {mach_meas:.3f} degrees from {int(upper.sum())} wavefront intersections, formula {mach_theory:.3f} degrees (difference {rel:.2%})")
    notes: list[str] = []
    if 0.98 * c <= source_speed <= c:
        notes.append("At v = c the Doppler formula diverges and the Mach angle is 90 degrees: the waves pile up in front of the source into a shock. "
                     "Pick a speed clearly below or above c to see a measurable effect.")
    return DopplerRun(c, f0, vs, s0, o0, vo, t_emit, arrive, emit_pos, inst, theory_mid, mid_t, mach_theory, mach_meas, cone_pts, snap_t, ck, notes)
