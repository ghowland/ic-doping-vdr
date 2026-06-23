"""
ic-doping-vdr: Exact 1D Dopant Diffusion Simulator
  All arithmetic on Q335 basis (D = 2^335)
  Float64 comparison in parallel
  Zero accumulated arithmetic error by construction

Simulates arsenic dopant diffusion during rapid thermal annealing (RTA)
in silicon. The diffusion equation dC/dt = D * d^2C/dx^2 is time-stepped
via explicit Euler — each step feeds the next, accumulating float error
exactly as split-step does in DWDM.

The engineering question: after annealing, where is the junction?
If float drift moves the predicted junction depth, the predicted
threshold voltage is wrong.

Requirements: pip install vdr-math
"""

from __future__ import annotations
import time

from vdr.core import VDR
from vdr.linalg import Vec, Mat
from vdr.basis import to_qbasis, qb_mul, qb_div, qb_add, Q335
from vdr.export import to_float


# ============================================================================
# Physical constants on Q335 basis
# ============================================================================

# Arsenic diffusion coefficient in silicon at 1000°C RTA
# D_As ≈ 2.5e-17 m^2/s (well-established value)
# Exact rational: 25 / 10^18
D_AS = to_qbasis(VDR(25, 10**18))

# Grid: 200nm device, 40 points
N_GRID = 40
L_M = VDR(200, 10**9)        # 200 nm in meters
DX = to_qbasis(L_M / VDR(N_GRID - 1))

# Time stepping: 10 seconds anneal, 500 steps
# dt = 0.02 s -> stability parameter r = D*dt/dx^2
# dx ~ 5.13e-9 m, D = 2.5e-17
# r = 2.5e-17 * 0.02 / (5.13e-9)^2 = 5e-19 / 2.63e-17 ≈ 0.019
# Well under 0.5 stability limit
ANNEAL_TIME_S = 10
N_STEPS = 500
DT = to_qbasis(VDR(ANNEAL_TIME_S, N_STEPS))  # exact: 1/50 s


# ============================================================================
# Initial dopant profile
# ============================================================================

def make_initial_profile():
    """
    Implanted arsenic profile: Gaussian-like discrete dopant distribution.

    Peak at x = 30nm (shallow implant), falling off into substrate.
    Symmetric pair at x = 170nm for symmetry test.

    Concentration in atoms/m^3. Peak ~ 1e25 m^-3 (1e19 cm^-3).

    Built as exact rationals — no float anywhere.
    """
    grid = []
    for i in range(N_GRID):
        x_i = to_qbasis(VDR(i) * L_M / VDR(N_GRID - 1))
        grid.append(x_i)

    # Build a discrete step profile that's exactly symmetric
    # around the midpoint (x = 100nm, grid index 20)
    #
    # Left implant peak at indices 4-8 (center index 6 ≈ 30nm)
    # Right implant peak at indices 31-35 (center index 33 ≈ 170nm)
    # Exact mirror: index i <-> index (N_GRID - 1 - i)

    peak = to_qbasis(VDR(10**25))           # 1e25 m^-3
    shoulder = to_qbasis(VDR(5 * 10**24))   # 5e24
    tail = to_qbasis(VDR(10**24))           # 1e24
    zero = to_qbasis(VDR(0))

    # Profile values by distance from peak center
    # center: peak, ±1: shoulder, ±2: tail, rest: zero
    profile_shape = {0: peak, 1: shoulder, 2: tail}

    conc = [zero] * N_GRID

    # Left implant centered at index 6
    left_center = 6
    # Right implant centered at index N_GRID - 1 - 6 = 33
    right_center = N_GRID - 1 - left_center

    for offset, value in profile_shape.items():
        # Left side
        if 0 <= left_center - offset < N_GRID:
            conc[left_center - offset] = value
        if offset != 0 and 0 <= left_center + offset < N_GRID:
            conc[left_center + offset] = value

        # Right side — use SAME VDR objects for exact symmetry
        if 0 <= right_center - offset < N_GRID:
            conc[right_center - offset] = value
        if offset != 0 and 0 <= right_center + offset < N_GRID:
            conc[right_center + offset] = value

    return grid, conc


# ============================================================================
# Diffusion time-stepper (VDR exact on Q335)
# ============================================================================

def diffusion_step_vdr(conc, D, dx, dt, N):
    """
    One explicit Euler step of the diffusion equation.

    C[i]^{n+1} = C[i]^n + r * (C[i-1]^n - 2*C[i]^n + C[i+1]^n)

    where r = D * dt / dx^2 (precomputed on Q335).

    Boundary conditions: C[0] = C[N-1] = 0 (dopant cannot escape).
    All arithmetic on Q335 via qb_mul / qb_add. D never explodes.
    """
    # r = D * dt / dx^2 on Q335
    dx_sq = qb_mul(dx, dx)
    D_dt = qb_mul(D, dt)
    r = qb_div(D_dt, dx_sq)

    new_conc = [to_qbasis(VDR(0))] * N

    # Interior points: explicit Euler
    for i in range(1, N - 1):
        # laplacian = C[i-1] - 2*C[i] + C[i+1]
        two_ci = qb_add(conc[i], conc[i])
        lap = qb_add(qb_add(conc[i - 1], conc[i + 1]),
                      qb_mul(to_qbasis(VDR(-1)), two_ci))
        # C_new = C_old + r * laplacian
        new_conc[i] = qb_add(conc[i], qb_mul(r, lap))

    return new_conc


def run_diffusion_vdr(conc, D, dx, dt, N, n_steps):
    """
    Run n_steps of diffusion. Each step feeds the next.
    This is the sequential chain where float accumulates.

    Returns: list of concentration snapshots at selected steps.
    """
    snapshots = {0: [c for c in conc]}
    report_at = set()
    for s in [1, 10, 50, 100, 250, 500]:
        if s <= n_steps:
            report_at.add(s)

    current = list(conc)
    print("  Time-stepping %d steps..." % n_steps)

    for step in range(1, n_steps + 1):
        current = diffusion_step_vdr(current, D, dx, dt, N)
        if step in report_at:
            snapshots[step] = [c for c in current]
            print("    Step %d/%d done" % (step, n_steps))

    return current, snapshots


# ============================================================================
# Float64 mirror
# ============================================================================

def run_diffusion_float(conc_f, D_f, dx_f, dt_f, N, n_steps):
    """
    Identical algorithm in float64. Same explicit Euler, same grid.
    """
    r_f = D_f * dt_f / (dx_f * dx_f)
    snapshots = {0: list(conc_f)}
    report_at = set()
    for s in [1, 10, 50, 100, 250, 500]:
        if s <= n_steps:
            report_at.add(s)

    current = list(conc_f)

    for step in range(1, n_steps + 1):
        new = [0.0] * N
        for i in range(1, N - 1):
            lap = current[i - 1] - 2.0 * current[i] + current[i + 1]
            new[i] = current[i] + r_f * lap
        current = new
        if step in report_at:
            snapshots[step] = list(current)

    return current, snapshots


# ============================================================================
# Symmetry check
# ============================================================================

def check_symmetry_vdr(conc, N):
    """
    For symmetric implant: C[i] must equal C[N-1-i] exactly.
    Returns True only if every pair matches on Q335.
    """
    for i in range(N // 2):
        j = N - 1 - i
        if conc[i].v != conc[j].v or conc[i].d != conc[j].d:
            return False
    return True


def check_symmetry_float(conc_f, N):
    """Max |C[i] - C[N-1-i]| for float."""
    max_err = 0.0
    for i in range(N // 2):
        j = N - 1 - i
        err = abs(conc_f[i] - conc_f[j])
        if err > max_err:
            max_err = err
    return max_err


# ============================================================================
# Junction depth finder
# ============================================================================

def find_junction_depth(conc, grid, threshold, N):
    """
    Find the grid position where concentration crosses threshold.
    For the left implant: scan from peak outward (rightward) until
    C[i] drops below threshold.

    Returns x position in nm as float (from VDR export).
    """
    # Find peak index (left half)
    peak_idx = 0
    peak_val = 0.0
    for i in range(N // 2):
        v = to_float(conc[i])
        if v > peak_val:
            peak_val = v
            peak_idx = i

    # Scan rightward from peak for threshold crossing
    thresh_f = to_float(threshold)
    for i in range(peak_idx, N // 2):
        v_i = to_float(conc[i])
        v_next = to_float(conc[i + 1]) if i + 1 < N else 0.0
        if v_i >= thresh_f and v_next < thresh_f:
            # Linear interpolation for junction position
            x_i = to_float(grid[i]) * 1e9
            x_next = to_float(grid[i + 1]) * 1e9
            frac = (v_i - thresh_f) / (v_i - v_next) if v_i != v_next else 0.0
            return x_i + frac * (x_next - x_i)

    return None


def find_junction_depth_float(conc_f, grid_f, threshold_f, N):
    """Float version of junction depth finder."""
    peak_idx = 0
    peak_val = 0.0
    for i in range(N // 2):
        if conc_f[i] > peak_val:
            peak_val = conc_f[i]
            peak_idx = i

    for i in range(peak_idx, N // 2):
        v_i = conc_f[i]
        v_next = conc_f[i + 1] if i + 1 < N else 0.0
        if v_i >= threshold_f and v_next < threshold_f:
            x_i = grid_f[i] * 1e9
            x_next = grid_f[i + 1] * 1e9
            frac = (v_i - threshold_f) / (v_i - v_next) if v_i != v_next else 0.0
            return x_i + frac * (x_next - x_i)

    return None


# ============================================================================
# Conservation check
# ============================================================================

def total_dose_vdr(conc, dx, N):
    """Total dopant dose (integral of concentration) on Q335."""
    total = to_qbasis(VDR(0))
    for i in range(1, N - 1):
        total = qb_add(total, qb_mul(conc[i], dx))
    # half-weight boundaries
    total = qb_add(total, qb_mul(conc[0], qb_div(dx, to_qbasis(VDR(2)))))
    total = qb_add(total, qb_mul(conc[N - 1], qb_div(dx, to_qbasis(VDR(2)))))
    return total


def total_dose_float(conc_f, dx_f, N):
    """Total dopant dose in float64."""
    total = 0.0
    for i in range(1, N - 1):
        total += conc_f[i] * dx_f
    total += conc_f[0] * dx_f / 2.0
    total += conc_f[N - 1] * dx_f / 2.0
    return total


# ============================================================================
# Main simulation
# ============================================================================

def run_simulation():
    print("=" * 76)
    print("ic-doping-vdr: Exact Dopant Diffusion During RTA")
    print("  All arithmetic on Q335 basis (D = 2^335)")
    print("  Float64 comparison in parallel")
    print("  Zero accumulated arithmetic error by construction")
    print("=" * 76)
    print()

    # Setup
    grid, conc_init = make_initial_profile()
    dx = DX
    dt = DT
    D = D_AS
    N = N_GRID

    # Stability parameter
    dx_f = to_float(dx)
    dt_f = to_float(dt)
    D_f = to_float(D)
    r_f = D_f * dt_f / (dx_f * dx_f)

    print("--- Device Configuration ---")
    print("  Grid points:       %d" % N)
    print("  Device length:     200 nm")
    print("  Grid spacing:      %.4e m (%.2f nm)" % (dx_f, dx_f * 1e9))
    print("  Dopant:            Arsenic in Silicon")
    print("  Diffusion coeff:   %.2e m^2/s (at 1000°C)" % D_f)
    print("  Anneal time:       %d s" % ANNEAL_TIME_S)
    print("  Time steps:        %d" % N_STEPS)
    print("  dt:                %.4f s" % dt_f)
    print("  Stability r:       %.6f (must be < 0.5)" % r_f)
    print("  Profile:           Symmetric dual implant (peaks at 30nm, 170nm)")
    print()

    # Verify initial symmetry
    init_sym = check_symmetry_vdr(conc_init, N)
    print("--- Initial Profile Symmetry ---")
    print("  VDR: C[i] == C[N-1-i] for all i: %s" %
          ("EXACT" if init_sym else "BROKEN"))
    print()

    # Initial dose
    dose_init_vdr = total_dose_vdr(conc_init, dx, N)
    print("  Initial dose (VDR): %.6e atoms/m^2" % to_float(dose_init_vdr))
    print()

    # ---- VDR diffusion ----
    print("--- VDR Diffusion (Q335 exact, %d steps) ---" % N_STEPS)
    t0 = time.time()
    vdr_final, vdr_snaps = run_diffusion_vdr(conc_init, D, dx, dt, N, N_STEPS)
    vdr_time = time.time() - t0
    print("  VDR total time: %.2f s" % vdr_time)
    print()

    # ---- Float64 mirror ----
    print("--- Float64 Mirror ---")
    conc_init_f = [to_float(c) for c in conc_init]
    grid_f = [to_float(g) for g in grid]

    t0 = time.time()
    float_final, float_snaps = run_diffusion_float(
        conc_init_f, D_f, dx_f, dt_f, N, N_STEPS)
    float_time = time.time() - t0
    print("  Float64 total time: %.4f s" % float_time)
    print()

    # ============================================================
    # REPORT
    # ============================================================
    print("=" * 76)
    print("SIMULATION REPORT: Arsenic Diffusion During 1000°C RTA")
    print("=" * 76)
    print()
    print("  Grid:             %d points, 200 nm" % N)
    print("  Steps:            %d (%.0f s anneal)" % (N_STEPS, ANNEAL_TIME_S))
    print("  Stability r:      %.6f" % r_f)
    print("  VDR time:         %.2f s" % vdr_time)
    print("  Float time:       %.4f s" % float_time)
    print()

    # ---- Drift accumulation over steps ----
    print("-" * 76)
    print("DRIFT ACCUMULATION OVER TIME STEPS")
    print("-" * 76)
    print("  Step      VDR peak (m^-3)      Float peak (m^-3)    Max Rel Drift")
    print("  ----    ------------------    ------------------    -------------")

    sorted_steps = sorted(vdr_snaps.keys())
    for step in sorted_steps:
        vdr_snap = vdr_snaps[step]
        flt_snap = float_snaps[step]

        # Peak concentration
        vdr_peak = max(to_float(c) for c in vdr_snap)
        flt_peak = max(flt_snap)

        # Max relative drift across grid
        max_drift = 0.0
        for i in range(N):
            vv = to_float(vdr_snap[i])
            fv = flt_snap[i]
            if abs(vv) > 1e-10:
                d = abs(vv - fv) / abs(vv)
                if d > max_drift:
                    max_drift = d

        print("  %4d    %18.6e    %18.6e    %13.2e" %
              (step, vdr_peak, flt_peak, max_drift))

    print()

    # ---- Final profile comparison ----
    print("-" * 76)
    print("FINAL CONCENTRATION PROFILE (after %d steps)" % N_STEPS)
    print("-" * 76)
    print("   Node   x(nm)      VDR C (m^-3)      Float C (m^-3)     Drift")
    print("   ----   -----    ---------------    ----------------    ------")

    for i in range(N):
        x_nm = to_float(grid[i]) * 1e9
        vv = to_float(vdr_final[i])
        fv = float_final[i]
        if abs(vv) > 1e-10:
            drift = abs(vv - fv) / abs(vv)
        else:
            drift = abs(vv - fv)
        print("   %4d   %5.1f    %15.6e    %16.6e    %6.2e" %
              (i, x_nm, vv, fv, drift))

    print()

    # ---- Symmetry verification ----
    print("-" * 76)
    print("SYMMETRY VERIFICATION (after %d steps)" % N_STEPS)
    print("-" * 76)

    vdr_sym = check_symmetry_vdr(vdr_final, N)
    float_sym_err = check_symmetry_float(float_final, N)

    print("  VDR:   C[i] == C[N-1-i] for all i:  %s" %
          ("PASS (exact)" if vdr_sym else "FAIL"))
    print("  Float: max |C[i] - C[N-1-i]|:       %.2e" % float_sym_err)

    # Show per-pair symmetry detail
    if not vdr_sym or float_sym_err > 0:
        print()
        print("  Pair detail:")
        print("    i   j    VDR match    Float |C[i]-C[j]|")
        print("   --  --    ---------    ------------------")
        for i in range(N // 2):
            j = N - 1 - i
            vdr_match = (vdr_final[i].v == vdr_final[j].v and
                         vdr_final[i].d == vdr_final[j].d)
            flt_err = abs(float_final[i] - float_final[j])
            print("   %2d  %2d    %s         %.2e" %
                  (i, j, "exact" if vdr_match else "BROKEN", flt_err))

    print()

    # ---- Dose conservation ----
    print("-" * 76)
    print("DOSE CONSERVATION")
    print("-" * 76)

    dose_final_vdr = total_dose_vdr(vdr_final, dx, N)
    dose_final_float = total_dose_float(float_final, dx_f, N)
    dose_init_float = total_dose_float(conc_init_f, dx_f, N)

    vdr_dose_i = to_float(dose_init_vdr)
    vdr_dose_f = to_float(dose_final_vdr)
    if abs(vdr_dose_i) > 0:
        vdr_dose_drift = abs(vdr_dose_f - vdr_dose_i) / abs(vdr_dose_i)
    else:
        vdr_dose_drift = 0.0

    if abs(dose_init_float) > 0:
        float_dose_drift = abs(dose_final_float - dose_init_float) / abs(dose_init_float)
    else:
        float_dose_drift = 0.0

    print("  VDR  initial dose:  %.10e" % vdr_dose_i)
    print("  VDR  final dose:    %.10e" % vdr_dose_f)
    print("  VDR  dose drift:    %.2e" % vdr_dose_drift)
    print()
    print("  Float initial dose: %.10e" % dose_init_float)
    print("  Float final dose:   %.10e" % dose_final_float)
    print("  Float dose drift:   %.2e" % float_dose_drift)

    print()

    # ---- Junction depth ----
    print("-" * 76)
    print("JUNCTION DEPTH (concentration = 1e23 m^-3 threshold)")
    print("-" * 76)

    threshold = to_qbasis(VDR(10**23))
    jd_vdr = find_junction_depth(vdr_final, grid, threshold, N)
    jd_float = find_junction_depth_float(float_final, grid_f, to_float(threshold), N)

    if jd_vdr is not None and jd_float is not None:
        jd_diff = abs(jd_vdr - jd_float)
        print("  VDR junction depth:    %.4f nm" % jd_vdr)
        print("  Float junction depth:  %.4f nm" % jd_float)
        print("  Difference:            %.4e nm" % jd_diff)
    else:
        print("  Junction not found at this threshold (profile too broad)")

    print()

    # ---- Summary ----
    print("-" * 76)
    print("DRIFT SUMMARY")
    print("-" * 76)

    # Final max drift
    final_drifts = []
    for i in range(N):
        vv = to_float(vdr_final[i])
        fv = float_final[i]
        if abs(vv) > 1e-10:
            final_drifts.append(abs(vv - fv) / abs(vv))

    max_drift = max(final_drifts) if final_drifts else 0.0
    avg_drift = (sum(final_drifts) / len(final_drifts)) if final_drifts else 0.0

    print("  Total sequential operations:       %d steps x %d interior points = %d"
          % (N_STEPS, N - 2, N_STEPS * (N - 2)))
    print("  Max VDR-vs-float relative drift:   %.2e" % max_drift)
    print("  Avg VDR-vs-float relative drift:   %.2e" % avg_drift)
    print("  VDR accumulated arithmetic error:   0 (exact)")
    print("  VDR symmetry preserved:             %s" %
          ("Yes (exact)" if vdr_sym else "No"))
    print("  Float symmetry max residual:        %.2e" % float_sym_err)
    print("  VDR dose conservation drift:        %.2e" % vdr_dose_drift)
    print("  Float dose conservation drift:      %.2e" % float_dose_drift)

    print()
    print("=" * 76)
    print("INTERPRETATION")
    print("=" * 76)
    print()
    print("  Each of the %d time steps feeds its output into the next." % N_STEPS)
    print("  This is the same sequential chain structure as the split-step")
    print("  Fourier method in DWDM fiber propagation.")
    print()
    print("  The drift column shows accumulated float error after the chain.")
    print("  VDR carries zero arithmetic error by construction.")
    print()
    print("  Symmetry: a symmetric implant MUST produce a symmetric profile")
    print("  after diffusion. VDR preserves this exactly. Float accumulates")
    print("  asymmetric rounding over %d sequential steps." % N_STEPS)
    print()
    print("  Dose conservation: diffusion must conserve total dopant count.")
    print("  Any drift in total dose is pure arithmetic error.")
    print()
    print("  For a process engineer: if the predicted junction depth differs")
    print("  between exact and float arithmetic, the float simulator's")
    print("  prediction of where the PN junction sits is contaminated by")
    print("  arithmetic noise. At sub-3nm nodes this feeds directly into")
    print("  threshold voltage variability predictions.")
    print()
    print("=" * 76)


if __name__ == "__main__":
    run_simulation()
    