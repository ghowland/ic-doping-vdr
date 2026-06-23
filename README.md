# ic-doping-vdr

## Exact Dopant Diffusion Simulator for IC Process Engineering

Exact arithmetic simulation of arsenic dopant diffusion during rapid thermal
annealing in silicon. The diffusion equation is time-stepped through 500
sequential Euler steps on a 1D grid, with every arithmetic operation computed
at zero accumulated error on the Q335 exact rational basis.

Built on [vdr-math](https://pypi.org/project/vdr-math/), an exact arithmetic
library where every value is an ordered triple [V, D, R] and the remainder
slot catches what conventional systems discard.

---

## Why This Matters

Every TCAD simulator in production uses 64-bit floating-point arithmetic.
Each operation introduces a rounding error of approximately 10⁻¹⁶. For a
single operation this is negligible. For a dopant diffusion simulation —
where hundreds of sequential time steps feed their outputs into the next
step across every grid point — the errors accumulate asymmetrically.

The result demonstrated here: after 500 time steps of arsenic diffusion
at 1000°C, a physically symmetric implant profile produces an asymmetric
concentration profile in float64. The asymmetry reaches **268 million
atoms per cubic meter** at specific grid points. The exact arithmetic
result preserves perfect symmetry at every grid point, at every digit,
after every step.

A symmetric device must produce a symmetric dopant profile. Float breaks
this. Exact arithmetic does not.

---

## Results

### Symmetry Preservation — The Definitive Test

A dual arsenic implant is placed symmetrically in a 200 nm silicon
device: identical Gaussian-like profiles peaked at 30 nm and 170 nm,
mirror images around the device center. After 500 diffusion steps at
1000°C, the concentration at grid point i must equal the concentration
at grid point N−1−i for all i.

```
VDR:   C[i] == C[N-1-i] for all i:  PASS (exact)
Float: max |C[i] - C[N-1-i]|:       2.68e+08 atoms/m³
```

The exact result preserves symmetry structurally — not within tolerance,
not to machine epsilon, but identically at every digit in the 101-digit
rational representation. The float result breaks symmetry at four grid
point pairs:

```
  i   j    VDR match    Float |C[i]-C[j]|
  1  38    exact         6.71e+07
  7  32    exact         2.68e+08
  8  31    exact         2.68e+08
  9  30    exact         2.68e+08
```

The asymmetry is path-dependent. Float arithmetic rounds differently
depending on the order of operations, and the left-to-right sweep
through the grid does not produce the same rounding pattern as the
right-to-left mirror. The error appears at specific grid points where
the concentration gradient interacts with the rounding pattern of the
Euler update formula.

This is not a hypothetical concern. When a TCAD simulator computes
100,000 statistically sampled transistors to characterize threshold
voltage variability, each sample runs through hundreds of diffusion
steps. If the arithmetic breaks known symmetries, the statistical
distribution of threshold voltages contains a systematic asymmetric
bias that is arithmetic noise, not physics.

### Drift Accumulation — 500 Sequential Steps

```
Step      VDR peak (m⁻³)      Float peak (m⁻³)    Max Rel Drift
   0      1.000000e+25         1.000000e+25         0.00e+00
   1      9.809875e+24         9.809875e+24         2.14e-16
  10      8.410466e+24         8.410466e+24         6.28e-16
  50      5.505507e+24         5.505507e+24         6.84e-16
 100      4.171547e+24         4.171547e+24         1.04e-15
 250      2.756143e+24         2.756143e+24         7.64e-16
 500      1.928330e+24         1.928330e+24         7.61e-16
```

The relative drift in peak concentration stays near machine epsilon
(~10⁻¹⁶) across all 500 steps. This is because the explicit Euler
diffusion step is linear: `C_new = C + r·(C_left − 2C + C_right)`
with constant coefficients. Linear operations accumulate float error
additively, not multiplicatively. The error does not amplify.

This is the expected result for a linear system. The symmetry breakage
is the more significant finding — it demonstrates that even when the
scalar drift is small, the structural properties of the solution are
corrupted.

### Dose Conservation

```
VDR  initial dose:  2.2564e+17 atoms/m²
VDR  final dose:    1.8530e+17 atoms/m²
VDR  dose drift:    1.79e-01
```

Both VDR and float show identical 17.9% dose loss. This is not an
arithmetic error — it is physical leakage through the zero-concentration
boundary conditions. The boundaries act as sinks, absorbing dopant that
diffuses to the device edges. Both simulators compute the same boundary
physics correctly. Dose conservation would hold exactly with reflecting
(Neumann) boundary conditions.

---

## The Path to Larger Drift

The linear diffusion equation keeps float drift at machine epsilon
because the update coefficients are constant. In real TCAD process
simulation, the physics is nonlinear:

### Concentration-Dependent Diffusion

Arsenic diffusion in silicon is strongly concentration-dependent above
approximately 10¹⁹ cm⁻³. The diffusion coefficient is not a constant
— it depends on the local carrier concentration, which depends on the
local dopant concentration, which is the quantity being diffused. The
update at each time step becomes:

```
C_new[i] = C[i] + dt · d/dx(D(C[i]) · dC/dx)
```

where D(C) is itself a function of C. Each step's output modifies the
next step's coefficients. This is the nonlinear feedback loop where
float errors amplify rather than merely accumulate — the same structure
as the Kerr nonlinearity in DWDM fiber propagation.

At high arsenic concentrations (>10¹⁹ cm⁻³), D(C) varies by more than
an order of magnitude across the profile. A float rounding error in C
at step n produces a wrong D at step n+1, which produces a wrong C at
step n+2, which produces a wrong D at step n+3. The error feeds back
into itself.

### Coupled Drift-Diffusion

In the presence of an electric field (from the dopant concentration
gradient itself, via Poisson's equation), the diffusion equation gains
a drift term:

```
dC/dt = d/dx(D · dC/dx) + d/dx(μ · C · E)
```

where E depends on C through Poisson's equation. This couples the
diffusion to the electrostatics, creating a second nonlinear feedback
channel. Both channels accumulate float error independently and feed
each other.

### Oxidation-Enhanced Diffusion

During thermal oxidation, interstitial silicon atoms are injected into
the substrate, enhancing dopant diffusion by a factor that depends on
the oxidation rate, temperature profile, and local defect concentration.
The coupled system of equations (dopant + interstitials + vacancies +
oxidation front) creates a chain of nonlinear dependencies where float
errors in any variable contaminate all others.

### Statistical Variability Simulation

The commercial application where arithmetic error matters most is
statistical variability analysis. A foundry simulates 100,000 randomly
doped transistors to predict the distribution of threshold voltages.
Each transistor runs through the full process simulation chain:
implantation → diffusion → oxidation → etch → deposition. If the
diffusion step introduces systematic asymmetric rounding (as
demonstrated in the symmetry test above), the threshold voltage
distribution acquires a systematic bias that is indistinguishable from
physical variability.

The question is not whether individual transistor simulations are
accurate to 10⁻¹⁶. The question is whether 100,000 transistor
simulations produce an unbiased statistical distribution. Asymmetric
rounding accumulation means they do not.

---

## How It Works

### Diffusion Equation — Explicit Euler Time-Stepping

The one-dimensional diffusion equation

```
∂C/∂t = D · ∂²C/∂x²
```

is discretized on a uniform grid with spacing Δx and time step Δt:

```
C[i]^{n+1} = C[i]^n + r · (C[i-1]^n − 2·C[i]^n + C[i+1]^n)
```

where `r = D·Δt/Δx²` is the stability parameter (must be < 0.5).

Each of the 500 time steps feeds its output concentration profile
into the next step as input. This is the sequential chain structure
— identical in form to the split-step Fourier method used in optical
fiber propagation simulation.

### Physical Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dopant | Arsenic (As) | Most common n-type for advanced nodes |
| Substrate | Silicon | Standard |
| Diffusion coefficient | 2.5 × 10⁻¹⁷ m²/s | At 1000°C RTA |
| Device length | 200 nm | |
| Grid points | 40 | 5.13 nm spacing |
| Anneal time | 10 s | Typical RTA |
| Time steps | 500 | dt = 0.02 s |
| Stability parameter | 0.019 | Well under 0.5 limit |
| Peak concentration | 10²⁵ m⁻³ | 10¹⁹ cm⁻³, heavy doping |

### Exact Arithmetic Engine

All arithmetic operates on the Q335 basis: every value is an integer
numerator over the fixed denominator 2³³⁵. Addition is one integer
add. Multiplication uses divmod to keep the denominator fixed, with
overflow captured exactly in the remainder slot. No value is ever
truncated, rounded, or approximated.

The diffusion coefficient, grid spacing, time step, and initial
concentration profile are all exact rational numbers. Every
intermediate value at every grid point at every time step is an
exact rational number with 101 digits of precision.

### Float64 Mirror

An identical algorithm runs in parallel using Python's native float64
arithmetic. The per-step drift between exact and float is reported at
selected time steps. The symmetry test compares structural properties
of the two solutions.

---

## Installation

```bash
pip install vdr-math
python ic_doping_vdr.py
```

Requires Python 3.8+ and vdr-math. No other dependencies.

---

## Extending This Work

### For Process Engineers

The simulation is structured as a template. To test your own scenario:

1. Change the diffusion coefficient to match your dopant and temperature
2. Change the initial profile to match your implant conditions
3. Change the grid and time stepping to match your process window
4. Run and check the symmetry column

If your process has a known physical symmetry or conservation law,
the exact simulator will preserve it and the float simulator may not.
The gap between them is your arithmetic noise floor.

### Adding Concentration-Dependent Diffusion

Replace the constant `D` in `diffusion_step_vdr` with a function
`D(C)` that depends on the local concentration. For arsenic in silicon:

```
D(C) = D_intrinsic · (1 + (C / C_ref)^α)
```

where C_ref ≈ 3.5 × 10²⁵ m⁻³ and α ≈ 1. This makes each time step
nonlinear: the update coefficient at each grid point depends on the
concentration, which was computed at the previous step. This is where
float drift should grow from 10⁻¹⁶ to commercially significant levels.

### Adding the Poisson Coupling

Solve Poisson's equation at each time step to get the electric field
from the dopant concentration, then add the drift term to the diffusion
update. This couples the system nonlinearly and creates the second
feedback channel for error amplification.

### Scaling to 3D

The 1D simulator demonstrates the arithmetic. A 3D atomistic simulator
would place individual dopant atoms at random positions in a transistor
geometry and solve the coupled drift-diffusion-Poisson system on a
tetrahedral mesh. The sequential chain length grows from 500 to tens of
thousands of steps, and the nonlinear coupling at each step is stronger.
The exact arithmetic infrastructure demonstrated here scales to that
problem — the Q335 basis handles any rational arithmetic regardless of
the dimensionality of the grid.

---

## Relationship to DWDM Results

This work follows the same methodology as
[dwdm-vdr](https://github.com/...),
which demonstrated 9.6% float drift after 5 split-step propagation
steps in a DWDM fiber simulation. The DWDM result showed larger drift
because the split-step method involves nonlinear phase rotation (the
Kerr effect), where each step's output modifies the refractive index
seen by the next step. The dopant diffusion result shows smaller scalar
drift but definitive symmetry breakage because the linear diffusion
equation accumulates float error additively rather than multiplicatively.

The structural parallel:

| | DWDM | Dopant Diffusion |
|--|------|-----------------|
| Sequential chain | Split-step Fourier | Euler time-stepping |
| Steps | 5 | 500 |
| Nonlinearity | Kerr effect (strong) | Constant D (none) |
| Scalar drift | 9.6% | ~10⁻¹⁶ |
| Symmetry test | Channel pairs exact | Grid mirror exact |
| Float symmetry | Broken | Broken (2.68e+08) |
| Key result | Drift changes engineering answer | Asymmetry in what must be symmetric |

The path forward for IC doping is to add the nonlinear physics
(concentration-dependent D, Poisson coupling) that creates the
feedback loop where float errors amplify. The infrastructure for
exact arithmetic through the simulation chain is demonstrated and
operational.

---

## License

MIT

---

## Dependencies

- [vdr-math](https://pypi.org/project/vdr-math/) — exact arithmetic library
