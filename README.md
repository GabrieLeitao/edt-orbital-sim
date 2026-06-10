# EDT Deorbiting Simulation: High-Fidelity Hybrid Multi-Body System

## Project Objective
This project provides a high-fidelity Python simulation of an Active Debris Removal (ADR) mission. The scenario involves a **100kg Spacecraft (SC)** that has captured a **target satellite** using a net and rope. To perform the deorbiting maneuver, the SC deploys a **Electrodynamic Tether (EDT)** with a tip mass.

## Mathematical Methods & High-Fidelity Assumptions

The simulation follows advanced methodologies from scientific literature (e.g., *Zhong & Zhu, 2014*, *ProSEDS Mission Reports*) to ensure both physical realism and numerical stability.

### 1. Multi-Body Dynamics (Lumped Mass Model)
The system is modeled as a chain of **7 point masses** (default: $N_{edt}=5$) in Earth-Centered Inertial (ECI) coordinates.
- **Node Indices:** Node 0 (Tip) -> Nodes 1-4 (EDT Beads) -> Node 5 (Spacecraft) -> Node 6 (Target).
- **Material-Based Stiffness:** Unlike basic models using arbitrary springs, this simulation derives stiffness ($k = EA/L$) from real material properties:
  - **EDT:** 70 GPa Aluminum (1.5mm diameter).
  - **Rope:** 100 GPa Kevlar (2mm diameter).
- **Rayleigh (Proportional) Damping:** Implements stiffness-proportional damping ($c = \beta k$). This allows the use of high-fidelity stiffness while suppressing high-frequency numerical "chatter" (the bouncing effect) without sacrificing physical accuracy.
- **Smooth-Slack Transition:** Replaces the discontinuous `max(0, tension)` with a sigmoid-scaled smooth transition. This eliminates numerical shocks when the tether retightens.

### 2. Environmental Forces
- **Gravity (J2):** Includes Earth's central gravity and the **J2 perturbation** (zonal harmonic $J_2 = 1.0826 \times 10^{-3}$).
- **Magnetic Field:** Implements the **IGRF-2000 Model** (up to Degree 4). This captures the Earth's non-dipole components and the South Atlantic Anomaly (SAA).
- **Atmospheric Drag:** **Multi-layer Exponential Model** (Vallado/US Standard Atmosphere 1976) with **Harris-Priester Diurnal Bulge** correction.

## Mission Configurations
The simulation now supports two initial alignment modes, selectable at the start of a new mission:
- **Perpendicular (Default):** The Spacecraft and Target are separated in-track (horizontal), with the EDT deployed radially inward. This is the standard "Gravity-Gradient stable" starting point for most tether missions.
- **Radial:** All components (Target, SC, and EDT) are aligned along the local vertical. The Target is farthest from Earth, and the EDT tip is closest. This mode tests the system's response to a purely vertical deployment.

## Environmental Fidelity & Scientific Suitability

### Is this good for scientific orbital simulation?
This engine is designed as a **high-fidelity multi-body dynamics simulator** for tethered systems. With the recent integration of IGRF and multi-layer atmospheric models, it provides significant scientific value for LEO mission analysis.

| Feature | Current Model | Scientific Requirement | Impact |
| :--- | :--- | :--- | :--- |
| **Gravity** | J2 (Oblateness) | EGM96 (70x70 harmonics) | J2 captures 99% of perturbations. Missing higher terms affects sub-meter precision over months. |
| **Magnetic Field** | IGRF-2000 (Deg 4) | IGRF-13 | IGRF-2000 Deg 4 captures >99% of the field strength. Excellent for Lorentz force fidelity. |
| **Atmosphere** | Multi-layer Exp + Bulge | NRLMSISE-00 | Current model captures primary altitude and diurnal trends. NRLMSISE adds solar flux (F10.7) sensitivity. |
| **Third Body** | None | Sun/Moon/SRP | Required for high-altitude (MEO/GEO) or long-duration LEO missions. |

**Verdict:** 
- **For Tether Dynamics & Deorbiting Research:** Excellent. The combination of multi-body coupling, material-specific damping, IGRF magnetic fields, and dynamic atmospheric density makes this a robust tool for studying EDT performance.
- **For Precision Navigation:** Good, but requires EGM96 gravity and full IGRF-13/solar-flux models for centimeter-level accuracy over long epochs.

### Note: The "Snapping" & "Slingshot" Phenomenon
If you observe the EDT "snapping" or "slingshotting" the spacecraft:
1. **Libration Instability:** The Lorentz force acts as a non-conservative drag. If the current is too high, the tether swings ("librates") away from the vertical. If it swings past the stable limit, it may go slack and then violently "whip" back when tension returns.
2. **Numerical Stiffness:** The aluminum EDT is extremely stiff ($E = 70$ GPa). Small displacements cause massive forces. The `smooth_tension` function in `dynamics.py` mitigates this, but extreme maneuvers may still trigger sharp transients.
3. **Remedy:** Reduce `I_edt` in `params.py` or increase `beta_edt` (damping) to stabilize the system.

### 3. Numerical Integration
Four compiled integrators live in `integrators.py`; they keep the integration loop entirely outside the Python interpreter, eliminating the per-RHS-call overhead that dominated the previous `scipy.solve_ivp` profile.

- **`RK45`** (default): Adaptive Dormand-Prince 5(4) implemented in pure `@njit`, with FSAL and cubic-Hermite dense output. Best for the smooth-orbit regime — roughly 2× faster than the previous scipy RK45 path.
- **`LSODA`**: numbalsoda's compiled LSODA, called via a `@cfunc` RHS adapter. Implicit BDF for the stiff aluminum EDT modes.
- **`RADAU`**: Scipy's Radau (implicit) implementation, useful for extremely stiff systems where BDF methods are preferred.
- **`VERLET`**: numba compiled Fixed-step Velocity Verlet integrator of order 2

## Mission Configurations
The simulation supports multiple initial alignment modes:
- **Perpendicular (Default):** The Spacecraft and Target are separated in-track (horizontal), with the EDT deployed radially inward. This is the standard "Gravity-Gradient stable" starting point for most tether missions.
- **Radial:** All components (Target, SC, and EDT) are aligned along the local vertical.
- **Full In-Track (SC_EDT_TARGET only):** The whole SC-EDT-Target chain is laid along the velocity direction; Target leading, SC trailing.

## File Structure
- `src/`: Core source code for the simulation.
    - `params.py`: Configuration for material properties and system masses.
    - `dynamics.py`: Core physics engine with Rayleigh damping, smooth-slack logic, and J2/Drag/Lorentz forces.
    - `engine.py`: Initialization kernel and ODE driver.
    - `analysis.py`: Telemetry engine and mission result calculation.
    - `analyze_decay.py`: Batch analysis tool for fitting performance laws ($D = c + k \frac{L^2}{M}$).
    - `visualize.py`: Interactive 2x2 dashboard.
    - `stability.py`: Runs preflight stress test to check margins and if method is stable.
    - `simulate.py`: Main entry point with CLI/Questionary interface.
- `tests/`: Validation and test scripts.
    - `validate_physics.py`: Checks structural integrity and energy conservation.
    - `convergence_test.py`: Numerical convergence analysis.

## Building and Running

### Prerequisites
Install dependencies using pip:
```bash
pip install -r requirements.txt
```
Note: using a virtual environment is recommended, with e.g.:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. **Validate Physics**:
   ```bash
   python tests/validate_physics.py
   ```
   *Checks structural integrity and energy conservation in a conservative scenario.*
2. **Run Mission**:
   ```bash
   python src/simulate.py
   ```
   *Propagates the full deorbiting mission. Supports periodic binary checkpointing for lossless resume.*

   **Resume an interrupted run:**
   ```bash
   python src/simulate.py --resume
   ```

   **Batch/CLI Overrides:**
   ```bash
   python src/simulate.py --target-mass 500 --edt-length 1000 --inclination 45 --control
   ```

   **No checkpoint:**
   To skip periodic checkpointing, use:
   ```bash
   python src/simulate.py --no-checkpoint
   ```
   **Select integrator:**
   ```bash
   python src/simulate.py --method RK45    # Numba Dormand-Prince 5(4), default
   python src/simulate.py --method LSODA   # numbalsoda LSODA
   python src/simulate.py --method RADAU   # Scipy Radau
   ```
3. **Visualize**:
   ```bash
   python src/visualize.py
   ```
   *Interactive scrubbing with inverted Radial axis (Down = Earth) for standard physics interpretation.*

4. **Batch Simulations**:
   ```bash
   python batch_run.py
   ```
   *Executes a parallelized parameter sweep (e.g., Mass, Length, Inclination) across multiple CPU cores for large-scale mission analysis.*


## Assumptions
- Constant current along the EDT.
- Rigid-body rotational dynamics of the satellites are neglected (point-mass approximation).


## TODO
- [x] plot current on EDT
- [x] plot Lorentz force and air drag on same graph
- [x] altitude km throughout orbit
- [x] high-fidelity environment (IGRF + Multi-layer Drag)
- [ ] test with higher sigmoid (25 vs 50)
- [ ] test implicit model w/ vs wo/ jacobian matrix (LSODA via numbalsoda needs analytic Jacobian wired through `@cfunc`)
- [x] orbit inclination
- [x] compiled integration loop (Numba RK45 + numbalsoda LSODA)