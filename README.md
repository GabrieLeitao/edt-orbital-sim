# EDT Deorbiting Simulation: High-Fidelity Hybrid Multi-Body System

## Project Objective
This project provides a high-fidelity Python simulation of an Active Debris Removal (ADR) mission. The scenario involves a **100kg Spacecraft (SC)** that has captured a **target satellite**.

### Supported System Configurations
The simulation supports two primary hardware architectures:
1. **Direct EDT Link to the net:** The Spacecraft is connected directly to the Target satellite via the Electrodynamic Tether.
2. **Rope-Link with Dangling EDT:** The Spacecraft and Target are connected by a short (10m) non-conducting rope, while the EDT (200m) dangles from the Spacecraft with a tip mass at the end.


## Mission Configurations
The simulation supports multiple initial alignment modes, selectable at the start of a new mission:
- **Perpendicular (in System config 2 only):** The Spacecraft and Target are separated in-track (horizontal), with the EDT deployed radially inward. This is the standard "Gravity-Gradient stable" starting point for most tether missions.
- **Radial:** All components (Target, SC, and EDT) are aligned along the local vertical. The Target is farthest from Earth, and the EDT tip is closest. This mode tests the system's response to a purely vertical deployment.
- **Full In-Track (in System config 1 only):** The whole SC-EDT-Target chain is laid along the velocity direction; Target leading, SC trailing.


## Mathematical Methods & High-Fidelity Assumptions


### 1. Multi-Body Dynamics (Lumped Mass Model)
The system is modeled as a chain of **7 point masses** (including the Spacecraft and Target, $N_{edt}=5$) in Earth-Centered Inertial (ECI) coordinates.
- **Node Indices:** Node 0 (Tip) -> Nodes 1-4 (EDT Beads) -> Node 5 (Spacecraft) -> Node 6 (Target).
- **Material-Based Stiffness:** Unlike basic models using arbitrary springs, this simulation derives stiffness ($k = EA/L$) from real material properties:
  - **EDT:** 70 GPa Aluminum (1.5mm diameter).
  - **Rope:** 100 GPa Kevlar (2mm diameter).
- **Rayleigh (Proportional) Damping:** Implements stiffness-proportional damping ($c = \beta k$). This allows the use of high-fidelity stiffness while suppressing high-frequency numerical "chatter" (the bouncing effect) without sacrificing physical accuracy.
- **Smooth-Slack Transition:** Replaces the discontinuous `max(0, tension)` with a sigmoid-scaled smooth transition. This eliminates numerical shocks when the tether retightens.

### 2. Environmental Forces
- **Magnetic Field (IGRF-2000):** Implements the **International Geomagnetic Reference Field (8th Gen)** up to Degree 4. Captures Earth's non-dipole components and the South Atlantic Anomaly (SAA). *Ref: IAGA Working Group V-MOD (2000).*
- **Atmospheric Drag:** **High-Fidelity Multi-layer Exponential Model** (ref: *Vallado 2013, Table 8-4*). Uses **US Standard Atmosphere 1976** for the 0 km base, **CIRA-72** for 25–500 km, and **CIRA-72 with $T_\infty = 1000~\text{K}$** for 500–1000 km. Includes a **Harris-Priester Diurnal Bulge** correction (*ref: Harris & Priester 1962*) that models density variation lagged by ~2 hours from the sub-solar point.

### 3. Numerical Integration
Three compiled integrators live in `integrators.py`; they keep the integration loop entirely outside the Python interpreter, eliminating the per-RHS-call overhead that dominated the previous `scipy.solve_ivp` profile.

- **`RK45`** (default): Adaptive Dormand-Prince 5(4) implemented in pure `@njit`, with FSAL and cubic-Hermite dense output. Best for the smooth-orbit regime — roughly 2× faster than the previous scipy RK45 path.
- **`LSODA`**: numbalsoda's compiled LSODA, called via a `@cfunc` RHS adapter. Implicit BDF for the stiff aluminum EDT modes.
- **`VERLET`**: Numba compiled Fixed-step Velocity Verlet integrator of order 2. Critical for long-term energy conservation and libration analysis in conservative (non-dissipative) scenarios.

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
   python src/simulate.py --method VERLET  # Velocity Verlet
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

## Next Steps
- **Rigid-body dynamics:** consider moment of inertia and rigid-body dynamics of the satellites.
- **High-Degree Gravity:** Implement EGM96 (up to 70x70) to improve sub-meter precision over long durations.
- **Updated Magnetic Model:** Transition from IGRF-2000 to IGRF-13 for modern epoch accuracy.
- **Advanced Atmosphere:** Integrate NRLMSISE-00 to include sensitivity to solar flux (F10.7) and geomagnetic activity.
- **Third Body Perturbations:** Add Sun/Moon gravitational influence and Solar Radiation Pressure (SRP) for high-altitude mission support.sition from IGRF-2000 to IGRF-13 for modern epoch accuracy.
