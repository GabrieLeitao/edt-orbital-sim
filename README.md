# EDT Deorbiting Simulation: High-Fidelity Hybrid Multi-Body System

## Project Objective
This project provides a high-fidelity Python simulation of an Active Debris Removal (ADR) mission. The scenario involves a **100kg Spacecraft (SC)** that has captured a **target satellite** using a net and rope. To perform the deorbiting maneuver, the SC deploys a **Electrodynamic Tether (EDT)** with a tip mass.

## Mathematical Methods & High-Fidelity Assumptions

The simulation follows advanced methodologies from scientific literature (e.g., *Zhong & Zhu, 2014*, *ProSEDS Mission Reports*) to ensure both physical realism and numerical stability.

### 1. Multi-Body Dynamics (Lumped Mass Model)
The system is modeled as a chain of 13 point masses in Earth-Centered Inertial (ECI) coordinates.
- **Material-Based Stiffness:** Unlike basic models using arbitrary springs, this simulation derives stiffness ($k = EA/L$) from real material properties:
  - **EDT:** 70 GPa Aluminum (1mm diameter).
  - **Rope:** 100 GPa Kevlar/Polymer (2mm diameter).
- **Rayleigh (Proportional) Damping:** Implements stiffness-proportional damping ($c = \beta k$). This allows the use of high-fidelity stiffness while suppressing high-frequency numerical "chatter" (the bouncing effect) without sacrificing physical accuracy.
- **Smooth-Slack Transition:** Replaces the discontinuous `max(0, tension)` with a sigmoid-scaled smooth transition. This eliminates numerical shocks when the tether retightens, simulating the microscopic "tightening" of molecular bonds.

### 2. Environmental Forces
- **Gravity (J2):** Includes Earth's central gravity and the **J2 perturbation** (zonal harmonic $J_2 = 1.0826 \times 10^{-3}$). This captures the primary orbital perturbations (nodal regression, perigee precession).
- **Magnetic Field:** Implements the **IGRF-2000 Model** (up to Degree 4). This captures the Earth's non-dipole components and the South Atlantic Anomaly (SAA), crucial for realistic Lorentz force calculations in LEO.
- **Atmospheric Drag:** **Multi-layer Exponential Model** (Vallado/US Standard Atmosphere 1976) with **Harris-Priester Diurnal Bulge** correction and **Earth Rotation** (relative wind). This accounts for altitude-dependent scale heights and day/night density variations (2x-5x difference in LEO).

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
Solved using **`scipy.integrate.solve_ivp`** with the **LSODA** method. The solver is tuned for the "stiff" equations of motion characteristic of high-tension tethered systems.

## File Structure
- `params.py`: Configuration for material properties (Young's Modulus, damping constants) and system masses.
- `dynamics.py`: Core physics engine with Rayleigh damping, smooth-slack logic, and J2/Drag/Lorentz forces.
- `engine.py`: Initialization kernel (Stable Gravity-Gradient configuration) and ODE driver.
- `analysis.py`: Telemetry engine for SMA decay, energy conservation, and libration analysis.
- `visualize.py`: Interactive 2x2 dashboard with ECI, Relative In-Plane (In-Track vs Radial), and LVLH 3D views.
- `stability.py`: Runs preflight stress test to check margins and if method is stable.

## Building and Running

### Prerequisites
Install dependencies using pip:
```bash
pip install -r requirements.txt
```
Note: using a virtual environment is recommended, with e.g.:
```bash
python3 -m venv .venv
```

1. **Validate Physics**:
   ```bash
   python validate_physics.py
   ```
   *Checks structural integrity and energy conservation in a conservative scenario.*
2. **Run Mission**:
   ```bash
   python simulate.py
   ```
   *Propagates the full deorbiting mission with active Lorentz forces and drag. Supports periodic binary checkpointing for lossless resume.*

   **No checkpoint:**
   To skip periodic checkpointing and intermediate CSV saves, use:
   ```bash
   python simulate.py --no-checkpoint
   ```
   **No preflight stress test:**
   To skip preflight stress validation test for material stability, use:
   ```bash
   python simulate.py --no-test
   ```
3. **Visualize**:
   ```bash
   python visualize.py
   ```
   *Interactive scrubbing with inverted Radial axis (Down = Earth) for standard physics interpretation.*


## Assumptions
- Constant current along the EDT.
- Rigid-body rotational dynamics of the satellites are neglected (point-mass approximation).


## TODO
- [x] plot current on EDT
- [x] plot Lorentz force and air drag on same graph
- [x] altitude km throughout orbit
- [x] high-fidelity environment (IGRF + Multi-layer Drag)
- [ ] test with higher sigmoid (25 vs 50)
- [ ] test implicit model w/ vs wo/ jacobian matrix 
- [ ] orbit inclination
