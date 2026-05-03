# EDT Deorbiting Simulation: Hybrid Multi-Body System (Python Version)

## Project Objective
This project provides a high-fidelity Python simulation of an Active Debris Removal (ADR) mission. The scenario involves a **100kg Spacecraft (SC)** that has captured a **target satellite (<1000kg)** using a net and rope. To perform the deorbiting maneuver, the SC deploys a 2km **Electrodynamic Tether (EDT)** with a tip mass.

The simulation captures:
- **Tether Flexibility:** Discretized multi-bead model for the EDT.
- **Gravity-Gradient Stability:** The system is configured with the SC and EDT below the target for stability.
- **Lorentz Force Deorbiting:** Realistic orbital decay based on Earth's magnetic field and tether current.

## Mathematical Methods

### 1. Multi-Body Dynamics (Lumped Mass Model)
The system is modeled as a chain of point masses (beads) in Earth-Centered Inertial (ECI) coordinates. 
- **Hybrid Fidelity:** The SC-Target connection is a single viscoelastic link; the EDT is discretized into $N$ segments.
- **Non-linear Tension:** Spring-damper model with slack handling ($T = \max(0, k\Delta L + c\dot{L})$).
- **Configuration:** Stable radial stack (Earth → Tip → EDT → SC → Target).

### 2. Environmental Forces
- **Gravity & J2:** Includes the spherical Earth term and the **J2 perturbation** (zonal harmonic model) for Earth's oblateness.
- **Lorentz Force:** $F_L = \int I(dl \times B)$, using a **Tilted Dipole** magnetic field model.
- **Atmospheric Drag:** Exponential density model.

### 3. Numerical Integration
Solved using **`scipy.integrate.solve_ivp`** with the **LSODA** method, ideal for the stiff dynamics of tethered systems.

## File Structure
- `params.py`: Configuration class for masses, material properties, and orbital elements.
- `environment.py`: Modular environment engine (Magnetic field, Atmosphere).
- `dynamics.py`: Core physics engine with detailed J2, tension, and drag logic (Numba-accelerated).
- `engine.py`: Shared simulation kernel (Initialization and ODE Integration).
- `analysis.py`: Shared telemetry and data export engine (SMA, Energy, Libration).
- `frames.py`: Coordinate transformation module (ECI to LVLH).
- `simulate.py`: Lean entry point for the deorbiting mission.
- `validate_physics.py`: Lean entry point for structural and energy truth checks.
- `visualize.py`: Interactive 3D visualizer with Global (ECI), Local (Relative), and Technical (LVLH) views.
- `results/`: Directory containing all exported CSVs and plot images.
- `legacy_matlab/`: Original MATLAB implementation for historical reference.
- `requirements.txt`: Python package dependencies.

## How to Run
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Validate Physics**:
   ```bash
   python validate_physics.py
   ```
   *Outputs:* `results/validation_results.csv` and `results/validation_plots.png`.
4. **Run Simulation**:
   ```bash
   python simulate.py
   ```
   *Outputs:* `results/simulation_results.csv` and `results/simulation_plots.png`.
5. **Visualize Results**:
   ```bash
   python visualize.py
   ```
   *Features:* Dual ECI/LVLH analysis, time-scrubbing slider, and play/pause controls.

## Key Technical Features

### 1. High-Fidelity Coordinate Frames (`frames.py`)
To analyze the "double pendulum" dynamics with technical precision, the simulation includes an **LVLH (Local Vertical Local Horizontal)** transformation:
- **Radial Axis**: Always points toward Earth center.
- **In-Track Axis**: Aligned with the orbital velocity vector.
- **Stability Analysis**: The LVLH view in `visualize.py` allows you to see the tether libration without the "spinning" of the ECI frame.

### 2. Advanced Validation (`validate_physics.py`)
The system is validated through:
- **Energy Conservation**: Checks that mechanical energy is conserved to within 1e-5 in the conservative case.
- **Structural Integrity**: Monitors rope stretch and EDT curvature to ensure physical constraints are respected.

## Assumptions
- Tilted Dipole magnetic field.
- Constant current along the EDT (modularly upgradeable in `environment.py`).
- Satellites as point masses (rotational dynamics of the bodies themselves are neglected).
