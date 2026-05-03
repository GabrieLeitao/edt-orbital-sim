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
- `simulate.py`: Main entry point for integration and data export.
- `visualize.py`: Interactive 3D visualizer with time-scrubbing and play/pause.
- `validate_physics.py`: Physics validation tool (Structural & Energy checks).
- `results/`: Directory containing all exported CSVs and plot images.
- `requirements.txt`: Python package dependencies.
- `README_MATLAB.md`: Original documentation for the MATLAB implementation.

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
   *Controls:* Use the slider to scrub through time, and the Play/Pause button for automatic playback.

## Key Features

### 1. Advanced Validation (`validate_physics.py`)
Beyond energy conservation, the validation script now provides a comprehensive **Structural Integrity Report**:
- **Geometric Constraints**: Monitors rope and tether stretch to ensure physical limits are respected.
- **Libration Analysis**: Tracks in-plane pitch angles to verify gravity-gradient stability.
- **Automated PASS/FAIL**: Empirically validates the dynamics engine's accuracy.

### 2. Interactive 3D Visualizer (`visualize.py`)
A modular tool that allows for detailed post-mission analysis:
- **Time Scrubbing**: "Go back in time" to analyze critical maneuvers or oscillations.
- **Dual-View**: Synchronized Global (ECI) and Relative (Target-fixed) perspectives.
- **Universal Data Support**: Automatically loads from the `results/` directory.

## Assumptions
- Tilted Dipole magnetic field.
- Constant current along the EDT (modularly upgradeable in `environment.py`).
- Satellites as point masses (rotational dynamics of the bodies themselves are neglected).
