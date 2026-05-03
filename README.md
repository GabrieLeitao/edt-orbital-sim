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
- `simulate.py`: Main entry point for integration, CSV export, and plotting.
- `visualize.py`: Standalone script for dual-view 3D animation of orbital and relative dynamics.
- `validate_physics.py`: Physics validation tool that verifies energy conservation (Zero-Current test).
- `requirements.txt`: Python package dependencies.
- `README_MATLAB.md`: Original documentation for the MATLAB implementation.

## How to Run
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Validate Physics**: (Recommended first step)
   ```bash
   python validate_physics.py
   ```
4. **Run Simulation**:
   ```bash
   python simulate.py
   ```
5. **Visualize Results**:
   ```bash
   python visualize.py
   ```

## Key Tools

### 1. Physics Validation (`validate_physics.py`)
To ensure the multi-body dynamics are mathematically correct, this script runs a "Conservative Test" with zero electrodynamic current and zero drag.
- **Metric**: It calculates the total mechanical energy (Kinetic + Gravitational Potential + Elastic Potential) for every frame.
- **Success**: A relative energy error < 1e-4 confirms that the "double pendulum" math and J2 perturbations are physically sound.

### 2. 3D Visualization Tool (`visualize.py`)
Provides a dual-view window to analyze the system's behavior:
- **Global View**: Shows the orbital path around a wireframe Earth. The camera automatically follows the system as it deorbits.
- **Relative View**: A zoomed-in view fixed on the target satellite. This is the best way to observe the **libration (swinging)** and **double-pendulum** dynamics of the SC and EDT.

## Assumptions
- Tilted Dipole magnetic field.
- Constant current along the EDT (modularly upgradeable in `environment.py`).
- Satellites as point masses (rotational dynamics of the bodies themselves are neglected).
