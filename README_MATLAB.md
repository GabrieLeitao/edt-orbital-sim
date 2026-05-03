# EDT Deorbiting Simulation: Hybrid Multi-Body System

## Project Objective
This project provides a high-fidelity simulation of an Active Debris Removal (ADR) mission. The scenario involves a **100kg Spacecraft (SC)** that has captured a **target satellite (<1000kg)** using a net and rope. To perform the deorbiting maneuver, the SC deploys a 2km **Electrodynamic Tether (EDT)** with a tip mass. The simulation aims to capture the non-linear dynamics of the tether (flexibility, slackness) and the resulting orbital decay due to Lorentz forces.

## Mathematical Methods

### 1. Multi-Body Dynamics (Lumped Mass Model)
The system is modeled as a chain of point masses (beads) in Earth-Centered Inertial (ECI) coordinates. 
- **Hybrid Fidelity:** The SC-Target connection is simplified to a single viscoelastic link, while the EDT is discretized into $N$ segments to capture transverse vibrations and curvature ("skip-rope" effects).
- **Non-linear Tension:** Tension is modeled as a spring-damper system ($T = k\Delta L + c\dot{L}$) but is constrained to be non-negative ($\max(0, T)$), naturally handling tether slackness.

### 2. Environmental Forces
- **Gravity:** Includes the spherical Earth term ($\mu/r^2$) and the **J2 perturbation** to account for Earth's oblateness.
- **Lorentz Force:** Discretized across each EDT segment: $F_L = \int I(dl \times B)$. It uses a **Tilted Dipole** model for the Earth's magnetic field ($B$).
- **Atmospheric Drag:** Based on an exponential density model and relative velocity.

### 3. Numerical Integration
- **MATLAB:** The system is solved using **`ode113`**.
- **Python:** The system is solved using **`scipy.integrate.solve_ivp`** with the **LSODA** method.

## File Structure

### MATLAB Implementation
- `get_params.m`: Central configuration file.
- `environment.m`: Modular environment engine.
- `tether_dynamics.m`: The core physics engine.
- `simulate.m`: The main entry point.

### Python Implementation
- `params.py`: Central configuration class.
- `environment.py`: Modular environment engine.
- `dynamics.py`: The core physics engine.
- `simulate.py`: The main entry point.
- `requirements.txt`: Python package dependencies.

## How to Run

### MATLAB
1. Open MATLAB and navigate to the project directory.
2. Run `simulate.m`.

### Python
1. Ensure you have Python 3.8+ installed.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the simulation: `python simulate.py`

## Assumptions & Simplifications
- **KISS Principle:** The Earth's magnetic field is a tilted dipole; plasma density is modeled via an exponential decay.
- **Constant Current:** Current is assumed constant along the EDT.
- **Rigid Bodies:** Satellites are modeled as point masses.
