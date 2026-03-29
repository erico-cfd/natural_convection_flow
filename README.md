# 2.5D Natural Convection Solver (Boussinesq Approximation)

This repository contains a Python implementation of a 2D fluid flow solver for **Natural Convection** in a square cavity. It utilizes the Boussinesq approximation to couple the velocity field with temperature gradients.

---

## Governing Equations

The solver uses the non-dimensional Navier-Stokes equations coupled with the Energy equation. The buoyancy force is driven by the Rayleigh number ($Ra$) and the Prandtl number ($Pr$).

### 1. Conservation of Momentum
The velocity field $u = (u, v)$ is governed by:

$$
\frac{\partial u}{\partial t} + (u \cdot \nabla) u = -\nabla p + Pr \nabla^2 u
$$

$$
\frac{\partial v}{\partial t} + (u \cdot \nabla) v = -\nabla p + Pr \nabla^2 v + Ra \cdot Pr \cdot \theta
$$

Where $\theta$ is the non-dimensional temperature.

### 2. Conservation of Energy
The temperature field evolution is governed by:

$$
\frac{\partial \theta}{\partial t} + (u \cdot \nabla) \theta = \nabla^2 \theta
$$

### 3. Continuity Equation
For incompressibility:

$$
\nabla \cdot u = 0
$$

---

## Numerical Scheme

* **Grid:** A **Staggered MAC Grid** is used to prevent checkerboard pressure oscillations. Velocity components $u$ and $v$ are defined on cell faces, while pressure $p$ and temperature $\theta$ are defined at cell centers.
* **Projection Method:** 1. Calculate intermediate velocities $u^*$ and $v^*$ ignoring the pressure gradient.
    2. Solve the **Poisson Pressure Equation** using Successive Over-Relaxation (SOR):
       $$\nabla^2 p = \frac{\nabla \cdot u^*}{\Delta t}$$
    3. Project the intermediate velocity onto a divergence-free space.
* **Time Stepping:** Explicit Forward Euler for advection and diffusion.
* **Stabilization:** Uses an over-relaxation parameter $\omega = 1.5$ for the iterative pressure solver.

---

## Configuration

The simulation is initialized with the following physical parameters:
* **Rayleigh Number ($Ra$):** $1000.0$
* **Prandtl Number ($Pr$):** $0.71$ (Air)
* **Mesh Size:** $25 \times 25$
* **Boundary Conditions:**
    * Left Wall ($x=0$): Hot ($T=1$)
    * Right Wall ($x=L$): Cold ($T=0$)
    * Top/Bottom Walls: Adiabatic (No heat flux)
    * All Walls: No-slip ($u=v=0$)

---

## Usage

### Dependencies
* `numpy`
* `matplotlib`
* `numba` (for potential optimization)

### Running the simulation
Simply execute the script:
```bash
python natural_convection.py
