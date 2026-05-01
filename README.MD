# Physics-Informed Neural Network (PINN) vs. Classical Numerical Methods

## Description
This project explores the intersection of Deep Learning and Numerical Analysis. It implements a **Physics-Informed Neural Network (PINN)** to solve the 1D Heat Equation and compares the results with the **Finite Difference Method (FDM)**, a classical numerical approach.

## Motivation
Traditional numerical methods (like FDM, FVM) require dense grids for high accuracy, which can be computationally expensive for complex PDEs. PINNs integrate physical laws directly into the neural network's loss function, allowing for high-fidelity solutions even with sparse data. This project demonstrates my expertise in **Scientific Machine Learning (SciML)**.

## Methods Implemented
- **PINN Solver:** Built using the `DeepXDE` library with a PyTorch backend.
- **Numerical Solver:** Finite Difference Method (FDM) implemented from scratch for baseline validation.

## Features
- Implementation of the 1D Heat Equation: $u_t = \alpha u_{xx}$
- **Physics-Informed Loss:** Embedding the PDE, initial conditions (IC), and boundary conditions (BC) into the loss function.
- **Accuracy Comparison:** Direct visualization of PINN predictions vs. FDM ground truth.
- **High Precision:** Achieved a training loss of ~9.07e-06.

## Results Summary
- PINN prediction perfectly aligns with the FDM solution at $t=0.5$.
- Demonstrated the ability of neural networks to respect physical constraints without labeled training data.
- Comparison plots show negligible error between classical and AI-driven solvers.

## Technologies Used
- **Python**
- **DeepXDE** (Scientific Machine Learning library)
- **PyTorch** (Backend for PINN)
- **NumPy & Matplotlib** (Numerical baseline and visualization)

## How to Run

1. Install required libraries: pip install -r requirements.txt

2. Run the program:
 -python pinn_ode_solver.py

## Visualization
comparison_plot.png
*Comparison: PINN vs Finite Difference Method at t=0.5*

## Related Research
This work complements my ongoing interest in numerical efficiency, as discussed in my preprint:
**"Accuracy–Efficiency Comparison of Fixed-Step and Adaptive Runge–Kutta Methods for Ordinary Differential Equations"**
DOI: https://doi.org

## Author
Md Asaduzzaman
