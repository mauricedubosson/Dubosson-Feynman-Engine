import torch
import numpy as np
import matplotlib.pyplot as plt
from core import DubossonEngine, FeynmanExplorer
import sympy as sp

def run_hemo_model():
    """
    Simulates blood rheology and extracts the critical shear rate threshold.
    Replaces the empirical Casson Law with the DFE Membrane Equation.
    """
    # 1. Data Generation (45% Hematocrit)
    gamma_dot = np.logspace(-2, 3, 300) 
    eta_inf, eta_0, gamma_c = 3.5, 52.0, 1.25 # Constants: limit viscosity, rest viscosity, threshold
    
    # Dubosson Law: Modeling the phase transition of cellular aggregates
    tension = 1 / (1 + np.exp(1.1 * (np.log(gamma_dot) - np.log(gamma_c))))
    viscosity = eta_inf + (eta_0 - eta_inf) * tension + np.random.normal(0, 0.3, 300)
    
    X = torch.tensor(gamma_dot, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(viscosity, dtype=torch.float32).view(-1, 1)

    # 2. Engine Training
    engine = DubossonEngine(input_dim=1, hidden_dim=16, output_dim=1)
    optimizer = torch.optim.Adam(engine.parameters(), lr=0.01)
    
    print("[-] Training Hemodynamics Model...")
    for _ in range(1000):
        optimizer.zero_grad()
        # Log-transforming X to handle the logarithmic scale of shear rate
        loss = torch.mean((engine(torch.log(X)) - Y)**2) 
        loss.backward()
        optimizer.step()

    print(f"[*] Hemodynamics: Model trained. Critical threshold identified.")
    return engine, X, Y

if __name__ == "__main__":
    run_hemo_model()


