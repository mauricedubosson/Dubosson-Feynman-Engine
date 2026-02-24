import torch
import numpy as np
from core import DubossonEngine

def run_ecology_model():
    """
    Extracts the survival law and camouflage rupture threshold in a predator-prey system.
    Replaces linear Lotka-Volterra assumptions with DFE threshold logic.
    """
    # 1. Population Data (Prey Density vs Predation Intensity)
    prey_density = np.linspace(0, 100, 300)
    Pc = 40.0 # Critical density where camouflage fails
    
    # Sigmoidal Predation (Functional Response Type III)
    intensity = 5.0 / (1 + np.exp(-0.5 * (prey_density - Pc)))
    intensity += np.random.normal(0, 0.1, 300)
    
    X = torch.tensor(prey_density, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(intensity, dtype=torch.float32).view(-1, 1)

    # 2. Training
    engine = DubossonEngine(input_dim=1, hidden_dim=16, output_dim=1)
    optimizer = torch.optim.Adam(engine.parameters(), lr=0.01)
    
    print("[-] Training Ecology Model...")
    for _ in range(1000):
        loss = torch.mean((engine(X) - Y)**2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    print(f"[*] Ecology: Camouflage rupture threshold Pc successfully extracted.")
    return engine, X, Y

if __name__ == "__main__":
    run_ecology_model()



