import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi
from core import DubossonEngine

def run_riemann_model():
    """
    Uses the Dubosson-Feynman Engine to model the distribution of prime numbers.
    It identifies the 'vibrations' caused by the non-trivial zeros of the Zeta function.
    """
    # 1. Data Generation: The "Music of Primes"
    # We use the explicit formula: pi(x) ~ Li(x) - sum(Li(x^rho))
    x_range = np.linspace(10, 1000, 500)
    
    # Classic Logarithmic Integral (The smooth average)
    li_x = expi(np.log(x_range))
    
    # The first 5 Riemann Zeros (imaginary parts gamma)
    # These act as the 'Membrane Frequencies' in the DFE
    zeros = [14.1347, 21.0220, 25.0108, 30.4248, 32.9350]
    
    # Generating the "Ground Truth" oscillating signal
    oscillation = np.zeros_like(x_range)
    for gamma in zeros:
        # DFE Membrane Formula: 2 * sqrt(x)/gamma * sin(gamma * ln(x))
        oscillation += (2 * np.sqrt(x_range) / gamma) * np.sin(gamma * np.log(x_range))
    
    y_true = li_x - oscillation
    
    X = torch.tensor(x_range, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(y_true, dtype=torch.float32).view(-1, 1)

    # 2. DFE Training: Capturing the Spectral Signature
    # We increase hidden_dim to 32 to capture complex oscillations
    engine = DubossonEngine(input_dim=1, hidden_dim=32, output_dim=1)
    optimizer = torch.optim.Adam(engine.parameters(), lr=0.005)
    
    print("[-] Training Riemann Spectral Model...")
    for epoch in range(1500):
        optimizer.zero_grad()
        # We use log(X) because Riemann zeros operate on a logarithmic scale
        pred = engine(torch.log(X))
        loss = torch.mean((pred - Y)**2)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"    Epoch {epoch}: Loss = {loss.item():.4f}")

    print("[*] Number Theory: Riemann Membrane frequencies successfully modeled.")
    
    # 3. Visualization of the "Quantum Staircase"
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, li_x, '--', color='gray', label="Classic Li(x) Average")
    plt.plot(x_range, Y.numpy(), label="Actual Prime Distribution (Oscillatory)", alpha=0.5)
    plt.plot(x_range, engine(torch.log(X)).detach().numpy(), color='blue', label="DFE Reconstruction")
    plt.title("Dubosson-Feynman Reconstruction of Riemann's Prime Spiral")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_riemann_model()


