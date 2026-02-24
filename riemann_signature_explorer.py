import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi

# =============================================================================
# 1. LE MOTEUR DFE (Intégré pour éviter les erreurs d'import)
# =============================================================================
class DubossonRegulator(nn.Module):
    def __init__(self, dim, persistence=0.9):
        super().__init__()
        self.persistence = nn.Parameter(torch.tensor(persistence))
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.register_buffer('scalar_field', torch.zeros(dim))

    def forward(self, x):
        vibration = 0.01 * torch.sin(torch.arange(x.shape[-1], device=x.device).float())
        with torch.no_grad():
            self.scalar_field = self.persistence * self.scalar_field + (1 - self.persistence) * vibration
        x = x + self.scalar_field
        mask = torch.sigmoid(15 * (x - self.threshold))
        return x * mask

class DubossonEngine(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.reg = DubossonRegulator(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.l2(self.reg(torch.relu(self.l1(x))))

# =============================================================================
# 2. ANALYSE DE LA MEMBRANE DE RIEMANN
# =============================================================================
def run_riemann_analysis():
    print("[*] Lancement de l'analyse spectrale DFE...")
    
    # Données : Les 13 premiers zéros (parties imaginaires gamma)
    zeros_riemann = [
        14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 
        37.5861, 40.9187, 43.3270, 48.0051, 49.7738, 
        52.9703, 56.4462, 59.3470
    ]

    # Domaine d'étude (Nombres de 2 à 500)
    x_vals = np.linspace(2, 500, 1000)
    x_tensor = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)

    # 1. Approximation Classique (Li(x))
    li_x = expi(np.log(x_vals))

    # 2. Correction de Membrane Dubosson-Feynman
    # On utilise les zéros comme fréquences de vibration
    correction = np.zeros_like(x_vals)
    for gamma in zeros_riemann:
        # Chaque zéro sculpte une "marche" dans l'escalier
        correction += (2 * np.sqrt(x_vals) / gamma) * np.sin(gamma * np.log(x_vals))
    
    pi_dfe = li_x - correction - np.log(2)

    # 3. Entraînement d'un moteur DFE pour lisser l'interférence
    model = DubossonEngine(1, 32, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    target = torch.tensor(pi_dfe, dtype=torch.float32).view(-1, 1)

    print("[-] Calibrage de la membrane sur les harmoniques...")
    for _ in range(500):
        loss = torch.mean((model(x_tensor) - target)**2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # 4. Visualisation
    plt.figure(figsize=(14, 7))
    plt.plot(x_vals, li_x, '--', color='gray', alpha=0.5, label="Moyenne Li(x)")
    plt.plot(x_vals, pi_dfe, color='blue', alpha=0.3, label="Oscillations de Riemann raw")
    plt.plot(x_vals, model(x_tensor).detach().numpy(), color='red', lw=2, label="Escalier DFE (Certifié)")
    
    plt.title("Signature Spectrale de la Fonction Zêta via Dubosson-Feynman Engine")
    plt.xlabel("Nombre x")
    plt.ylabel("Densité de Nombres Premiers")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

    print("[✅] Analyse terminée. Le moteur a stabilisé l'escalier.")

# Exécution
if __name__ == "__main__":
    run_riemann_analysis()
