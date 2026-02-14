python

import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp  # Pour la régression symbolique simplifiée
import numpy as np

# Définition de DubossonAI (copiée et adaptée du code original)
class DubossonRegulator(nn.Module):
    def __init__(self, dim, persistence=0.9, cp_bias=0.0245, threshold_init=0.5, no_cp_bias=False, no_forgetting=False):
        super().__init__()
        self.dim = dim
        self.no_cp_bias = no_cp_bias
        self.no_forgetting = no_forgetting
        self.persistence = nn.Parameter(torch.tensor(persistence))
        self.cp_bias = nn.Parameter(torch.tensor(cp_bias))
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
        self.register_buffer('scalar_field', torch.zeros(dim))
        self.register_buffer('stagnation_score', torch.zeros(dim))

    def forward(self, x, loss_signal=None):
        if x.dim() == 1: x = x.unsqueeze(0)
        vibration = 0.01 * torch.sin(torch.arange(self.dim, device=x.device).float())
        with torch.no_grad():
            self.scalar_field = self.persistence * self.scalar_field + (1 - self.persistence) * vibration
        x = x + self.scalar_field.unsqueeze(0)
        if not self.no_cp_bias: x = x + self.cp_bias * torch.relu(x)
        if loss_signal is not None:
            target_threshold = 0.5 * torch.sigmoid(torch.tensor(loss_signal, device=x.device))
            self.threshold.data = 0.9 * self.threshold.data + 0.1 * target_threshold
        mask = torch.sigmoid(15 * (x - self.threshold.unsqueeze(0)))
        x = x * mask
        if self.training and not self.no_forgetting:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                variation = torch.abs(batch_mean - self.scalar_field) + (1 - batch_var.clamp(min=0.001))
                self.stagnation_score += (variation < 0.01).float()
                forget_mask = (self.stagnation_score > 50).float()
                if forget_mask.any():
                    noise_scale = 0.05 * (1 + (loss_signal or 0.0))
                    noise = torch.randn_like(self.scalar_field) * noise_scale
                    self.scalar_field = (1 - forget_mask) * self.scalar_field + forget_mask * noise
                    self.stagnation_score *= (1 - forget_mask)
        return x.squeeze(0) if x.shape[0] == 1 else x

class MembraneLayer(nn.Module):
    def __init__(self, in_features, out_features, no_cp_bias=False, no_forgetting=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.regulator = DubossonRegulator(out_features, no_cp_bias=no_cp_bias, no_forgetting=no_forgetting)

    def forward(self, x, loss_signal=None):
        x = self.linear(x)
        x = self.regulator(x, loss_signal=loss_signal)
        return x

class DubossonAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, no_cp_bias=False, no_forgetting=False):
        super().__init__()
        self.membrane1 = MembraneLayer(input_size, hidden_size, no_cp_bias=no_cp_bias, no_forgetting=no_forgetting)
        self.membrane2 = MembraneLayer(hidden_size, output_size, no_cp_bias=no_cp_bias, no_forgetting=no_forgetting)

    def forward(self, x, loss_val=None):
        x = self.membrane1(x, loss_signal=loss_val)
        x = self.membrane2(x, loss_signal=loss_val)
        return x

# Génération de données sample (basé sur un exemple Feynman : distance euclidienne)
np.random.seed(42)
num_samples = 1000
x0 = np.random.uniform(0, 10, num_samples)
x1 = np.random.uniform(0, 10, num_samples)
x2 = np.random.uniform(0, 10, num_samples)
x3 = np.random.uniform(0, 10, num_samples)
y = np.sqrt((x0 - x1)**2 + (x2 - x3)**2) + np.random.normal(0, 0.01, num_samples)  # Ajout de bruit

X = torch.tensor(np.column_stack((x0, x1, x2, x3)), dtype=torch.float32)
Y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Phase 1: Fitting avec DubossonAI (remplace le NN standard d'AI Feynman)
model = DubossonAI(input_size=4, hidden_size=16, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(500):
    pred = model(X)
    loss = criterion(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Phase 2: Régression symbolique simplifiée (inspirée d'AI Feynman, brute-force avec sympy)
x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3')
candidates = [  # Liste d'opérations basiques comme dans AI Feynman
    sp.sqrt((x0 - x1)**2 + (x2 - x3)**2),
    sp.sqrt((x0 + x1)**2 + (x2 + x3)**2),  # Mauvais candidat pour test
    (x0 - x1) + (x2 - x3)
]

best_expr, best_error = None, float('inf')
pred_y = model(X).detach().numpy().flatten()  # Utilise les prédictions de DubossonAI comme proxy

for expr in candidates:
    func = sp.lambdify((x0, x1, x2, x3), expr, 'numpy')
    sym_y = func(x0, x1, x2, x3)
    error = np.mean((sym_y - pred_y)**2)
    if error < best_error:
        best_error = error
        best_expr = expr

