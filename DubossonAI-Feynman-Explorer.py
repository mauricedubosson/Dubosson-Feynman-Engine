import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from typing import Optional, List, Union

# =============================================================================
# 1. RÉGULATEUR DE DUBOSSON (PHYSICS-INFORMED LAYER)
# =============================================================================
class DubossonRegulator(nn.Module):
    """
    Cœur du moteur DFE. Gère la persistance du champ scalaire, 
    la détection de stagnation et la réactivité aux chocs.
    """
    def __init__(self, dim: int, persistence: float = 0.9, reactivity: float = 0.1):
        super().__init__()
        self.dim = dim
        self.persistence = nn.Parameter(torch.tensor(persistence))
        self.reactivity = nn.Parameter(torch.tensor(reactivity))
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
        # Champs dynamiques (non-entraînables par gradient direct)
        self.register_buffer('scalar_field', torch.zeros(dim))
        self.register_buffer('stagnation_score', torch.zeros(dim))

    def forward(self, x: torch.Tensor, shock_signal: Optional[float] = None) -> torch.Tensor:
        if x.dim() == 1: x = x.unsqueeze(0)
        
        # 1. Gestion de la vibration de membrane (Bruit quantique de fond)
        vibration = 0.01 * torch.sin(torch.arange(self.dim, device=x.device).float())
        
        # 2. Mise à jour du champ scalaire avec persistance
        with torch.no_grad():
            # Si un choc est détecté (shock_signal), on réduit la persistance instantanément
            p = self.persistence if shock_signal is None else self.persistence * (1 - self.reactivity)
            self.scalar_field = p * self.scalar_field + (1 - p) * vibration
        
        x = x + self.scalar_field.unsqueeze(0)
        
        # 3. Activation de la Membrane de Dubosson (Sigmoïde de transition)
        # On utilise une pente raide (15) pour marquer la transition de phase
        mask = torch.sigmoid(15 * (x - self.threshold))
        x = x * mask
        
        # 4. Mécanisme d'oubli (Forgetting) si stagnation
        if self.training:
            self._update_stagnation(x)
            
        return x

    def _update_stagnation(self, x):
        with torch.no_grad():
            variation = torch.abs(x.mean(dim=0) - self.scalar_field)
            self.stagnation_score += (variation < 0.005).float()
            
            # Reset des neurones stagnants (Évitement des minima locaux)
            reset_mask = (self.stagnation_score > 100).float()
            if reset_mask.any():
                self.scalar_field = (1 - reset_mask) * self.scalar_field + reset_mask * torch.randn_like(self.scalar_field) * 0.1
                self.stagnation_score *= (1 - reset_mask)

# =============================================================================
# 2. RÉSEAU DUBOSSON-FEYNMAN (ARCHITECTURE)
# =============================================================================
class MembraneLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.regulator = DubossonRegulator(out_f)

    def forward(self, x, shock=None):
        return self.regulator(self.linear(x), shock_signal=shock)

class DubossonEngine(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = MembraneLayer(input_dim, hidden_dim)
        self.layer2 = MembraneLayer(hidden_dim, output_dim)

    def forward(self, x, shock=None):
        x = self.layer1(x, shock=shock)
        return self.layer2(x, shock=shock)

# =============================================================================
# 3. EXPLORATEUR SYMBOLIQUE (EXTRACTEUR DE LOIS)
# =============================================================================
class FeynmanExplorer:
    """
    Outil de traduction des prédictions du DFE en équations mathématiques.
    """
    def __init__(self, variables: List[str]):
        self.symbols = sp.symbols(variables)
        self.best_expr = None

    def discover(self, model: nn.Module, x_data: torch.Tensor, candidates: List[sp.Expr]):
        """
        Cherche la meilleure expression symbolique par rapport aux prédictions du modèle.
        """
        model.eval()
        with torch.no_grad():
            y_pred = model(x_data).numpy().flatten()
        
        best_error = float('inf')
        
        for expr in candidates:
            # Conversion SymPy -> Numpy pour le test
            func = sp.lambdify(self.symbols, expr, 'numpy')
            try:
                # On déballe les colonnes de x_data comme arguments
                y_sym = func(*[x_data[:, i].numpy() for i in range(x_data.shape[1])])
                error = np.mean((y_sym - y_pred)**2)
                
                if error < best_error:
                    best_error = error
                    self.best_expr = expr
            except Exception:
                continue
        
        return self.best_expr, best_error

# =============================================================================
# 4. EXEMPLE D'UTILISATION (TEST DE VALIDATION)
# =============================================================================
if __name__ == "__main__":
    # Test sur une loi physique simple : Energie Cinétique E = 0.5 * m * v^2
    m = np.random.uniform(1, 10, 500)
    v = np.random.uniform(1, 10, 500)
    e = 0.5 * m * v**2
    
    X = torch.tensor(np.stack([m, v], axis=1), dtype=torch.float32)
    Y = torch.tensor(e, dtype=torch.float32).view(-1, 1)

    # Entraînement
    engine = DubossonEngine(2, 16, 1)
    optimizer = torch.optim.Adam(engine.parameters(), lr=0.01)
    
    print("[-] Entraînement du moteur sur la loi E = 1/2 mv²...")
    for _ in range(500):
        loss = torch.mean((engine(X) - Y)**2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Découverte Symbolique
    explorer = FeynmanExplorer(['m', 'v'])
    m_sym, v_sym = explorer.symbols
    candidates = [0.5 * m_sym * v_sym**2, m_sym * v_sym, m_sym + v_sym**2]
    
    expr, err = explorer.discover(engine, X, candidates)
    print(f"\n[DÉCOUVERTE] Équation identifiée : {expr}")
    print(f"[FIABILITÉ] Erreur résiduelle : {err:.2e}")

