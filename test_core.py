import unittest
import torch
import numpy as np
import sympy as sp
# Importing from the core directory
from core import DubossonRegulator, DubossonEngine, FeynmanExplorer

class TestDubossonEngine(unittest.TestCase):

    def setUp(self):
        """Initial configuration for each test."""
        self.input_dim = 2
        self.hidden_dim = 8
        self.output_dim = 1
        self.batch_size = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_regulator_output_shape(self):
        """Test 1: Verifies that the regulator preserves tensor dimensions."""
        reg = DubossonRegulator(dim=self.hidden_dim).to(self.device)
        x = torch.randn(self.batch_size, self.hidden_dim).to(self.device)
        output = reg(x)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim), 
                         "Regulator should not alter the input shape.")

    def test_shock_reactivity(self):
        """Test 2: Verifies that the shock signal modifies the scalar field."""
        reg = DubossonRegulator(dim=self.hidden_dim).to(self.device)
        x = torch.ones(1, self.hidden_dim).to(self.device)
        
        # Initial state capture
        initial_scalar = reg.scalar_field.clone()
        
        # Injecting a shock signal
        reg(x, shock_signal=1.0)
        new_scalar = reg.scalar_field
        
        # The scalar field must evolve differently than the default persistence
        self.assertFalse(torch.equal(initial_scalar, new_scalar), 
                         "Shock signal failed to trigger scalar field recalibration.")

    def test_engine_learning_convergence(self):
        """Test 3: Verifies that the engine can learn a fundamental linear law (y=2x)."""
        engine = DubossonEngine(1, 4, 1).to(self.device)
        optimizer = torch.optim.Adam(engine.parameters(), lr=0.1)
        
        # Data: y = 2x
        x_train = torch.linspace(0, 10, 20).view(-1, 1).to(self.device)
        y_train = 2 * x_train
        
        # Training loop (limited to 100 epochs for speed)
        for _ in range(100):
            optimizer.zero_grad()
            loss = torch.mean((engine(x_train) - y_train)**2)
            loss.backward()
            optimizer.step()
        
        final_loss = torch.mean((engine(x_train) - y_train)**2).item()
        self.assertLess(final_loss, 1.0, 
                        f"Engine failed to converge. Final Loss: {final_loss}")

    def test_symbolic_discovery_precision(self):
        """Test 4: Verifies that the FeynmanExplorer identifies the correct physical equation."""
        x_sym = sp.symbols('x')
        explorer = FeynmanExplorer(['x'])
        
        # Mock model simulating a perfect quadratic law y = x^2
        class MockModel(torch.nn.Module):
            def forward(self, x, shock=None): return x**2
            
        mock_model = MockModel()
        x_data = torch.linspace(1, 5, 10).view(-1, 1)
        candidates = [x_sym, x_sym**2, x_sym**3]
        
        best_expr, error = explorer.discover(mock_model, x_data, candidates)
        
        self.assertEqual(best_expr, x_sym**2, "FeynmanExplorer failed to identify the target equation.")
        self.assertLess(error, 1e-5, "Residual error in symbolic discovery is too high.")

if __name__ == '__main__':
    print("[-] Starting Dubosson-Feynman Engine v2.0 Validation Suite...")
    unittest.main()
