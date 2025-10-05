import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
import itertools
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

factor_names = ["x1", "x2", "x3", "x4", "x5", "x6"]

terms_num = {
    "x1": {"VL":0.1, "L":0.3, "M":0.5, "H":0.7, "VH":0.9},
    "x2": {"L":0.15, "M":0.48, "H":0.8},
    "x3": {"L":0.1, "M":0.35, "H":0.65, "VH":0.9},
    "x4": {"L":0.25, "M":0.5, "H":0.7, "VH":0.97},
    "x5": {"L":0.15, "M":0.42, "H":0.65, "VH":0.75},
    "x6": {"L":0.15, "M":0.42, "H":0.55, "VH":0.87}
}

def expand_features(x_vec: torch.Tensor, degree: int = 1):
    if x_vec.dim() == 1:
        x_vec = x_vec.unsqueeze(0)
    B, n = x_vec.shape
    features = [x_vec] 

    if degree == 2:
        squares = x_vec ** 2
        features.append(squares)
        cross_terms = []
        for i, j in itertools.combinations(range(n), 2):
            cross_terms.append((x_vec[:, i] * x_vec[:, j]).unsqueeze(1))
        cross_terms = torch.cat(cross_terms, dim=1) if cross_terms else None
        if cross_terms is not None:
            features.append(cross_terms)

    full_features = torch.cat(features, dim=1)
    return full_features

def get_feature_names(degree: int = 1):
    names = factor_names.copy()
    if degree == 2:
        names += [f"{f}^2" for f in factor_names]
        for i, j in itertools.combinations(factor_names, 2):
            names.append(f"{i}*{j}")
    return names

class BetaNode(nn.Module):
    def __init__(self, conditions: Dict[str, float], poly_degree: int = 1):
        super().__init__()
        self.conditions = conditions
        self.poly_degree = poly_degree
        expanded_dim = len(get_feature_names(poly_degree)) + 1
        coeff_init = torch.randn(expanded_dim) * 0.1
        self.coeffs = nn.Parameter(coeff_init.to(device))
        self.frozen_mask = torch.ones_like(self.coeffs, device=device)
        self.activation = 0.0

class ANFIS(nn.Module):
    def __init__(self, rete=None, poly_degree: int = 1):
        super().__init__()
        self.rules: List[BetaNode] = nn.ModuleList()
        self.X, self.y = None, None
        self.poly_degree = poly_degree
        self.feature_names = get_feature_names(poly_degree)

        if rete is not None:
            self.X, self.y = self._convert_rete_to_tensor(rete)
            for row in self.X:
                cond_dict = dict(zip(factor_names, row.tolist()))
                self.rules.append(BetaNode(cond_dict, poly_degree))

    def initialize_coeffs(self, weights: Dict[str, float], risk_reducing: set):
        with torch.no_grad():
            for r in self.rules:
                for i, f in enumerate(factor_names):
                    val = weights.get(f, 0.0)
                    if f in risk_reducing:
                        val = -val
                    r.coeffs[i+1].copy_(torch.tensor(val, device=device))

    def _convert_rete_to_tensor(self, rete):
        X_list, y_list = [], []
        for beta in rete.beta_nodes:
            numeric_conditions = [terms_num[f][term] for f, term in beta.conditions.items()]
            X_list.append(numeric_conditions)
            y_list.append(float(getattr(beta, "R_index", beta.R)))
        X_tensor = torch.tensor(X_list, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_list, dtype=torch.float32, device=device)
        return X_tensor, y_tensor

    def _calc_activations_batch(self, X_batch):
        batch_size = X_batch.size(0)
        n_rules = len(self.rules)
        X_exp = X_batch.unsqueeze(1).expand(batch_size, n_rules, len(factor_names))  
        rule_terms = torch.tensor([[rule.conditions[f] for f in factor_names] 
                                   for rule in self.rules], device=device)
        rule_terms_exp = rule_terms.unsqueeze(0).expand(batch_size, n_rules, len(factor_names))
        activations = 1 - torch.abs(X_exp - rule_terms_exp)
        activations = torch.clamp(activations, min=0.0)
        activations = torch.prod(activations, dim=2)
        activations = activations / activations.sum(dim=1, keepdim=True)  
        return activations

    def _calc_z_batch(self, X_batch):
        X_poly = expand_features(X_batch, self.poly_degree)
        coeffs_matrix = torch.stack([r.coeffs * r.frozen_mask for r in self.rules], dim=0)
        c0 = coeffs_matrix[:, 0]
        cf = coeffs_matrix[:, 1:]
        activations = self._calc_activations_batch(X_batch)
        z_rules = torch.matmul(X_poly, cf.t()) + c0.unsqueeze(0)
        z_total = (activations * z_rules).sum(dim=1)
        return z_total

    def predict_batch(self, X_batch):
        z_total = self._calc_z_batch(X_batch)
        return torch.sigmoid(z_total)

    def predict(self, x: Dict[str,float]):
        X_tensor = torch.tensor([x[f] for f in factor_names], dtype=torch.float32, device=device).unsqueeze(0)
        return self.predict_batch(X_tensor)[0]

    def train_model(self, lr=0.01, epochs=50, batch_size=32):
        if self.X is None or self.y is None:
            raise ValueError("No data")
        optimizer = optim.Adam(self.parameters(), lr=lr)
        n = self.X.size(0)

        for ep in range(epochs):
            perm = torch.randperm(n)
            mse_epoch = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                X_batch, y_batch = self.X[idx], self.y[idx]
                optimizer.zero_grad()
                y_pred = self.predict_batch(X_batch)
                loss = ((y_batch - y_pred) ** 2).mean()
                loss.backward()
                for r in self.rules:
                    if hasattr(r, 'frozen_mask'):
                        r.coeffs.grad *= r.frozen_mask
                optimizer.step()
                mse_epoch += loss.item() * X_batch.size(0)
            mse_epoch /= n
            print(f"Epoch {ep+1}/{epochs}, MSE={mse_epoch:.6f}")

    def train_and_reduce(self, initial_epochs=50, batch_size=32, reduction_epochs=20, significance_threshold=0.05, load_filepath=None):
        if load_filepath is not None:
            self.load_model(load_filepath)
        else:
            self.train_model(lr=0.01, epochs=initial_epochs, batch_size=batch_size)
        
        max_rounds = 5
        round_idx = 0

        while round_idx < max_rounds:
            print(f"\nModel reduction, round {round_idx+1}")
            print("Global equation:")
            print(self.get_global_polynomial_equation())

            global_coeffs = self.get_global_coefficients()
            insignificant = [i for i, c in enumerate(global_coeffs) if abs(c) < significance_threshold]

            new_insignificant = []
            for i in insignificant:
                if any(r.frozen_mask[i].item() == 1 for r in self.rules):
                    new_insignificant.append(i)

            if not new_insignificant:
                print("Reduction complete.")
                break

            print(f"Removing coefficients with indices: {insignificant} (|value| < {significance_threshold})")

            for r in self.rules:
                with torch.no_grad():
                    for i in insignificant:
                        r.frozen_mask[i] = 0 

            self.train_model(lr=0.01, epochs=reduction_epochs, batch_size=batch_size)
            round_idx += 1

    
    def get_global_coefficients(self):
        coeff_len = len(self.rules[0].coeffs)
        global_coeffs = torch.zeros(coeff_len, device=self.rules[0].coeffs.device)
        for r in self.rules:
            global_coeffs += (r.coeffs * r.frozen_mask).detach()     
        global_coeffs /= len(self.rules)
        return global_coeffs

    def get_global_polynomial_equation(self):
        n_rules = len(self.rules)
        coeffs_sum = torch.zeros(len(self.feature_names)+1, device=device)
        for r in self.rules:
            coeffs_sum += (r.coeffs * r.frozen_mask).detach()
        coeffs_avg = coeffs_sum / n_rules
        terms = []
        if coeffs_avg[0].item() != 0:
            terms.append(f"{coeffs_avg[0].item():.4f}")
        for i, f in enumerate(self.feature_names):
            if coeffs_avg[i+1].item() != 0:
                terms.append(f"{coeffs_avg[i+1].item():.4f}*{f}")
        return f"R = sigmoid({' + '.join(terms)})"

    def check_model_adequacy(self, rete_test):
        X_test, y_true = self._convert_rete_to_tensor(rete_test)
        y_pred = self.predict_batch(X_test)
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        rmse = torch.sqrt(torch.mean((y_true - y_pred)**2)).item()
        corr = torch.corrcoef(torch.stack([y_true, y_pred]))[0,1].item() if len(y_true) > 1 else 0.0
        return {'MAE': mae, 'RMSE': rmse, 'Correlation': corr}

    def save_model(self, filepath: str):
        data = {
            'rules': [{ 'conditions': r.conditions, 'coeffs': r.coeffs.detach().cpu()} for r in self.rules],
            'factor_names': factor_names,
            'poly_degree': self.poly_degree
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved in {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.rules = nn.ModuleList()
        self.poly_degree = data.get('poly_degree', 1)
        self.feature_names = get_feature_names(self.poly_degree)
        for r_data in data['rules']:
            r = BetaNode(r_data['conditions'], poly_degree=self.poly_degree)
            r.coeffs = nn.Parameter(r_data['coeffs'].to(device))
            r.frozen_mask = torch.ones_like(r.coeffs, device=device)
            self.rules.append(r)
        print(f"Model loaded from {filepath}")


