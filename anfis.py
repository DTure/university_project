import numpy as np
import pandas as pd
from typing import List, Dict
from itertools import combinations
import pickle

# Global parameters
factor_names = ["x1", "x2", "x3", "x4", "x5", "x6"]

terms_num = {
    "x1": {"VL":0.1, "L":0.3, "M":0.5, "H":0.7, "VH":0.9},
    "x2": {"L":0.15, "M":0.48, "H":0.8},
    "x3": {"L":0.1, "M":0.35, "H":0.65, "VH":0.9},
    "x4": {"L":0.25, "M":0.5, "H":0.7, "VH":0.97},
    "x5": {"L":0.15, "M":0.42, "H":0.65, "VH":0.75},
    "x6": {"L":0.15, "M":0.42, "H":0.55, "VH":0.87}
}

class BetaNode:
    def __init__(self, conditions: Dict[str, float]):
        self.conditions = conditions
        self.coeffs = self._init_coeffs()
        self.activation = 0.0
        self.frozen_coeffs = set()
    
    def _init_coeffs(self):
        coeffs = {}
        coeffs['c0'] = np.random.uniform(-0.1, 0.1)
        for f in factor_names:
            coeffs[f] = np.random.uniform(-0.1, 0.1)
        return coeffs

    def calc_z(self, x: Dict[str, float]):
        z = self.coeffs.get('c0', 0.0)
        for f in factor_names:
            z += self.coeffs.get(f, 0.0) * x[f]
        return z

    """def _init_coeffs(self):
        coeffs = {}
        coeffs['c0'] = np.random.uniform(-0.1,0.1)
        for f in factor_names:
            coeffs[f] = np.random.uniform(-0.1,0.1)
            coeffs[f+f] = np.random.uniform(-0.05,0.05)
        for (i,j) in combinations(factor_names,2):
            coeffs[i+j] = np.random.uniform(-0.05,0.05)
        return coeffs

    def calc_z(self, x: Dict[str,float]):
        z = self.coeffs.get('c0', 0.0)
        for f in factor_names:
            z += self.coeffs.get(f, 0.0) * x[f]
            z += self.coeffs.get(f+f, 0.0) * x[f]**2
        for (i,j) in combinations(factor_names,2):
            z += self.coeffs.get(i+j, 0.0) * x[i] * x[j]
        return z
    """
    
class ANFIS:
    def __init__(self, rete=None):
        self.rules: List[BetaNode] = []
        self.X, self.y = None, None
        if rete is not None:
            if not hasattr(rete, "beta_nodes"):
                raise ValueError("ANFIS waiting ReteNetwork")
            self.X, self.y = self._convert_rete_to_numpy(rete)
            self.rules = [
                BetaNode(dict(zip(factor_names, row))) for row in self.X
            ]

    def _convert_rete_to_numpy(self, rete):
        X_list = []
        y_list = []
        for beta in rete.beta_nodes:
            numeric_conditions = [terms_num[f][term] for f, term in beta.conditions.items()]
            X_list.append(numeric_conditions)
            y_list.append(float(getattr(beta, "R_index", beta.R)))
        return np.array(X_list, dtype=float), np.array(y_list, dtype=float)

    def _calc_activations(self, x: Dict[str,float]):
        for rule in self.rules:
            activation = 1.0
            for f in factor_names:
                term_value = rule.conditions[f]
                activation *= max(0.0, 1 - abs(x[f] - term_value))
            rule.activation = activation
        total = sum(r.activation for r in self.rules)
        for r in self.rules:
            r.activation /= total if total > 0 else 1.0 / len(self.rules)
    
    def initialize_coeffs(self, weights: Dict[str,float], risk_reducing: set):
        signed_weights = {k: (-v if k in risk_reducing else v) for k, v in weights.items()}
        for r in self.rules:
            for f in factor_names:
                r.coeffs[f] = signed_weights.get(f, 0.0)

    def predict(self, x: Dict[str,float]):
        self._calc_activations(x)
        y_total = sum(r.activation*r.calc_z(x) for r in self.rules)
        return 1/(1+np.exp(-y_total))

    def train_base(self, lr=0.01, epochs=50, batch_size=32):
        n = len(self.y)
        for ep in range(epochs):
            mse = 0.0
            indices = np.random.permutation(n)

            for i in range(0, n, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = self.X[batch_idx]
                y_batch = self.y[batch_idx]

                grads = [ {k:0.0 for k in r.coeffs} for r in self.rules ]

                for x_vec, y_true in zip(X_batch, y_batch):
                    x = dict(zip(factor_names, x_vec))
                    self._calc_activations(x)
                    z = sum(r.activation * r.calc_z(x) for r in self.rules)
                    y_pred = 1 / (1 + np.exp(-z))
                    error = y_true - y_pred
                    mse += error**2
                    dL_dz = -2 * error * y_pred * (1 - y_pred)

                    for r_i, r in enumerate(self.rules):
                        grad = dL_dz * r.activation
                        grads[r_i]['c0'] += grad
                        for f in factor_names:
                            grads[r_i][f] += grad * x[f]
                            #grads[r_i][f+f] += grad * (x[f]**2)
                        #for (i,j) in combinations(factor_names,2):
                           # grads[r_i][i+j] += grad * x[i] * x[j]

                for r, g in zip(self.rules, grads):
                    for k in r.coeffs:
                        if k in r.frozen_coeffs:
                            continue 
                        r.coeffs[k] -= lr * (g[k] / len(X_batch))


            mse /= n
            print(f"Epoch {ep+1}/{epochs}, MSE={mse:.6f}")


    def train_and_reduce(self, initial_epochs=50, batch_size=64, reduction_epochs=20, significance_threshold=0.05,load_filepath=None):
        if load_filepath is not None:
            self.load_model(load_filepath)
        else:
            self.train_base(lr=0.01, epochs=initial_epochs, batch_size=batch_size)

        def get_insignificant_coeffs(rule, significance_threshold):
            return [k for k,v in rule.coeffs.items() if k != 'c0' and abs(v) < significance_threshold]

        reduction_round = 0
        while reduction_round < 3:
            reduction_round += 1
            print(f"\nModel reduction, round {reduction_round}")
            removed_any = False

            for r in self.rules:
                insignificant = get_insignificant_coeffs(r, significance_threshold)
                if insignificant:
                    removed_any = True
                    for k in insignificant:
                        r.coeffs[k] = 0.0
                        r.frozen_coeffs.add(k)

            if not removed_any:
                print("All coefficients are significant. Reduction complete.")
                break
            print(self.get_global_polynomial_equation())
            print(f"Retraining the model after reduction, epochs={reduction_epochs}")
            self.train_base(lr=0.01, epochs=reduction_epochs,batch_size=batch_size)
    
    """def get_global_polynomial_equation(self):
        all_keys = ['c0'] + factor_names + [f+f for f in factor_names] + [i+j for i,j in combinations(factor_names,2)]
        global_coeffs = {k: 0.0 for k in all_keys}
        
        n_rules = len(self.rules)
        for r in self.rules:
            for k in r.coeffs:
                global_coeffs[k] += r.coeffs[k] / n_rules

        terms = []
        if global_coeffs['c0'] != 0:
            terms.append(f"{global_coeffs['c0']:.4f}")
        for f in factor_names:
            if global_coeffs[f] != 0:
                terms.append(f"{global_coeffs[f]:.4f}*{f}")
            if global_coeffs[f+f] != 0:
                terms.append(f"{global_coeffs[f+f]:.4f}*{f}^2")
        for (i,j) in combinations(factor_names,2):
            if global_coeffs[i+j] != 0:
                terms.append(f"{global_coeffs[i+j]:.4f}*{i}*{j}")
        
        equation = " + ".join(terms)
        return f"R = sigmoid({equation})"
    """
    def get_global_polynomial_equation(self):
        all_keys = ['c0'] + factor_names
        global_coeffs = {k: 0.0 for k in all_keys}
        
        n_rules = len(self.rules)
        for r in self.rules:
            for k in r.coeffs:
                global_coeffs[k] += r.coeffs[k] / n_rules

        terms = []
        if global_coeffs['c0'] != 0:
            terms.append(f"{global_coeffs['c0']:.4f}")
        for f in factor_names:
            if global_coeffs[f] != 0:
                terms.append(f"{global_coeffs[f]:.4f}*{f}")

        equation = " + ".join(terms)
        return f"R = sigmoid({equation})"


    
    def check_model_adequacy(self, rete_test):
        X_test, y_true = self._convert_rete_to_numpy(rete_test)
        X_dicts = [dict(zip(factor_names, row)) for row in X_test]
        y_pred = np.array([self.predict(x) for x in X_dicts])
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        corr = np.corrcoef(y_true, y_pred)[0,1] if len(y_true) > 1 else 0.0
        return {'MAE': mae, 'RMSE': rmse, 'Correlation': corr}
    
    def save_model(self, filepath: str):
        data = {
            'rules': [{ 'conditions': r.conditions, 'coeffs': r.coeffs } for r in self.rules],
            'factor_names': factor_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved in {filepath}")
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.rules = []
        for r_data in data['rules']:
            r = BetaNode(r_data['conditions'])
            r.coeffs = r_data['coeffs']
            self.rules.append(r)
        print(f"Model load from {filepath}")
