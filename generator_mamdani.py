import itertools
import csv
import pandas as pd
from pathlib import Path

class MamdaniRuleBaseGenerator:
    # Default generator parameters
    DEFAULT_TERMS = {
        "x1": {"VL": 0.1, "L": 0.3, "M": 0.5, "H": 0.7, "VH": 0.9},
        "x2": {"L": 0.15, "M": 0.48, "H": 0.8},
        "x3": {"L": 0.1, "M": 0.35, "H": 0.65, "VH": 0.9},
        "x4": {"VH": 0.97, "H": 0.7, "M": 0.5, "L": 0.25},
        "x5": {"L": 0.15, "M": 0.42, "H": 0.65, "VH": 0.75},
        "x6": {"L": 0.15, "M": 0.42, "H": 0.55, "VH": 0.87},
    }

    DEFAULT_WEIGHTS = {
        "x1": 0.154,
        "x2": 0.225,
        "x3": 0.354,
        "x4": 0.043,
        "x5": 0.124,
        "x6": 0.1,
    }

    DEFAULT_RISK_REDUCING = {"x4", "x5", "x6"}

    def __init__(self, terms=None, weights=None, risk_reducing=None):
        self.terms = terms if terms is not None else self.DEFAULT_TERMS
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS
        self.risk_reducing = risk_reducing if risk_reducing is not None else self.DEFAULT_RISK_REDUCING
        self.factor_names = list(self.terms.keys())
        self.rules = []

    @staticmethod
    def classify_risk(r_index):
        if r_index <= 0.21:
            return "L"
        elif r_index <= 0.41:
            return "M"
        elif r_index <= 0.65:
            return "H"
        else:
            return "VH"

    def generate_rules(self):
        term_lists = [list(self.terms[f].keys()) for f in self.factor_names]
        self.rules = []

        for combo in itertools.product(*term_lists):
            r_index = 0.0
            for f_name, term_name in zip(self.factor_names, combo):
                v = self.terms[f_name][term_name]
                if f_name in self.risk_reducing:
                    v = 1.0 - v
                r_index += self.weights[f_name] * v

            consequence = self.classify_risk(r_index)
            self.rules.append({
                **dict(zip(self.factor_names, combo)),
                "R_index": round(r_index, 6),
                "R": consequence
            })
        return self.rules

    def save_local(self, path="rules_full.csv"):
        if not self.rules:
            raise ValueError("Rules not generated yet use generate_rules().")

        out_path = Path(path)
        with out_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.factor_names + ["R_index", "R"])
            writer.writeheader()
            for r in self.rules:
                writer.writerow(r)

    def compare_with_expert(self, expert_path):
        if not self.rules:
            raise ValueError("Rules not generated yet. Call generate_rules() first.")

        expert = pd.read_csv(expert_path)

        # conversion of an expert number into a term
        def num_to_term(factor, value):
            term_values = self.terms[factor]
            closest_term = min(term_values.items(), key=lambda x: abs(x[1]-value))[0]
            return closest_term

        for f in self.factor_names:
            first_value = expert[f].iloc[0]
            if not isinstance(first_value, str):
                expert[f+"_term"] = expert[f].apply(lambda v: num_to_term(f, v))
            else:
                expert[f+"_term"] = expert[f]

        # index recalculation
        def calc_r_index(row):
            r_index = 0
            for f in self.factor_names:
                v = self.terms[f][row[f+"_term"]]
                if f in self.risk_reducing:
                    v = 1 - v
                r_index += self.weights[f] * v
            return r_index

        expert["R_index_calc"] = expert.apply(calc_r_index, axis=1)
        expert["R_calc"] = expert["R_index_calc"].apply(self.classify_risk)

        auto_rules = pd.DataFrame(self.rules)
        auto_rules["key"] = auto_rules[self.factor_names].agg("-".join, axis=1)
        expert["key"] = expert[[f+"_term" for f in self.factor_names]].agg("-".join, axis=1)

        merged = pd.merge(auto_rules, expert, on="key", suffixes=("_auto", "_expert"))
        merged["match"] = merged["R_auto"] == merged["R_calc"]

        return merged



