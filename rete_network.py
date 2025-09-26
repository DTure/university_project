import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class FuzzyReteNode:
    def __init__(self):
        self.output_nodes: List['FuzzyReteNode'] = []

    def connect(self, node: 'FuzzyReteNode'):
        self.output_nodes.append(node)

class AlphaNode(FuzzyReteNode):
    def __init__(self, factor: str, term: str):
        super().__init__()
        self.factor = factor
        self.term = term

    def activate(self, fact: Dict[str, str]):
        if fact.get(self.factor) == self.term:
            for node in self.output_nodes:
                node.activate(self.factor, True)

class BetaNode(FuzzyReteNode):
    def __init__(self, conditions: Dict[str, str], R: str, R_index: float):
        super().__init__()
        self.conditions = conditions
        self.R = R
        self.R_index = R_index
        self.matched_conditions = {k: False for k in conditions}

    def activate(self, fact_name: str, matched: bool):
        self.matched_conditions[fact_name] = matched
        if all(self.matched_conditions.values()):
            for node in self.output_nodes:
                node.activate(self.R)

    def reset(self):
        self.matched_conditions = {k: False for k in self.conditions}

class OutputNode:
    def __init__(self):
        self.activated_rules: List[str] = []

    def activate(self, R: str):
        self.activated_rules.append(R)

    def reset(self):
        self.activated_rules = []

class ReteNetwork:
    def __init__(self, rules_data):
        self.alpha_nodes: Dict[Tuple[str,str], AlphaNode] = {}
        self.beta_nodes: List[BetaNode] = []
        self.output_node = OutputNode()

        if isinstance(rules_data, pd.DataFrame):
            rules_df = rules_data
        elif isinstance(rules_data, list):
            rules_df = pd.DataFrame(rules_data)
        else:
            raise ValueError("rules_data must be DataFrame or list[dict]")

        for idx, row in rules_df.iterrows():
            conditions = {f"x{i+1}": row[f"x{i+1}"] for i in range(6)}
            beta_node = BetaNode(conditions, R=row["R"], R_index=row["R_index"])
            beta_node.connect(self.output_node)
            self.beta_nodes.append(beta_node)

            for factor, term in conditions.items():
                key = (factor, term)
                if key not in self.alpha_nodes:
                    self.alpha_nodes[key] = AlphaNode(factor, term)
                self.alpha_nodes[key].connect(beta_node)

    def run_fact(self, fact: Dict[str,str]) -> List[str]:
        self.output_node.reset()
        for beta in self.beta_nodes:
            beta.reset()
        for (factor, term), alpha in self.alpha_nodes.items():
            alpha.activate(fact)
        return self.output_node.activated_rules