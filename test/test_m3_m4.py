import pandas as pd
from generator_mamdani import MamdaniRuleBaseGenerator
from rete_network import ReteNetwork
from anfis import ANFIS

print("TEST: MamdaniRuleBaseGenerator")
gen = MamdaniRuleBaseGenerator() # init, check type of parameters in class 
print(gen.factor_names) 

rules = gen.generate_rules() # generete
print(f"Rules count {len(rules)}.")
print("Rule example:", rules[:1])

gen.save_local("test_rules.csv") # save, if need

merged = gen.compare_with_expert("expert_rules.csv") # compare expert (exist but not need)
match_percent = merged["match"].mean() * 100
print(f"Number of rules involved in the comparison: {len(merged)}")
print(f"Percentage of matches with expert rules: {match_percent:.2f}%")

print("\nTEST: ReteNetwork")
rete = ReteNetwork(rules) # init
print(f"Rete created: {len(rete.alpha_nodes)} alpha-nodes and {len(rete.beta_nodes)} beta-nodes.") # info

fact = {"x1": "VL", "x2": "L", "x3": "M", "x4": "H", "x5": "M", "x6": "L"} # usage(exist but not need)
fired_rules = rete.run_fact(fact)
print(f"Active rules: {len(fired_rules)}")
print("Example of fired rules:", fired_rules[:5])