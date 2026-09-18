"""Smoke test: confirma que Brian2 simula e neuprint-python importa, antes de mexer em dado real."""
from brian2 import NeuronGroup, StateMonitor, run, ms, mV, start_scope

start_scope()

eqs = """
dv/dt = (1.0 - v) / (10*ms) : 1
"""
G = NeuronGroup(1, eqs, method="exact")
G.v = 0
M = StateMonitor(G, "v", record=True)

run(50 * ms)

print("Brian2 OK — v final:", float(M.v[0][-1]))

import neuprint
print("neuprint-python OK — versão:", neuprint.__version__ if hasattr(neuprint, "__version__") else "importado sem erro")
