# Benchmarks neurais

Cada arquivo traz o hardware junto do numero. Numero sem hardware nao serve.

| arquivo | o que e |
|---|---|
| `lif_cpu_numpy_*.json` | passo LIF denso em NumPy, Ryzen 7 5700X |
| `lif_unity_compute_*.json` | o mesmo passo em HLSL/D3D12, RX 6700 XT |
| `lif_gpu_estado.json` | estado final da GPU, lido por `tests/test_lif_gpu_equivalencia.py` |

Reproduzir:

```
.venv\Scripts\python benchmarks\neural\bench_cpu.py
unity cmd eval --code "Drosobot.Compute.LifBenchmark.RodarTudo()"
unity cmd eval --code "Drosobot.Compute.LifBenchmark.Validar(1000, 400)"
.venv\Scripts\python tests\test_lif_gpu_equivalencia.py
```

Resumo e interpretacao: `docs/research/README.md`.

O que estes numeros NAO cobrem: propagacao sinaptica esparsa (125M arestas),
custo de readback num laco fechado, e qualquer coisa de fisica.
