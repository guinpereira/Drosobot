# Third-party notices

O Drosobot usa e estuda software de terceiros. Este arquivo registra o que e de
quem, e o que derivamos de quem.

Nenhum codigo upstream foi copiado para este repositorio. Os clones usados na
pesquisa ficam em `research/upstream/`, que **nao e versionado** (ver
`research/upstream/.gitignore`); os commits exatos estao em
`research/UPSTREAM_LOCK.md` e podem ser reconstruidos com
`research/clone_upstream.sh`.

---

## Dependencias em tempo de execucao

| projeto | licenca | uso |
|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) (Google DeepMind) | Apache-2.0 | motor de fisica |
| [FlyGym / NeuroMechFly](https://github.com/NeLy-EPFL/flygym) (NeLy-EPFL) | Apache-2.0 | modelo biomecanico da mosca, arenas, retina |
| [Brian2](https://github.com/brian-team/brian2) | CeCILL-2.1 | simulador spiking de referencia |
| [NumPy](https://numpy.org/) | BSD-3-Clause | numerica |
| [pandas](https://pandas.pydata.org/) | BSD-3-Clause | leitura das tabelas do conectoma |
| [Newtonsoft.Json](https://www.newtonsoft.com/json) (via `com.unity.nuget.newtonsoft-json`) | MIT | JSON na Unity |
| [navis](https://github.com/navis-org/navis), [navis-flybrains](https://github.com/navis-org/navis-flybrains) | GPL-3.0 | esqueletos e transformacoes de template (usado offline, no pipeline do Blender) |

## Dados

| fonte | condicoes |
|---|---|
| **Male CNS connectome** (`male-cns:v1.0`), Janelia / neuPrint | dados publicos de conectoma. Os CSVs em `connectome/` derivam de consultas ao neuPrint. Citar Janelia/FlyEM em qualquer publicacao. |

## Estudados para a pesquisa de compute (nao distribuidos)

Clonados em `research/upstream/` e analisados em `docs/research/`. **Nenhuma
linha foi copiada.** O que produzimos sao descricoes e medicoes.

| projeto | licenca | o que aprendemos |
|---|---|---|
| [FlyGym 2.x](https://github.com/NeLy-EPFL/flygym) | Apache-2.0 | getters sob demanda em vez de observacao empurrada; `GeomFittingOption`; composicao de cena |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 | perfil interno do passo, pares de colisao, MJX |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | Apache-2.0 | estrategia de paralelizacao por `worldid`; fusao de kernel, reducao em tile, especializacao em tempo de compilacao |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) | Apache-2.0 | modelo de device (CPU/CUDA apenas) |

### Sobre derivacao conceitual

`unity/DrosobotLab/Assets/Compute/LifStep.compute` implementa o mesmo integrador
LIF exato de `sim/fast_lif.py`, que por sua vez implementa as equacoes de
**Shiu et al. 2024, Nature 634:210**. Nao deriva de codigo do MJWarp nem do Warp.

As tecnicas de otimizacao descritas em `docs/research/MJWARP_IMPLEMENTATION.md`
(fusao de kernel, reducao em nivel de workgroup, especializacao em tempo de
compilacao, avaliacao em lote, carregamento adiado) sao **ideias de engenharia
lidas no MJWarp**, documentadas com atribuicao. Se alguma delas for implementada
aqui, o arquivo correspondente deve dizer de onde veio a ideia -- como
`LifStep.compute` ja diz de onde vem a matematica.

## Documentacao consultada

- [HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html) (AMD)
- [HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) (AMD)
- [HIP SDK for Windows -- system requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html) (AMD)
- [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/) (AMD)

## Referencias cientificas

- **Shiu, P.K., Sterne, G.R., Spiller, N. et al.** "A leaky integrate-and-fire
  computational model based on the connectome of the entire adult Drosophila
  brain reveals insights into sensorimotor processing." *Nature* **634**, 210
  (2024). -- origem dos parametros biofisicos e da convencao de sinal por
  neurotransmissor.
- **NeuroMechFly v2 / FlyGym**, NeLy-EPFL. -- modelo neuromecanico.
- **Male CNS connectome**, Janelia Research Campus / FlyEM.
