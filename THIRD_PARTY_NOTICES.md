# Third-party notices

O Drosobot usa e estuda software de terceiros. Este arquivo registra o que e de
quem, e o que derivamos de quem.

Os clones usados na pesquisa ficam em `research/upstream/`, que **nao e
versionado** (ver `research/upstream/.gitignore`); os commits exatos estao em
`research/UPSTREAM_LOCK.md` e podem ser reconstruidos com
`research/clone_upstream.sh`.

Um arquivo deste repositorio **e obra derivada** de codigo upstream: o kernel de
cinematica do backend GPU. Ver [Obra derivada](#obra-derivada). Todo o resto e
descricao, medicao ou implementacao independente.

---

## Dependencias em tempo de execucao

| projeto | licenca | uso |
|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) (Google DeepMind) | Apache-2.0 | motor de fisica |
| [FlyGym / NeuroMechFly](https://github.com/NeLy-EPFL/flygym) (NeLy-EPFL) | Apache-2.0 | modelo biomecanico da mosca, arenas, retina |
| `flybody` (`fruitfly.xml`, distribuido nos assets do FlyGym) | Apache-2.0 | **paleta de cores** da mosca na Unity: os valores de `body`, `lower`, `brown`, `membrane`, `red` e `bristle-brown` vem de la. A atribuicao aos nossos 68 segmentos e nossa, e esta marcada como ASSUMPTION na interface |
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

## Obra derivada

Todos derivam do **MuJoCo 3.9.0**, que e a versao do binario que roda os
experimentos. A versao importa: o `mjc_PlaneConvex` mudou de algoritmo entre
3.9 e 3.13, e portar da versao errada produz contato quase certo. Ver
`VERSAO_MUJOCO_PORTADA` em `sim/gpu_physics/compilador.py`, que recusa um
runtime divergente. A fonte de consulta e o worktree
`research/upstream/mujoco-3.9.0` (tag `3.9.0`), separado do clone principal
justamente para que "abrir o arquivo" nao volte a abrir a versao errada.

| arquivo | deriva de | licenca |
|---|---|---|
| `sim/gpu_physics/kernels/cinematica.cl` | `engine_core_smooth.c` (`mj_kinematics1`, `mj_kinematics2`) e `engine_util_spatial.c` (`mju_mulQuat`, `mju_rotVecQuat`, `mju_quat2Mat`, `mju_axisAngle2Quat`, `mju_normalize4`) | Apache-2.0, (c) DeepMind Technologies Limited |
| `sim/gpu_physics/kernels/dinamica.cl` | `engine_core_smooth.c` (`mj_comPos`, `mj_crb`, `mj_factorI`, `mj_solveLD`, `mj_comVel`, `mj_rne`), `engine_passive.c` (`mj_springdamper`), `engine_forward.c` (`mj_fwdActuation`, `mj_Euler`), `engine_util_spatial.c` (`mju_inertCom`, `mju_dofCom`, `mju_crossMotion`, `mju_crossForce`, `mju_mulInertVec`) | idem |
| `sim/gpu_physics/kernels/colisao.cl` | `engine_collision_convex.c` (`mjc_PlaneConvex`, `addplanemesh`, `mjc_meshSupport`) | idem |
| `sim/gpu_physics/kernels/restricao.cl` | `engine_core_constraint.c` (`mj_instantiateContact`, `mj_diagApprox`, `mj_makeImpedance`, `getimpedance`, `mj_referenceConstraint`), `engine_core_util.c` (`mj_jac`), `engine_util_spatial.c` (`mju_makeFrame`), `engine_core_smooth.c` (`mj_transmission`, ramo `mjTRN_BODY`) | idem |
| `sim/gpu_physics/kernels/solver.cl` | SEMANTICAS de `engine_core_constraint.c` (`mj_constraintUpdate_impl`) e `engine_solver.c` (`mj_solveNewton`): objetivo, custo e forca por linha. O ALGORITMO e proprio -- Newton denso com recuo de Armijo, nao a fatoracao unica com atualizacoes de posto 1 do original | idem |

Nao e inspiracao: e traducao de C para OpenCL C, **termo a termo**, preservando
a ordem das operacoes de ponto flutuante e os atalhos para quaternio identidade
e vetor nulo. A preservacao e deliberada -- sem ela, a comparacao com o `mjData`
mediria o porte em vez da fisica -- e torna o parentesco explicito em vez de
acidental.

O que e nosso no arquivo: a reorganizacao do percurso serial da arvore em
niveis paralelos (`sim/gpu_physics/estrutura.py`), o estado em `__local` e o
laco residente. Nada disso existe no original, que e serial por construcao.

A licenca Apache-2.0 exige preservar avisos e declarar mudancas; o cabecalho do
`.cl` nomeia a origem, e esta tabela e o aviso. A licenca completa esta em
`research/upstream/mujoco/LICENSE` e em
<https://www.apache.org/licenses/LICENSE-2.0>.

Nada do runtime do NVIDIA Warp foi portado. Os kernels do MuJoCo Warp sao
escritos no DSL do Warp e o CUDA e gerado em tempo de execucao -- nao existem
`.cu` para traduzir, entao ele so podia ser lido como referencia de engenharia,
e foi.

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
