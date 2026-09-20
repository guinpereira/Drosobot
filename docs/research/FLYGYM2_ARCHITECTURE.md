# FlyGym 2.x: o que mudou e de onde vem o ganho

Lido em `flygym@38c8ec6` (v2.1.0-7) contra `flygym@v1.2.1`, que e a versao que o
Drosobot roda hoje. Ver `research/UPSTREAM_LOCK.md`.

Tudo aqui foi lido no codigo ou medido nesta maquina. Onde eu nao medi, esta
escrito que nao medi.

---

## Resumo em tres frases

1. O ganho de **CPU** e real, vem de tirar camada de Python do caminho quente, e
   **serve pro Drosobot**.
2. O ganho de **GPU** e de VAZAO entre simulacoes paralelas, nao de latencia de
   uma simulacao. Do jeito que o Drosobot roda -- uma mosca, laco interativo --
   ele **nao se aplica**.
3. Nenhum dos dois ataca o que de fato domina o nosso passo, que medimos aqui
   como sendo **colisao narrowphase entre malhas (87%)**.

---

## 1. Reorganizacao do pacote

| | 1.2.1 | 2.1.0 |
|---|---|---|
| raiz | `flygym/` | `src/flygym/` |
| linhas .py | 16.525 (87 arquivos) | 12.840 (59 arquivos) |
| API | Gymnasium (`step(action) -> obs, reward, ...`) | imperativa (`step()` + getters) |
| MuJoCo | via `dm_control` (`physics.bind`, `named.data`) | bindings `mujoco` diretos |
| montagem de cena | `Fly` + `Arena` | `compose/` (`fly/`, `world/`, `physics.py`, `pose.py`) |
| GPU | nao existe | `warp/` (`simulation.py`, `rendering.py`, `utils.py`) |
| Python | >=3.7 | **>=3.12,<3.15** |
| MuJoCo | 3.2.x | **>=3.9,<3.10** |

Nao e upgrade de versao menor. E outra API sobre outra base, e os nomes dos
segmentos mudaram: `LFTarsus1` virou `lf_tarsus1`. Existe tabela de traducao
pronta em `src/flygym/utils/api1to2.py` (`BODY_NAMES_OLD2NEW`), o que poupa
trabalho na migracao do Drosobot -- nossos `contact_sensor_placements` passam
direto por ela.

---

## 2. De onde vem o ganho de CPU

### O que o 1.x faz por passo de fisica

`Simulation.step()` (1.2.1, `simulation.py:180`) chama `fly.post_step()` pra cada
mosca, que chama `get_observation()` (`fly.py:1140`) -- **sempre**, a cada passo.
Dentro dela, por passo:

```python
actuated_joint_sensordata = physics.bind(self._actuated_joint_sensors).sensordata
for i, joint in enumerate(self.actuated_joints):        # laco Python
    joint_obs[:3, i] = actuated_joint_sensordata[...]
...
ang_pos = R.from_quat(quat[[1,2,3,0]]).as_euler("ZYX")  # scipy
contact_forces = physics.named.data.cfrc_ext[self.contact_sensor_placements]
for contact in physics.data.contact:                    # laco Python sobre contatos
    id_ = np.where(self._adhesion_actuator_geom_id == contact.geom1)
```

Quatro coisas caras, todas por passo:

- `physics.bind(...)` -- indexacao por nome do `dm_control`
- lacos Python sobre juntas
- `physics.named.data.cfrc_ext[lista_de_strings]` -- indexacao por nome de novo
- **laco Python sobre cada contato**, com `np.where` dentro

A `timestep` cientifica do Drosobot e `1e-4`, entao isso roda **10.000 vezes por
segundo simulado**.

### O que o 2.x faz

```python
def step(self) -> None:
    mj.mj_step(self.mj_model, self.mj_data)     # simulation.py:87, e so isso
```

As observacoes viraram **getters sob demanda**, cada um uma indexacao unica com
um array de inteiros pre-computado:

```python
def get_joint_angles(self, fly_name):
    internal_ids = self._intern_qposadrs_by_fly[fly_name]
    return self.mj_data.qpos[internal_ids]
```

A mudanca conceitual e essa: o 1.x **empurra** uma observacao completa todo
passo; o 2.x deixa voce **puxar** so o que precisa, quando precisa. Some o
`dm_control`, somem os lacos Python, some o scipy do caminho quente.

Pro Drosobot isso encaixa bem: nos so lemos retina (100 Hz) e posicao. Estamos
pagando por uma observacao completa 10.000 vezes por segundo pra usar uma fracao
dela 100 vezes.

### Quanto isso vale aqui -- medido

Modelo NeuroMechFly, `timestep=1e-4`, 3000 passos, Ryzen 7 5700X, flygym 1.2.1:

| caminho | us/passo | s de relogio por segundo simulado |
|---|---|---|
| `mj_step` puro, sem flygym | 1583 | **15,8** |
| `Simulation.step()` 1.x, sem visao | 2610 | 26,1 |
| `Simulation.step()` 1.x, visao 100 Hz | 3002 | 30,0 |

O wrapper 1.x custa **39%** do passo sem visao. A visao custa 13% do passo com
visao.

Ou seja: eliminar o overhead do wrapper leva de ~30 para ~18 s/s. **Ganho real de
~1,7x, disponivel na CPU, sem GPU nenhuma.** Nao e o "10x" que se le por ai, e a
diferenca importa: o resto do 10x depende de outras mudancas (ver secao 5 e o
documento de features de fisica).

---

## 3. De onde vem o ganho de GPU -- e por que nao e nosso

`src/flygym/warp/simulation.py` e um wrapper fino sobre `mujoco_warp`:

```python
class GPUSimulation(Simulation):
    def __init__(self, world, n_worlds, ...):
        self.mjw_model, self.mjw_data = self._mj_structs_to_mjw_structs()

    def step(self) -> None:
        mjw.step(self.mjw_model, self.mjw_data)     # e so isso
```

Os getters viram `wp.launch` de kernels de gather (`wp_gather_indexed_cols_2d`
etc., em `warp/utils.py`) que devolvem arrays `(n_worlds, ...)`.

**O parametro que manda e `n_worlds`.** O proprio benchmark oficial diz como o
numero e formado (`src/flygym_demo/benchmark/time_gpu_simulation.py`):

```python
df["steps_per_second"] = sim_steps * df["n_worlds"] / df["walltime_s"]
df["realtime_factor"]  = df["steps_per_second"] * sim_timestep
```

O `realtime_factor` publicado e **somado sobre os mundos**. O RTF de UMA mosca e
esse numero dividido por `n_worlds`. Um "300x" com 1024 mundos e ~0,3x por mosca.

Isso nao e critica ao FlyGym: pra treinar politica por RL, vazao agregada e
exatamente a metrica certa. Mas o Drosobot roda **uma** mosca num laco fechado
com o conectoma, e o que nos limita e **latencia por passo**, nao vazao. As duas
coisas nao se convertem uma na outra.

O benchmark tambem usa `wp.ScopedCapture()` (CUDA graph capture) pra amortizar o
custo de lancar kernel -- outra tecnica que so paga quando o passo e curto e
repetido milhares de vezes em paralelo.

---

## 4. Como a cena e montada (`compose/`)

```
compose/
    base.py            protocolo comum
    physics.py         opcoes de fisica (solver, integrador, tolerancias)
    pose.py            poses cinematicas (KinematicPosePreset.NEUTRAL)
    fly/
        base_fly.py    (872 linhas) base: juntas, atuadores, adesao, camera
        neuromechfly.py    o modelo que nos usamos
        flybody.py     (915) variante flybody
        musculoskeletal.py (570) atuadores musculares
    world/
        flat_ground.py, complex_terrain.py, tethered_world.py, ...
```

A montagem virou declarativa e explicita:

```python
fly = NeuroMechFly(geom_fitting_option=GeomFittingOption.UNMODIFIED)
skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,
                    joint_preset=JointPreset.LEGS_ONLY)
fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
fly.add_actuators(dofs, actuator_type=ActuatorType.POSITION, kp=50.0, ...)
fly.add_leg_adhesion()
world = FlatGroundWorld()
world.add_fly(fly, spawn_position, spawn_rotation)
```

**`GeomFittingOption` e o achado mais importante desta secao.** O benchmark
oficial tem um parametro `simplify_geom` que troca `UNMODIFIED` por
`ALL_TO_CAPSULES`. Dado o que medimos (87% do passo em colisao de malha), essa
opcao ataca exatamente o nosso gargalo -- e nao e GPU, e geometria. Ver
`NEUROMECHFLY_PHYSICS_FEATURES.md`.

Ela **muda a fisica de contato** e por isso nao pode ser ligada sem validacao.

---

## 5. Visao

`src/flygym/vision/retina.py` (276 linhas). A transformacao imagem crua ->
omatideos e Numba:

```python
@nb.njit(parallel=False)
def _raw_image_to_hex_pxls(...)
@nb.njit(parallel=True)
def _hex_pxls_to_human_readable(...)
@nb.njit(parallel=True)
def _correct_fisheye(...)
```

O 1.2.1 ja usa Numba aqui tambem, entao esta nao e uma fonte de ganho na
migracao. O custo de visao que medimos (13% do passo) e majoritariamente
**renderizar os dois olhos no MuJoCo**, nao converter para hexagonos.

---

## 6. Renderizacao

- CPU: `rendering.py` (481 linhas), `Renderer` classico.
- GPU: `warp/rendering.py` (443) com `WarpGPUBatchRenderer` e `WarpCPURenderer`.
  O batch renderer e ray-tracing sobre muitos mundos ao mesmo tempo -- de novo,
  vazao.

`modify_world_for_batch_rendering()` altera o modelo pra caber no renderer em
lote, e o codigo avisa e recompila quando isso acontece. Bom sinal de honestidade
do upstream: a modificacao nao e silenciosa.

---

## 7. Copia CPU <-> GPU

No caminho GPU, `put_model`/`put_data` acontecem **uma vez** (e de novo em
`reset()`, que recria as structs de proposito, com um comentario explicando que
`mjw.reset_data()` perderia o keyframe). Depois disso, `set_actuator_inputs`
aceita `wp.array` e nao copia; so copia se voce passar numpy.

`time` e a excecao: `float(self.mjw_data.time.numpy()[0])` faz sincronizacao a
cada leitura. Num laco que le o tempo por passo, isso serializa GPU e CPU.

Ler o estado por passo (que e o que o Drosobot faria, pra alimentar o circuito)
**desmonta** o beneficio do graph capture. Essa e uma limitacao estrutural do
laco fechado, nao um detalhe de implementacao.

---

## 8. Limitacoes que o proprio codigo declara

`GPUSimulation._strip_unsupported_options_for_mjwarp()` remove opcoes que o
MJWarp nao suporta e avisa:

```python
if (noslip_iters := world.mjcf_root.option.noslip_iterations) > 0:
    warnings.warn("MJWarp does not support noslip iterations. Changing "
                  f"option/noslip_iterations from {noslip_iters} to 0.")
```

Isto merece destaque: **o nosso modelo usa `noslip_iterations=100`** (medido).
Ir pro caminho GPU zera essa opcao, o que **muda a fisica de contato**. Nao e
uma otimizacao neutra, e uma troca -- e teria que ser validada, nao assumida.

---

## 9. O que o Drosobot pode aproveitar

| item | aproveitavel? | por que |
|---|---|---|
| getters sob demanda em vez de obs completa | **sim** | ~1,7x medido, so CPU |
| bindings `mujoco` diretos, sem `dm_control` | **sim** | parte do mesmo ganho |
| `GeomFittingOption.ALL_TO_CAPSULES` | **talvez, com validacao** | ataca os 87% |
| tabela `BODY_NAMES_OLD2NEW` | sim | migracao dos nomes de sensor |
| `GPUSimulation` / mujoco_warp | **nao, hoje** | precisa NVIDIA; e o ganho e vazao |
| batch renderer | nao | idem |
| retina Numba | ja temos equivalente | 1.2.1 tambem usa Numba |

## 10. O que ainda nao sei

- Quanto do passo 2.x na CPU e de fato mais rapido **no mesmo modelo**: nao rodei
  o 2.x nesta maquina (exige Python >=3.12 e mujoco 3.9; o nosso venv e outro).
  Os 39% de overhead do wrapper 1.x sao medidos; o ganho total da migracao e
  **estimado** a partir deles.
- Se `ALL_TO_CAPSULES` preserva a marcha. Precisa de trajetoria de referencia.
- Se MuJoCo 3.9 mudou o narrowphase o bastante pra alterar sozinho o quadro.
