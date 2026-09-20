r"""
Quem sao os backends de fisica, e o que cada um declara sobre si.

A receita ja registrava `physics`. O que ela registrava era o ADAPTADOR
(`flygym1`, `flygym2`) -- util enquanto havia so uma fisica por tras de cada um.
Com um motor proprio entrando, o campo precisa distinguir duas coisas que antes
coincidiam:

    quem  MONTA  o modelo e a arena        o adaptador
    quem  RESOLVE a dinamica a cada passo  o backend fisico

`flygym2-mujoco` e `drosobot-gpu` montam o MESMO modelo -- e tem que montar, ou
a comparacao nao mede o solver, mede dois corpos diferentes. O que muda e quem
integra.

## O que o backend NAO pode mudar

O hash da ciencia. Trocar de motor fisico nao e mudanca cientifica: o corpo, o
timestep, o conectoma e a transducao continuam os mesmos, e `hash_ciencia()`
tem que sair igual dos dois lados. Se mudar, alguem mexeu no que nao devia, e
ha teste para isso (`tests/test_gpu_physics.py`).

O que o backend PRECISA registrar e outra coisa: versao, commit, precisao,
device, solver, integrador e o hash do modelo fisico. Dois motores resolvendo a
mesma ciencia ainda podem divergir numericamente, e sem esses campos a
divergencia fica sem explicacao seis meses depois.

## Nomes

    flygym1           FlyGym 1.2.1 / mujoco 3.2.7   referencia historica
    flygym2-mujoco    FlyGym 2.1.0 / mujoco 3.9     referencia atual
    drosobot-gpu      Drosobot GPU Physics          o motor proprio

`flygym1` e `flygym2` continuam aceitos e viram os nomes novos: corridas
gravadas antes desta mudanca nao podem deixar de ser legiveis.
"""
from __future__ import annotations

# nome canonico -> (adaptador que monta, familia do solver)
BACKENDS = {
    "flygym1": {
        "adaptador": "flygym1",
        "familia": "mujoco-cpu",
        "descricao": "FlyGym 1.2.1 sobre MuJoCo 3.2.7, CPU",
        "referencia": True,
    },
    "flygym2-mujoco": {
        "adaptador": "flygym2",
        "familia": "mujoco-cpu",
        "descricao": "FlyGym 2.1.0 sobre MuJoCo 3.9, CPU. Baseline de validacao.",
        "referencia": True,
    },
    "drosobot-gpu": {
        "adaptador": "flygym2",
        "familia": "drosobot-gpu",
        "descricao": ("Drosobot GPU Physics: modelo montado pelo FlyGym 2.x, "
                      "dinamica resolvida por kernels proprios em OpenCL"),
        "referencia": False,
        # Estagios que hoje rodam de fato na GPU. O resto continua no MuJoCo, e
        # `cria()` RECUSA montar este backend enquanto for assim -- um nome de
        # backend no metadata com o MuJoCo integrando por tras seria
        # exatamente o tipo de dependencia mascarada que este projeto nao pode
        # ter.
        "estagios_na_gpu": ("cinematica",),
        "estagios_no_mujoco": ("inercia", "colisao", "restricoes", "solver",
                               "atuacao", "integracao"),
    },
}


class BackendIncompleto(RuntimeError):
    """
    O backend existe como nome, mas ainda nao integra o passo inteiro.

    Levanta em vez de cair no MuJoCo em silencio. Uma corrida gravada com
    `physics: drosobot-gpu` no metadata e o MuJoCo resolvendo a dinamica seria
    um numero errado com procedencia de numero certo.
    """


# O que estava gravado antes -> o nome canonico de hoje.
APELIDOS = {
    "flygym2": "flygym2-mujoco",
    "fg1": "flygym1", "1": "flygym1",
    "fg2": "flygym2-mujoco", "2": "flygym2-mujoco",
    "gpu": "drosobot-gpu", "drosobot": "drosobot-gpu",
}


def canonico(nome: str) -> str:
    """Nome canonico do backend. Levanta com a lista se nao existir."""
    n = (nome or "flygym2-mujoco").strip().lower()
    n = APELIDOS.get(n, n)
    if n not in BACKENDS:
        raise ValueError(
            f"backend fisico desconhecido: {nome!r}. Conhecidos: "
            f"{sorted(BACKENDS)} (apelidos: {sorted(APELIDOS)})")
    return n


def adaptador_de(nome: str) -> str:
    """Qual adaptador monta o modelo para este backend."""
    n = canonico(nome)
    b = BACKENDS[n]
    faltando = b.get("estagios_no_mujoco")
    if faltando:
        raise BackendIncompleto(
            f"{n} ainda nao resolve o passo inteiro. Na GPU: "
            f"{', '.join(b['estagios_na_gpu'])}. Ainda no MuJoCo: "
            f"{', '.join(faltando)}. Rodar a bateria com este nome gravaria "
            "uma procedencia falsa. Use physics='flygym2-mujoco' e veja "
            "docs/GPU_PHYSICS.md para o estado medido de cada estagio.")
    return b["adaptador"]


def descreve(nome: str) -> dict:
    """
    O bloco que vai para o `metadata.json` da corrida.

    Sem o `mjModel` em maos (que so existe depois do `reset`), sai o que da
    para saber do nome. `completa()` acrescenta o resto.
    """
    n = canonico(nome)
    b = BACKENDS[n]
    return {
        "backend": n,
        "familia": b["familia"],
        "adaptador": b["adaptador"],
        "descricao": b["descricao"],
        "eh_referencia": b["referencia"],
    }


def completa(bloco: dict, corpo) -> dict:
    """
    Acrescenta ao bloco o que so o adaptador montado sabe.

    Versao, solver, integrador, timestep, pares de colisao e -- quando o
    backend tem um -- device e precisao. Tudo medido do objeto que vai rodar,
    nunca escrito a mao: um campo copiado a mao envelhece em silencio, que e
    justamente o que `hash_modelo` existe para evitar do outro lado.
    """
    bloco = dict(bloco)
    resumo = corpo.resumo() if corpo is not None else {}
    bloco["versao_adaptador"] = getattr(corpo, "versao", None)
    bloco["timestep_s"] = resumo.get("timestep")
    bloco["pares_colisao"] = resumo.get("collision_pairs")
    bloco["nv"] = resumo.get("nv")

    modelo = getattr(corpo, "sim", None)
    mj_model = getattr(modelo, "mj_model", None) if modelo is not None else None
    if mj_model is not None:
        from gpu_physics.compilador import RecursoNaoSuportado, compila

        try:
            mod = compila(mj_model)
            bloco["physics_model_hash"] = mod.hash_modelo
            bloco["subset_mjcf"] = mod.subset
            bloco["solver"] = mod.subset["solver"]
            bloco["integrador"] = mod.subset["integrador"]
        except RecursoNaoSuportado as e:
            # Nao e erro: o baseline do MuJoCo roda modelos que o backend GPU
            # nao cobre. Registrar POR QUE nao ha hash vale mais que omitir.
            bloco["physics_model_hash"] = None
            bloco["fora_do_subset_gpu"] = str(e)

    # o que o backend GPU acrescenta
    device = getattr(corpo, "device_gpu", None)
    if device is not None:
        bloco["device"] = device.capacidades()
        bloco["precisao"] = getattr(corpo, "precisao", None)
    return bloco
