"""
A receita de um experimento: tudo que decide o resultado, num objeto só.

Uma corrida só é reproduzível se der para dizer, meses depois, exatamente o que
foi rodado. Não "looming com o CNS inteiro" — mas qual semente, qual versão do
conectoma, qual azimute do estímulo, quais parâmetros do LIF, qual backend.

A `Receita` é esse objeto. Ela vai inteira para o `metadata.json` de cada
corrida, junto com o ambiente medido (SO, CPU, GPU, versões) e o commit do
repositório. É o que permite responder "isto ainda dá o mesmo resultado?".

## O hash dos parâmetros científicos

`hash_ciencia()` resume, num só valor, tudo que mudaria o resultado por razão
científica: as constantes de Shiu et al., o timestep, a escala de ponto fixo, e
a transdução retina -> taxa. Duas corridas com hashes diferentes **não são
comparáveis**, ainda que tenham a mesma semente.

Ele existe porque a alternativa é comparar quinze campos a olho e errar. Não
cobre o que não é ciência -- backend, telemetria, diretório de saída -- de
propósito: trocar de GPU não invalida a comparação, e o hash não pode sugerir
que invalida.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

ARENAS = ("looming", "flat", "optomotor", "obstaculos")
ESCOPOS = ("whole", "circuit")


@dataclass(frozen=True)
class Receita:
    """O que define um experimento. Congelada: uma receita não muda no meio."""

    nome: str
    versao: str = "1"
    seed: int = 0
    physics: str = "flygym2-mujoco"
    neural: str = "opencl"
    escopo: str = "whole"
    duracao_s: float = 2.0
    arena: str = "looming"
    colisao: str = "legs"
    # Parâmetros do estímulo, por arena. Ficam num dicionário aberto porque
    # cada arena tem os seus, e fechar isso num dataclass por arena faria
    # cada condição nova mexer em três lugares.
    estimulo: dict = field(default_factory=dict)
    # Rótulo da condição dentro de uma bateria (`esquerda`, `rapido`...).
    # Separado do nome porque o nome identifica o experimento e a condição
    # identifica o ponto dentro dele.
    condicao: str = ""
    notas: str = ""

    def __post_init__(self):
        if self.arena not in ARENAS:
            raise ValueError(f"arena desconhecida: {self.arena!r} (de {ARENAS})")
        # `physics` nomeia o BACKEND FISICO, nao so o adaptador. Recusar aqui um
        # nome errado e melhor que descobrir na decima corrida de uma bateria.
        sys.path.insert(0, str(RAIZ / "sim"))
        from physics.backends import canonico

        object.__setattr__(self, "physics", canonico(self.physics))
        if self.escopo not in ESCOPOS:
            raise ValueError(f"escopo desconhecido: {self.escopo!r}")
        if self.duracao_s <= 0:
            raise ValueError("duracao_s tem que ser positiva")

    # ------------------------------------------------------------ identidade

    @property
    def id_corrida(self) -> str:
        """Nome de pasta: legível, ordenável, e único dentro de uma bateria."""
        partes = [self.nome, self.escopo]
        if self.condicao:
            partes.append(self.condicao)
        partes.append(f"seed{self.seed}")
        return "_".join(p.replace(" ", "-") for p in partes)

    def para_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def de_dict(cls, d: dict) -> "Receita":
        campos = {f for f in cls.__dataclass_fields__}
        sobra = set(d) - campos
        if sobra:
            raise ValueError(
                f"campos desconhecidos na receita: {sorted(sobra)}. "
                "Errar o nome de um campo e ficar com o padrao em silencio e "
                "pior que falhar aqui.")
        return cls(**d)


def parametros_cientificos() -> dict:
    """
    Tudo que muda o resultado por razão científica, lido do código.

    Lido, não copiado: se alguém mudar `V_TH` em `model.py`, este dicionário
    muda junto e o hash muda junto. Uma cópia manual aqui envelheceria em
    silêncio, que é o modo de falha que este arquivo existe para evitar.
    """
    sys.path.insert(0, str(RAIZ / "sim"))
    from neural import model as m

    import drosobot_lab as lab

    return {
        "lif": {
            "V_REST_mV": m.V_REST, "V_RESET_mV": m.V_RESET, "V_TH_mV": m.V_TH,
            "TAU_M_ms": m.TAU_M, "TAU_S_ms": m.TAU_S, "T_REF_ms": m.T_REF,
            "T_DELAY_ms": m.T_DELAY, "W_SYN_mV": m.W_SYN_MV,
            "DT_NEURAL_ms": m.DT_MS, "ESCALA_ponto_fixo": m.ESCALA,
        },
        "transducao": {
            "DARK_THRESHOLD": lab.DARK_THRESHOLD,
            "LOOM_GAIN": lab.LOOM_GAIN,
            "LOOM_MAX_HZ": lab.LOOM_MAX_HZ,
            "VISION_HZ": lab.VISION_HZ,
            "JANELA_MS": lab.JANELA_MS,
        },
        "fisica": {"DT_s": lab.DT},
        "motor": {
            "BASE_DRIVE": lab.BASE_DRIVE,
            "ESCAPE_DRIVE": lab.ESCAPE_DRIVE,
            "ESCAPE_MS": lab.ESCAPE_MS,
        },
    }


def hash_ciencia() -> str:
    """Resumo curto dos parâmetros científicos. Muda se a ciência mudar."""
    bruto = json.dumps(parametros_cientificos(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(bruto).hexdigest()[:16]


def versao_conectoma() -> dict:
    """
    Que conectoma está em disco. Sem ele a reprodutibilidade é ficção.

    Vem do `meta.json` gerado junto com o CSR, não de um número escrito à mão.
    """
    sys.path.insert(0, str(RAIZ / "sim"))
    try:
        from neural import metadados

        m = metadados()
        return {
            "dataset": "male-cns:v1.0",
            "neurons": m.get("neurons"),
            "edges": m.get("edges"),
            "synapses": m.get("synapses"),
            "gerado_em": m.get("gerado_em"),
        }
    except Exception as e:                                    # noqa: BLE001
        return {"erro": f"{type(e).__name__}: {e}"}


def commit_do_repo() -> str:
    """SHA do commit, ou `sujo`/`desconhecido`. Nunca levanta."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(RAIZ),
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "desconhecido"
        sujo = subprocess.run(["git", "status", "--porcelain"], cwd=str(RAIZ),
                              capture_output=True, text=True, timeout=10)
        marca = "+sujo" if sujo.stdout.strip() else ""
        return sha.stdout.strip()[:12] + marca
    except Exception:                                         # noqa: BLE001
        return "desconhecido"


def ambiente() -> dict:
    """Onde isto rodou. Vai no metadata de cada corrida."""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "commit": commit_do_repo(),
    }


def backend_fisico(receita: Receita, corpo=None) -> dict:
    """
    O que rodou a dinamica: nome, familia, versao, solver, device, precisao.

    Separado de `hash_ciencia` de proposito. Trocar de motor fisico NAO e
    mudanca cientifica -- o corpo, o timestep e o conectoma continuam os
    mesmos -- entao o hash da ciencia nao pode mexer. O que muda e isto aqui, e
    e por isto que se explica uma divergencia numerica entre dois motores.
    """
    sys.path.insert(0, str(RAIZ / "sim"))
    from physics.backends import completa, descreve

    return completa(descreve(receita.physics), corpo)


def metadata(receita: Receita, runtime: dict | None = None, corpo=None) -> dict:
    """O `metadata.json` completo de uma corrida."""
    return {
        "receita": receita.para_dict(),
        "ambiente": ambiente(),
        "conectoma": versao_conectoma(),
        "hash_ciencia": hash_ciencia(),
        "backend_fisico": backend_fisico(receita, corpo),
        "parametros_cientificos": parametros_cientificos(),
        "runtime": runtime or {},
        "escopo_declarado": {
            # O nome longo, sempre. "cerebro completo" nao aparece em lugar
            # nenhum deste projeto, e o metadata e onde alguem de fora vai ler.
            "whole": "Male CNS whole-connectome simulation",
            "circuit": "Giant Fiber circuit subgraph",
        }[receita.escopo],
        "aviso_escopo": (
            "Whole-connectome simulation is NOT a complete functional brain: "
            "only the looming pathway (LC4/LPLC2) and the motor pathway "
            "(DNp01, TTMn) have modelled sensory/motor semantics. Every other "
            "neuron participates through connectivity alone."),
    }
