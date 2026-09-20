"""
Plataforma experimental do Drosobot.

    Receita     o que define um experimento: semente, escopo, arena, estimulo
    Registro    o que cada corrida deixa em disco
    roda()      executa uma receita, sem interface nenhuma
    compara()             circuito x conectoma inteiro, com numero
    compara_backends()    a mesma ciencia em dois motores fisicos, camada a
                          camada, da fisica ao desfecho

O laco de simulacao NAO mora aqui. Ele e o do `drosobot_lab.py`, o mesmo que a
interface dirige -- um segundo laco divergiria do primeiro e nenhum dos dois
seria confiavel.
"""
from .backends_comparados import compara_backends  # noqa: F401
from .corrida import roda  # noqa: F401
from .receita import (Receita, ambiente, backend_fisico,  # noqa: F401
                      hash_ciencia, metadata, parametros_cientificos,
                      versao_conectoma)
from .registro import Registro, le_resumo, lista_corridas  # noqa: F401

__all__ = ["Receita", "Registro", "roda", "le_resumo", "lista_corridas",
           "metadata", "hash_ciencia", "ambiente", "versao_conectoma",
           "parametros_cientificos", "backend_fisico", "compara_backends"]
