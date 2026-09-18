"""
Confere se existe conectividade real entre as camadas candidatas do circuito
optomotor antes de montar rede: T4/T5 -> HS/VS -> DNa02 (steering), e also
LPLC2/LC4 -> DNa02 (rota alternativa de virar por ameaca, nao so fugir/pular).
"""
from pathlib import Path
from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC

token = Path(__file__).parent.parent.joinpath(".neuprint_token").read_text().strip()
client = Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=token)


def check(pre_type, post_type):
    _, conn = fetch_adjacencies(NC(type=pre_type, regex=True), NC(type=post_type, regex=True))
    total = conn["weight"].sum() if len(conn) else 0
    n_pairs = len(conn)
    print(f"{pre_type:>10} -> {post_type:<10} : {n_pairs} pares de conexao, peso total {total}")


check("T4.*", "HS.*")
check("T4.*", "VS.*")
check("T5.*", "HS.*")
check("T5.*", "VS.*")
check("HS.*", "DNa02")
check("VS.*", "DNa02")
check("LPLC2", "DNa02")
check("LC4", "DNa02")
check("DNa02", ".*")  # pra onde DNa02 manda sinal, no geral
