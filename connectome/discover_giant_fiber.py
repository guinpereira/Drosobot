"""Passo 1: conecta no neuPrint e descobre o nome exato do tipo 'Giant Fiber' neste dataset.
Na literatura (hemibrain/FlyWire) o Giant Fiber costuma ser catalogado como DNp01.
Não assume — busca por padrão pra confirmar antes de montar o circuito de verdade.
"""
from pathlib import Path
from neuprint import Client, fetch_neurons, NeuronCriteria as NC

token = Path(__file__).parent.parent.joinpath(".neuprint_token").read_text().strip()

client = Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
print("Conectado:", client.fetch_version())

# Busca ampla: qualquer tipo/instância contendo "DNp01" ou "giant" (case-insensitive)
for pattern in ["DNp01", ".*[Gg]iant.*"]:
    print(f"\n--- Buscando padrão: {pattern} ---")
    try:
        neurons, roi_counts = fetch_neurons(NC(type=pattern, regex=True))
        if len(neurons) == 0:
            print("  nenhum resultado por 'type'")
        else:
            print(neurons[["bodyId", "type", "instance", "status", "pre", "post"]].to_string(index=False))
    except Exception as e:
        print("  erro:", e)
