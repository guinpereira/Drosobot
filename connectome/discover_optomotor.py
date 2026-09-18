"""
Passo 1 do circuito 2 (optomotor/visao): descobrir quais tipos classicos da
literatura (T4/T5 = deteccao de movimento, LPLC2/LC4 = looming, HS/VS = wide-field
tangential cells, DNa02 = descending "steering") realmente existem com esse nome
no dataset Male CNS -- mesma logica que usamos pra confirmar DNp01 (Giant Fiber).
Nao assume nada, so confirma antes de gastar tempo montando rede em cima de nome errado.
"""
from pathlib import Path
from neuprint import Client, fetch_neurons, NeuronCriteria as NC

token = Path(__file__).parent.parent.joinpath(".neuprint_token").read_text().strip()
client = Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=token)

candidates = [
    "T4.*", "T5.*",           # deteccao de movimento direcional (classico, todo conectoma tem)
    "LPLC2", "LC4", "LC6",    # detectores de looming/objeto
    "HS.*", "VS.*",           # wide-field tangential cells (optomotor classico)
    "DNa0[12]", "DNa03",      # descending "steering" (usado em modelos de otimotor recentes)
]

for pattern in candidates:
    try:
        neurons, _ = fetch_neurons(NC(type=pattern, regex=True))
        n = len(neurons)
        if n == 0:
            print(f"{pattern:>10}  -- 0 resultado")
        else:
            types_found = neurons["type"].value_counts()
            print(f"{pattern:>10}  -- {n} neuronios | tipos: {dict(types_found)}")
    except Exception as e:
        print(f"{pattern:>10}  -- erro: {e}")
