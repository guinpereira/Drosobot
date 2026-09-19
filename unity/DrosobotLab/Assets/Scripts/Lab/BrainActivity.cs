// Acende o CNS real conforme a atividade que chega da simulacao.
//
// A geometria vem de unity_assets/cns/cns.glb, exportado por
// blender/export_unity.py a partir da MORFOLOGIA RECONSTRUIDA por microscopia
// eletronica. Nenhum neuronio aqui foi desenhado; cada objeto neuron_<bodyId>
// e a reconstrucao daquele neuronio especifico no Male CNS.
//
// O que este script faz:
//   - liga cada bodyId ao objeto neuron_<bodyId> da cena
//   - le neural_activity e converte spike/taxa em brilho
//
// O que ele NAO faz: decidir quando um neuronio dispara. Isso vem pronto do
// circuito rodando sobre o conectoma. Aqui so existe apresentacao.
//
// ## Cor x atividade
//
// COR = papel no circuito (do metadata, escolha nossa de apresentacao).
// BRILHO = atividade (do modelo). Os dois sao separados de proposito: mudar o
// brilho nao muda de que grupo o neuronio e.
//
//     intensidade = base + ganhoSpike * exp(-idade/tau) + ganhoTaxa * taxaRecente
//
// Um spike pisca; uma populacao ativa fica luminosa. Compressao logaritmica no
// fim pra um grupo de 300 celulas nao virar uma mancha branca.

using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Drosobot.Lab
{
    public class BrainActivity : MonoBehaviour
    {
        [Header("Aparencia")]
        [Tooltip("Emissao de um neuronio em repouso")]
        public float baseIntensity = 0.30f;
        [Tooltip("Pulso no instante do spike")]
        public float spikeGain = 2.2f;
        [Tooltip("Quanto tempo o pulso do spike leva pra apagar (s)")]
        public float spikeTau = 0.18f;
        [Tooltip("Contribuicao da taxa de disparo recente (brilho sustentado)")]
        public float rateGain = 0.9f;
        [Tooltip("Janela da taxa recente (s)")]
        public float rateWindow = 0.5f;
        [Tooltip("Teto de emissao; acima disso comprime em log pra nao saturar")]
        public float softClip = 1.6f;

        [Header("Filtros")]
        public bool showNeurons = true;
        public string focusGroup = "";   // vazio = todos os grupos

        [Header("Estado")]
        public int mappedNeurons;
        public int unmappedBodyIds;

        private static readonly int EmissionColor = Shader.PropertyToID("_EmissionColor");
        // Built-in Render Pipeline usa _Color pro albedo; URP usa _BaseColor.
        // Escrevemos nos dois: o que nao existir no shader e simplesmente ignorado.
        private static readonly int ColorBuiltIn = Shader.PropertyToID("_Color");
        private static readonly int BaseColorURP = Shader.PropertyToID("_BaseColor");
        private Shader _shader;

        private class Neuron
        {
            public long bodyId;
            public string group;
            public string side;
            public string type;
            public string neurotransmitter;
            public string polarity;
            public Renderer renderer;
            public MaterialPropertyBlock block;
            public Color roleColor = Color.gray;
            public float spikeAge = 999f;   // segundos desde o ultimo spike
            public float recentRate;        // spikes/s suavizado
            public bool dimmed;
        }

        private readonly Dictionary<long, Neuron> _porBodyId = new Dictionary<long, Neuron>();
        // ordem dos bodyIds por camada, como o experiment_info declarou
        private readonly Dictionary<string, long[]> _idsPorCamada = new Dictionary<string, long[]>();
        private readonly HashSet<long> _semGeometria = new HashSet<long>();

        /// <summary>Chamado depois que o GLB estiver na cena e o metadata lido.</summary>
        public void Bind(Transform cnsRoot, NeuronMetadata metadata)
        {
            _porBodyId.Clear();

            // Material proprio em vez do que veio no FBX.
            //
            // Sem isso o cerebro inteiro saia de uma cor chapada: o material
            // importado nao responde a _EmissionColor, e no Built-in a keyword
            // _EMISSION precisa estar ligada pra emissao existir. Criando o
            // material aqui, o brilho por atividade funciona nos dois pipelines
            // e nao dependemos do que o exportador do Blender gerou.
            _shader = Shader.Find("Standard")
                      ?? Shader.Find("Universal Render Pipeline/Lit")
                      ?? Shader.Find("Sprites/Default");
            if (_shader == null)
                Debug.LogError("[brain] nenhum shader utilizavel encontrado");

            var porNome = new Dictionary<string, Renderer>();
            foreach (var r in cnsRoot.GetComponentsInChildren<Renderer>(true))
                porNome[r.gameObject.name] = r;

            foreach (var m in metadata.neurons)
            {
                if (!porNome.TryGetValue(m.objectName, out var rend)) continue;
                var cor = m.roleColor != null && m.roleColor.Length >= 3
                    ? new Color(m.roleColor[0], m.roleColor[1], m.roleColor[2])
                    : Color.gray;
                if (_shader != null)
                {
                    var mat = new Material(_shader) { color = cor * 0.5f };
                    mat.EnableKeyword("_EMISSION");
                    mat.globalIlluminationFlags = MaterialGlobalIlluminationFlags.RealtimeEmissive;
                    rend.material = mat;
                }
                var n = new Neuron
                {
                    bodyId = m.bodyId, group = m.group, side = m.side, type = m.type,
                    neurotransmitter = m.neurotransmitter, polarity = m.polarity,
                    renderer = rend, block = new MaterialPropertyBlock(), roleColor = cor,
                };
                _porBodyId[m.bodyId] = n;
            }
            mappedNeurons = _porBodyId.Count;
            Debug.Log($"[brain] {mappedNeurons} neuronios ligados ao CNS " +
                      $"(de {metadata.neurons.Length} no metadata)");
        }

        /// <summary>experiment_info: guarda a ordem dos bodyIds de cada camada.</summary>
        public void ApplyExperimentInfo(JObject msg)
        {
            _idsPorCamada.Clear();
            _semGeometria.Clear();
            var circuitos = msg["circuits"] as JArray;
            if (circuitos == null) return;
            foreach (var c in circuitos)
            {
                var nome = (string)c["name"];
                var ids = c["body_ids"] as JArray;
                if (nome == null || ids == null) continue;
                var arr = new long[ids.Count];
                for (int i = 0; i < ids.Count; i++)
                {
                    arr[i] = (long)ids[i];
                    if (!_porBodyId.ContainsKey(arr[i])) _semGeometria.Add(arr[i]);
                }
                _idsPorCamada[nome] = arr;
            }
            unmappedBodyIds = _semGeometria.Count;
            if (unmappedBodyIds > 0)
            {
                // Isto acontece de verdade e nao e bug: o experimento de looming
                // usa 1271 neuronios pre-sinapticos do GF, e o GLB so tem os 54
                // dos grupos originais. Avisamos em vez de acender algo errado.
                Debug.LogWarning($"[brain] {unmappedBodyIds} bodyIds do experimento nao tem " +
                                 "geometria no GLB; a atividade deles nao aparece no cerebro. " +
                                 "Regerar com connectome/fetch_skeletons_for_blender.py resolve.");
            }
        }

        /// <summary>neural_activity: spikes e estado por camada.</summary>
        public void ApplyNeuralActivity(JObject msg)
        {
            var camadas = msg["layers"] as JArray;
            if (camadas == null) return;

            foreach (var camada in camadas)
            {
                var nome = (string)camada["name"];
                if (nome == null || !_idsPorCamada.TryGetValue(nome, out var ids)) continue;
                var spikes = camada["spikes"] as JArray;
                if (spikes == null) continue;

                int n = Mathf.Min(ids.Length, spikes.Count);
                for (int i = 0; i < n; i++)
                {
                    int s = (int)spikes[i];
                    if (s <= 0) continue;
                    if (!_porBodyId.TryGetValue(ids[i], out var neur)) continue;
                    neur.spikeAge = 0f;
                    // taxa recente: quantos spikes por segundo, suavizada.
                    // A janela da rede e ~10 ms de mosca, entao `s` ja e contagem
                    // de janela; converter pra Hz exato exigiria saber a duracao,
                    // que nao mandamos. Usamos como intensidade relativa, nao Hz.
                    neur.recentRate += s;
                }
            }
        }

        void Update()
        {
            float dt = Time.deltaTime;
            float decaimento = Mathf.Exp(-dt / Mathf.Max(0.01f, rateWindow));

            foreach (var kv in _porBodyId)
            {
                var n = kv.Value;
                n.spikeAge += dt;
                n.recentRate *= decaimento;

                bool visivel = showNeurons &&
                               (string.IsNullOrEmpty(focusGroup) || n.group == focusGroup);
                if (n.renderer.enabled != visivel) n.renderer.enabled = visivel;
                if (!visivel) continue;

                float pulso = spikeGain * Mathf.Exp(-n.spikeAge / Mathf.Max(0.01f, spikeTau));
                float sustentado = rateGain * n.recentRate;
                float bruto = baseIntensity + pulso + sustentado;

                // Compressao suave: um grupo com centenas de celulas ativas nao
                // pode virar uma mancha branca sem forma.
                float i = bruto <= softClip
                    ? bruto
                    : softClip + Mathf.Log(1f + (bruto - softClip));

                var albedo = n.roleColor * Mathf.Clamp01(0.45f + i * 0.35f);
                n.block.SetColor(EmissionColor, n.roleColor * i);
                n.block.SetColor(ColorBuiltIn, albedo);
                n.block.SetColor(BaseColorURP, albedo);
                n.renderer.SetPropertyBlock(n.block);
            }
        }

        /// <summary>Dados do neuronio pro inspector. Devolve null se nao existir.</summary>
        public Dictionary<string, string> Describe(long bodyId)
        {
            if (!_porBodyId.TryGetValue(bodyId, out var n)) return null;
            return new Dictionary<string, string>
            {
                // DATA: medido no conectoma
                { "bodyId", n.bodyId.ToString() },
                { "type", n.type ?? "(sem dado)" },
                { "group", n.group ?? "(sem dado)" },
                { "side", n.side ?? "(sem dado)" },
                { "neurotransmitter", n.neurotransmitter ?? "(sem dado)" },
                // MODEL: derivado pela regra de Shiu et al. sobre esse dado
                { "polarity", n.polarity ?? "(sem dado)" },
                // MODEL: estado atual do modelo biofisico
                { "recentActivity", n.recentRate.ToString("F2") },
            };
        }

        public IEnumerable<long> MappedBodyIds() => _porBodyId.Keys;
    }
}
