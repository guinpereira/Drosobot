// Desenha a conectividade entre os neuronios do CNS.
//
// As arestas sao DATA: vem do conectoma, agregadas por PAR de neuronios no
// export (blender/export_unity.py). O conectoma tem sinapse individual, mas
// desenhar milhoes de linhas nao informa nada -- uma aresta por par, com
// espessura proporcional ao numero de sinapses, informa.
//
// O SINAL de cada aresta (excitatoria/inibitoria) e MODEL, nao DATA: sai da
// regra de Shiu et al. aplicada ao neurotransmissor do pre-sinaptico. O
// conectoma mede o neurotransmissor, nao o efeito.
//
// A posicao de cada ponta e o centro do bounding box do neuronio -- e um resumo
// visual, NAO o local anatomico da sinapse. O conectoma sabe onde cada sinapse
// fica; nos nao estamos usando essa informacao aqui, e a linha reta entre
// centros nao deve ser lida como o trajeto do axonio.

using System.Collections.Generic;
using UnityEngine;

namespace Drosobot.Lab
{
    public class ConnectivityGraph : MonoBehaviour
    {
        public enum Filtro { Todas, Excitatorias, Inibitorias, Selecionado }

        [Header("Exibicao")]
        public bool show;
        public Filtro filtro = Filtro.Todas;
        [Tooltip("Espessura da aresta mais forte, em unidades de mundo")]
        public float espessuraMax = 0.09f;
        public float espessuraMin = 0.012f;
        [Tooltip("Pulso visual quando o pre-sinaptico dispara")]
        public bool pulsarNoSpike = true;

        [Header("Estado")]
        public int edgeCount;
        public int visibleEdges;

        private class Aresta
        {
            public long pre, post;
            public int weight;
            public int sign;
            public LineRenderer linha;
            public float pulso;      // 0..1, decai
        }

        private readonly List<Aresta> _arestas = new List<Aresta>();
        private Dictionary<long, Transform> _porBodyId;
        private BrainActivity _brain;
        private Material _matBase;

        private static readonly Color CorExcitatoria = new Color(0.40f, 0.72f, 1.00f);
        private static readonly Color CorInibitoria = new Color(1.00f, 0.42f, 0.42f);

        public void Build(Transform cnsRoot, NeuronMetadata metadata, BrainActivity brain)
        {
            _brain = brain;
            _porBodyId = new Dictionary<long, Transform>();
            foreach (var r in cnsRoot.GetComponentsInChildren<Renderer>(true))
                _porBodyId[0] = null;   // placeholder; preenchido abaixo por nome

            var porNome = new Dictionary<string, Renderer>();
            foreach (var r in cnsRoot.GetComponentsInChildren<Renderer>(true))
                porNome[r.gameObject.name] = r;

            var centro = new Dictionary<long, Vector3>();
            foreach (var n in metadata.neurons)
                if (porNome.TryGetValue(n.objectName, out var rend))
                    centro[n.bodyId] = rend.bounds.center;

            var sh = Shader.Find("Sprites/Default") ?? Shader.Find("Unlit/Color");
            _matBase = new Material(sh);

            var raiz = new GameObject("Connectivity").transform;
            raiz.SetParent(transform, false);

            int pesoMax = 1;
            if (metadata.edges != null)
                foreach (var e in metadata.edges) pesoMax = Mathf.Max(pesoMax, e.weight);

            if (metadata.edges != null)
                foreach (var e in metadata.edges)
                {
                    if (!centro.TryGetValue(e.pre, out var a)) continue;
                    if (!centro.TryGetValue(e.post, out var b)) continue;

                    var go = new GameObject($"edge_{e.pre}_{e.post}");
                    go.transform.SetParent(raiz, false);
                    var lr = go.AddComponent<LineRenderer>();
                    lr.useWorldSpace = true;
                    lr.positionCount = 2;
                    lr.SetPosition(0, a);
                    lr.SetPosition(1, b);
                    lr.material = _matBase;
                    lr.numCapVertices = 2;
                    lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    lr.receiveShadows = false;

                    // espessura em raiz quadrada: o peso vai de 1 a ~740 e uma
                    // escala linear faria as fracas sumirem por completo
                    float t = Mathf.Sqrt(e.weight / (float)pesoMax);
                    float esp = Mathf.Lerp(espessuraMin, espessuraMax, t);
                    lr.startWidth = lr.endWidth = esp;
                    lr.enabled = false;

                    _arestas.Add(new Aresta
                    {
                        pre = e.pre, post = e.post, weight = e.weight,
                        sign = e.sign == 0 ? 1 : e.sign, linha = lr,
                    });
                }

            edgeCount = _arestas.Count;
            Debug.Log($"[graph] {edgeCount} arestas construidas (peso max {pesoMax})");
        }

        /// <summary>Marca as arestas que saem de quem acabou de disparar.</summary>
        public void OnSpikes(IEnumerable<long> bodyIdsQueDispararam)
        {
            if (!show || !pulsarNoSpike) return;
            var conj = new HashSet<long>(bodyIdsQueDispararam);
            foreach (var a in _arestas) if (conj.Contains(a.pre)) a.pulso = 1f;
        }

        void Update()
        {
            long sel = _brain != null ? _brain.SelectedBodyId : -1;
            int visiveis = 0;
            float dt = Time.deltaTime;

            foreach (var a in _arestas)
            {
                a.pulso = Mathf.Max(0f, a.pulso - dt * 3f);

                bool mostrar = show;
                if (mostrar)
                    switch (filtro)
                    {
                        case Filtro.Excitatorias: mostrar = a.sign > 0; break;
                        case Filtro.Inibitorias: mostrar = a.sign < 0; break;
                        case Filtro.Selecionado:
                            mostrar = sel >= 0 && (a.pre == sel || a.post == sel); break;
                    }
                // com um neuronio selecionado, so o que toca nele
                if (mostrar && filtro != Filtro.Selecionado && sel >= 0)
                    mostrar = a.pre == sel || a.post == sel;

                if (a.linha.enabled != mostrar) a.linha.enabled = mostrar;
                if (!mostrar) continue;
                visiveis++;

                var cor = a.sign < 0 ? CorInibitoria : CorExcitatoria;
                float alfa = 0.16f + 0.7f * a.pulso;
                var c = new Color(cor.r, cor.g, cor.b, alfa);
                a.linha.startColor = a.linha.endColor = c;
            }
            visibleEdges = visiveis;
        }

        /// <summary>Resumo do neuronio pro inspector: peso que entra e que sai.</summary>
        public (int entra, int sai, int nEntra, int nSai) Resumo(long bodyId)
        {
            int entra = 0, sai = 0, nE = 0, nS = 0;
            foreach (var a in _arestas)
            {
                if (a.post == bodyId) { entra += a.weight; nE++; }
                if (a.pre == bodyId) { sai += a.weight; nS++; }
            }
            return (entra, sai, nE, nS);
        }
    }
}
