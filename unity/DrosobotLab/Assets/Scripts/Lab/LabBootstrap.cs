// Monta a cena do Drosobot Lab por codigo.
//
// POR QUE POR CODIGO: este projeto foi escrito sem acesso ao Editor da Unity
// (nao havia integracao MCP disponivel), entao autorar .unity/.prefab -- que sao
// YAML com GUIDs -- seria fragil. Construir a cena aqui tem duas vantagens que
// valem por si: da pra revisar no diff do git, e nao existe estado escondido no
// arquivo de cena.
//
// Pra usar: cena vazia, um GameObject, este componente. Play.
//
// ARQUITETURA -- o que este projeto NAO faz:
//   * nao simula fisica: MuJoCo/FlyGym e a autoridade, a Unity espelha a pose
//   * nao simula neuronio: o circuito roda em Python sobre o conectoma
//   * nao decide nada: tudo aqui e apresentacao do que chegou pela telemetria
//
// Se a simulacao estiver 25x mais lenta que tempo real (e esta -- ver README),
// isto continua a 60 fps. Lentidao da simulacao nao e travamento da interface,
// e o HUD mostra SIM/WALL/RTF exatamente pra isso ficar obvio.

using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;
using Drosobot.Telemetry;

namespace Drosobot.Lab
{
    public class LabBootstrap : MonoBehaviour
    {
        [Header("Telemetria")]
        public string host = "127.0.0.1";
        public int port = 8765;

        [Header("CNS")]
        [Tooltip("Caminho dentro de Assets/Resources (sem extensao)")]
        public string cnsResource = "CNS/cns";
        public string metadataResource = "CNS/neuron_metadata";

        [Header("Apresentacao")]
        public bool presentationMode;
        [Tooltip("Casca do cerebro/VNC visivel. Desligada por padrao: ela e contexto " +
                 "anatomico e, opaca, esconde justamente os neuronios que sao o assunto.")]
        public bool showShell;

        private TelemetryClient _tel;
        private BrainActivity _brain;
        private Transform _cnsRoot;
        private Transform _fly;
        private Camera _camBrain;
        private readonly List<Renderer> _shellRenderers = new List<Renderer>();
        private ConnectivityGraph _grafo;

        // ultimo estado recebido, so pra desenhar
        private string _expName = "(nenhum experimento)";
        private string _expDescription = "";
        private double _simTime, _wallTime, _rtf;
        private int _step;
        private Vector3 _flyPos;
        private float[] _drive = { 0f, 0f };
        private readonly Dictionary<string, string> _provenance = new Dictionary<string, string>();
        private readonly List<string> _eventos = new List<string>();
        private readonly Dictionary<string, int> _spikesPorCamada = new Dictionary<string, int>();
        private float[] _retinaDerivadaL, _retinaDerivadaR;
        private string _retinaLabel = "";

        // paineis
        private readonly List<Sparkline> _graficos = new List<Sparkline>();
        private Sparkline _gSensorL, _gSensorR, _gEsquerda, _gDireita, _gSelecionado;
        private readonly RetinaView _retL = new RetinaView(), _retR = new RetinaView();
        private float[] _retinaL, _retinaR;
        private readonly Dictionary<string, string> _infoSelecionado = new Dictionary<string, string>();
        private long _ultimoSelecionado = -1;

        IEnumerator Start()
        {
            // Sem isto a Unity congela o Play mode quando a janela perde o foco.
            // Aqui isso e inaceitavel por dois motivos: quem usa o Lab fica
            // alternando pro terminal do Python pra ver a simulacao, e a
            // verificacao automatizada (unity cmd) nunca tem o foco -- o
            // Time.frameCount ficava em 2 e o Update praticamente nao rodava.
            Application.runInBackground = true;

            BuildScene();

            _gSensorL = new Sparkline("sensor L (Hz)", new Color(0.35f, 0.65f, 1f), Provenance.Assumption);
            _gSensorR = new Sparkline("sensor R (Hz)", new Color(1f, 0.6f, 0.25f), Provenance.Assumption);
            _gEsquerda = new Sparkline("spikes esq", new Color(0.4f, 0.8f, 1f), Provenance.Model);
            _gDireita = new Sparkline("spikes dir", new Color(1f, 0.45f, 0.4f), Provenance.Model);
            _gSelecionado = new Sparkline("neuronio sel.", new Color(0.8f, 0.9f, 0.4f), Provenance.Model);
            _graficos.AddRange(new[] { _gSensorL, _gSensorR, _gEsquerda, _gDireita, _gSelecionado });

            _tel = gameObject.AddComponent<TelemetryClient>();
            _tel.host = host;
            _tel.port = port;
            _tel.OnMessage += OnTelemetry;

            yield return StartCoroutine(LoadCns());
        }

        // ------------------------------------------------------------- cena

        private void BuildScene()
        {
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.17f, 0.19f, 0.23f);

            var camGo = new GameObject("BrainCamera");
            _camBrain = camGo.AddComponent<Camera>();
            _camBrain.clearFlags = CameraClearFlags.SolidColor;
            _camBrain.backgroundColor = new Color(0.055f, 0.06f, 0.075f);
            _camBrain.transform.position = new Vector3(0f, 2f, -18f);
            _camBrain.transform.LookAt(Vector3.zero);
            camGo.AddComponent<OrbitCamera>();

            var luzGo = new GameObject("KeyLight");
            var luz = luzGo.AddComponent<Light>();
            luz.type = LightType.Directional;
            luz.intensity = 0.55f;
            luz.color = new Color(0.85f, 0.88f, 1f);
            luzGo.transform.rotation = Quaternion.Euler(45f, 35f, 0f);

            // marcador provisorio da mosca. NAO e o modelo do FlyGym.
            // Trocar por ele exige exportar a malha do NeuroMechFly; ate la, um
            // marcador honestamente identificado e melhor que anatomia inventada.
            var flyGo = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            flyGo.name = "FlyMarker (placeholder -- nao e o modelo do FlyGym)";
            flyGo.transform.localScale = new Vector3(0.6f, 1.2f, 0.6f);
            Destroy(flyGo.GetComponent<Collider>());
            _fly = flyGo.transform;

            _brain = gameObject.AddComponent<BrainActivity>();
            _grafo = gameObject.AddComponent<ConnectivityGraph>();
        }

        private IEnumerator LoadCns()
        {
            // Import NATIVO da Unity a partir de Assets/Resources, em vez de
            // carregar GLB em runtime com glTFast.
            //
            // O motivo nao e preferencia: glTFast arrastava Burst, Collections,
            // Mathematics, Mono Cecil e o Performance API, e a Unity 6 sinalizou
            // os cinco com "invalid signature". Nenhum era necessario -- a Unity
            // importa FBX sozinha, sem pacote algum. Menos dependencia, menos
            // aviso de seguranca, e o asset vira um asset normal do projeto.
            var prefab = Resources.Load<GameObject>(cnsResource);
            var metaTxt = Resources.Load<TextAsset>(metadataResource);

            if (prefab == null || metaTxt == null)
            {
                Debug.LogError(
                    $"[lab] CNS nao encontrado em Resources/{cnsResource}.\n" +
                    "Gere com:\n" +
                    "\"C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe\" " +
                    "--background --python blender\\export_unity.py");
                yield break;
            }

            var inst = Instantiate(prefab);
            inst.name = "CNS";
            _cnsRoot = inst.transform;
            yield return null;   // deixa a hierarquia assentar antes de ligar

            var meta = NeuronMetadata.FromJson(metaTxt.text);
            _brain.Bind(_cnsRoot, meta);

            // coverage vem como dicionario no JSON, que o JsonUtility nao le --
            // por isso passa pelo Newtonsoft aqui
            try
            {
                var raiz = JObject.Parse(metaTxt.text);
                if (raiz["coverage"] is JObject cov)
                {
                    var lista = new List<CoverageEntry>();
                    foreach (var kv in cov)
                        lista.Add(kv.Value.ToObject<CoverageEntry>());
                    _brain.SetCoverage(lista.ToArray());
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[lab] coverage ilegivel: {e.Message}");
            }
            _grafo.Build(_cnsRoot, meta, _brain);

            // malhas de contexto ficam translucidas e discretas: sao referencia
            // anatomica, nao o assunto
            foreach (var nome in meta.contextMeshes)
            {
                var t = FindDeep(_cnsRoot, nome);
                if (t == null) continue;
                // material proprio tambem aqui: o do FBX nao aceita transparencia
                var sh = Shader.Find("Standard") ?? Shader.Find("Universal Render Pipeline/Lit");
                foreach (var r in t.GetComponentsInChildren<Renderer>(true))
                {
                    var m = new Material(sh) { color = new Color(0.62f, 0.66f, 0.74f, 0.085f) };
                    SetTransparent(m);
                    m.SetFloat("_Glossiness", 0f);
                    r.material = m;
                    r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    r.receiveShadows = false;
                    r.enabled = showShell;
                    _shellRenderers.Add(r);
                }
            }
            Debug.Log($"[lab] CNS carregado: {meta.neuronCount} neuronios, " +
                      $"contexto: {string.Join(", ", meta.contextMeshes)}");
        }

        private static Transform FindDeep(Transform raiz, string nome)
        {
            if (raiz.name == nome) return raiz;
            foreach (Transform c in raiz)
            {
                var achado = FindDeep(c, nome);
                if (achado != null) return achado;
            }
            return null;
        }

        private static void SetTransparent(Material m)
        {
            // Transparencia no Standard do Built-in nao sai so mudando _Mode:
            // precisa dos blend modes, de desligar o ZWrite E das keywords. Sem
            // as keywords a casca do cerebro fica preta e opaca, escondendo
            // justamente os neuronios que queremos ver dentro dela.
            m.SetFloat("_Mode", 3f);                 // Transparent
            m.SetFloat("_Surface", 1f);              // URP
            m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetInt("_ZWrite", 0);
            m.DisableKeyword("_ALPHATEST_ON");
            m.EnableKeyword("_ALPHABLEND_ON");
            m.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            m.SetFloat("_Glossiness", 0f);
            m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        }

        // -------------------------------------------------------- telemetria

        private void OnTelemetry(string linha)
        {
            JObject msg;
            try { msg = JObject.Parse(linha); }
            catch { return; }

            switch ((string)msg["type"])
            {
                case "experiment_info":
                    _expName = (string)msg["name"] ?? "?";
                    _expDescription = (string)msg["description"] ?? "";
                    _provenance.Clear();
                    if (msg["provenance"] is JObject p)
                        foreach (var kv in p) _provenance[kv.Key] = (string)kv.Value;
                    _brain.ApplyExperimentInfo(msg);
                    break;

                case "frame":
                    _step = (int)msg["step"];
                    _simTime = (double)msg["sim_time"];
                    _rtf = (double)msg["rtf"];
                    var pos = msg["position"] as JArray;
                    if (pos != null && pos.Count >= 3)
                    {
                        // MuJoCo e Z-up, Unity e Y-up: trocamos os eixos aqui.
                        // Isto e APRESENTACAO. A pose autoritativa continua sendo
                        // a do MuJoCo; nada volta pra la.
                        _flyPos = new Vector3((float)pos[0], (float)pos[2], (float)pos[1]);
                        if (_fly != null) _fly.position = _flyPos;
                    }
                    var d = msg["drive"] as JArray;
                    if (d != null && d.Count >= 2) _drive = new[] { (float)d[0], (float)d[1] };
                    break;

                case "neural_activity":
                    _brain.ApplyNeuralActivity(msg);
                    _spikesPorCamada.Clear();
                    if (msg["layers"] is JArray camadas)
                        foreach (var c in camadas)
                        {
                            int soma = 0;
                            if (c["spikes"] is JArray sp) foreach (var v in sp) soma += (int)v;
                            _spikesPorCamada[(string)c["name"] ?? "?"] = soma;
                        }
                    // esquerda/direita da ultima camada (a motora), por hemisferio
                    int esq = 0, dir = 0;
                    if (msg["layers"] is JArray ls && ls.Count > 0)
                    {
                        var ultima = ls[ls.Count - 1];
                        if (ultima["spikes"] is JArray sp2)
                            for (int i = 0; i < sp2.Count; i++)
                                { if (i % 2 == 0) dir += (int)sp2[i]; else esq += (int)sp2[i]; }
                    }
                    _gEsquerda.Push(esq);
                    _gDireita.Push(dir);
                    _gSelecionado.Push(_brain != null ? _brain.SelectedActivity() : 0f);
                    break;

                case "retina":
                    if (msg["left"] is JArray rl)
                    {
                        if (_retinaL == null || _retinaL.Length != rl.Count) _retinaL = new float[rl.Count];
                        for (int i = 0; i < rl.Count; i++) _retinaL[i] = (float)rl[i];
                    }
                    if (msg["right"] is JArray rr)
                    {
                        if (_retinaR == null || _retinaR.Length != rr.Count) _retinaR = new float[rr.Count];
                        for (int i = 0; i < rr.Count; i++) _retinaR[i] = (float)rr[i];
                    }
                    if (msg["derived"] is JObject der)
                    {
                        foreach (var kv in der)
                        {
                            if (!(kv.Value is JObject lr)) continue;
                            _retinaLabel = kv.Key;
                            _retinaDerivadaL = new[] { (float)lr["L"] };
                            _retinaDerivadaR = new[] { (float)lr["R"] };
                            _gSensorL.Push(_retinaDerivadaL[0]);
                            _gSensorR.Push(_retinaDerivadaR[0]);
                            break;
                        }
                    }
                    break;

                case "event":
                    _eventos.Insert(0, $"{(double)msg["sim_time"]:F2}s  {(string)msg["kind"]}");
                    if (_eventos.Count > 8) _eventos.RemoveAt(_eventos.Count - 1);
                    break;
            }
        }

        // --------------------------------------------------------------- HUD
        // IMGUI de proposito: sem Editor nao da pra autorar prefabs de UI de
        // forma confiavel. E funcional e legivel; trocar por uGUI/UI Toolkit e
        // polimento, nao arquitetura.

        private GUIStyle _titulo, _rotulo, _mono;

        void OnGUI()
        {
            if (_titulo == null)
            {
                _titulo = new GUIStyle(GUI.skin.label) { fontSize = 17, fontStyle = FontStyle.Bold };
                _titulo.normal.textColor = new Color(0.92f, 0.94f, 0.98f);
                _rotulo = new GUIStyle(GUI.skin.label) { fontSize = 12 };
                _rotulo.normal.textColor = new Color(0.72f, 0.76f, 0.82f);
                _mono = new GUIStyle(_rotulo) { fontSize = 12, alignment = TextAnchor.UpperLeft };
            }

            GUI.color = new Color(1, 1, 1, 0.93f);
            GUI.Box(new Rect(10, 10, 360, presentationMode ? 130 : 330), GUIContent.none);
            GUILayout.BeginArea(new Rect(22, 20, 340, presentationMode ? 115 : 315));

            GUILayout.Label("DROSOBOT LAB", _titulo);
            GUILayout.Label(_expName, _rotulo);
            GUILayout.Space(6);

            // SIM/WALL/RTF em destaque: a simulacao roda ~25x abaixo de tempo
            // real e isso NAO e travamento. Deixar explicito evita o mal-entendido.
            GUILayout.Label($"SIM   {_simTime:F2} s", _mono);
            GUILayout.Label($"WALL  {_wallTime:F1} s", _mono);
            GUILayout.Label($"RTF   {_rtf:F3}x   (lento de proposito)", _mono);
            GUILayout.Label($"conexao: {(_tel != null && _tel.connected ? "ligada" : "aguardando simulacao")}", _mono);

            if (!presentationMode)
            {
                GUILayout.Space(8);
                GUILayout.Label("ATIVIDADE (spikes na janela)", _rotulo);
                foreach (var kv in _spikesPorCamada)
                {
                    int total = 0, semGeo = 0;
                    _brain.TotalPorCamada.TryGetValue(kv.Key, out total);
                    _brain.SemGeometriaPorCamada.TryGetValue(kv.Key, out semGeo);
                    int comGeo = System.Math.Max(0, total - semGeo);
                    // Dizer so "LC4/LPLC2 67" daria a entender que sao todos os
                    // que aparecem no cerebro. Mostramos quantos sao simulados e
                    // quantos tem morfologia individual.
                    string sufixo = semGeo > 0
                        ? $"   {total} sim / {comGeo} com morfologia"
                        : "";
                    GUILayout.Label($"  {kv.Key,-10} {kv.Value,4}{sufixo}", _mono);
                }

                if (_retinaDerivadaL != null)
                {
                    GUILayout.Space(6);
                    GUILayout.Label($"VISAO ({_retinaLabel})", _rotulo);
                    GUILayout.Label($"  L {_retinaDerivadaL[0],7:F1}   R {_retinaDerivadaR[0],7:F1}", _mono);
                }

                GUILayout.Space(6);
                GUILayout.Label("EVENTOS", _rotulo);
                foreach (var e in _eventos) GUILayout.Label("  " + e, _mono);
                GUILayout.Space(6);
                GUILayout.Label("B casca  N neuronios  P apresentacao", _mono);
                GUILayout.Label("botao direito seleciona  ESC limpa", _mono);
                if (_grafo != null)
                {
                    string estado = _grafo.show
                        ? _grafo.visibleEdges + "/" + _grafo.edgeCount
                        : "off";
                    GUILayout.Label($"C conexoes ({estado})  F filtro: {_grafo.filtro}", _mono);
                }
            }

            GUILayout.EndArea();

            if (!presentationMode)
            {
                DesenhaLegendaProcedencia();
                DesenhaPopulacao();
                DesenhaTimeline();
                DesenhaRetina();
                DesenhaInspector();
            }
        }

        private void DesenhaLegendaProcedencia()
        {
            // A separacao DATA / MODEL / ASSUMPTION e uma das razoes de o projeto
            // existir. Fica sempre visivel, nao escondida num menu.
            var r = new Rect(Screen.width - 250, 10, 240, 92);
            GUI.Box(r, GUIContent.none);
            GUILayout.BeginArea(new Rect(r.x + 12, r.y + 8, r.width - 20, r.height - 14));
            GUILayout.Label("PROCEDENCIA", _rotulo);
            foreach (Provenance p in new[] { Provenance.Data, Provenance.Model, Provenance.Assumption })
            {
                var antes = GUI.color;
                GUI.color = ProvenanceUtil.Color(p);
                GUILayout.Label($"  {ProvenanceUtil.Label(p)}  {Explica(p)}", _mono);
                GUI.color = antes;
            }
            GUILayout.EndArea();
        }

        private void DesenhaTimeline()
        {
            float alturaG = 46f, margem = 8f;
            float h = _graficos.Count * (alturaG + margem) + 30f;
            var r = new Rect(10, Screen.height - h - 10, 262, h);
            GUI.Box(r, GUIContent.none);
            GUILayout.BeginArea(new Rect(r.x + 10, r.y + 6, r.width - 18, r.height - 10));
            GUILayout.Label("SINAIS (janela rolante)", _rotulo);
            foreach (var g in _graficos)
            {
                var antes = GUI.color;
                GUI.color = ProvenanceUtil.Color(g.procedencia);
                GUILayout.Label(g.Rotulo, _mono);
                GUI.color = antes;
                var tex = g.Desenhar();
                var rg = GUILayoutUtility.GetRect(240, alturaG - 16);
                GUI.DrawTexture(rg, tex, ScaleMode.StretchToFill);
            }
            GUILayout.EndArea();
        }

        private void DesenhaRetina()
        {
            if (_retinaL == null && _retinaR == null) return;
            float lw = _retL.Largura, lh = _retL.Altura;
            var r = new Rect(Screen.width - 250, 112, 240, lh * 2 + 76);
            GUI.Box(r, GUIContent.none);
            GUILayout.BeginArea(new Rect(r.x + 10, r.y + 6, r.width - 18, r.height - 10));
            GUILayout.Label("VISAO  (721 omatideos/olho)", _rotulo);
            GUILayout.Label("LEFT EYE", _mono);
            GUI.DrawTexture(GUILayoutUtility.GetRect(lw, lh), _retL.Desenhar(_retinaL),
                            ScaleMode.StretchToFill);
            GUILayout.Label("RIGHT EYE", _mono);
            GUI.DrawTexture(GUILayoutUtility.GetRect(lw, lh), _retR.Desenhar(_retinaR),
                            ScaleMode.StretchToFill);
            GUILayout.EndArea();
        }

        private void DesenhaPopulacao()
        {
            if (_brain == null || _brain.AtividadeAgregada.Count == 0) return;
            var linhas = new List<string>();
            foreach (var kv in _brain.AtividadeAgregada)
            {
                if (!_brain.SemGeometriaPorCamada.TryGetValue(kv.Key, out int semGeo) || semGeo <= 0)
                    continue;
                linhas.Add(kv.Key + "|" + kv.Value.ToString("F0") + "|" + semGeo);
            }
            if (linhas.Count == 0) return;

            var r = new Rect(290, 10, 330, 34 + linhas.Count * 40);
            GUI.Box(r, GUIContent.none);
            GUILayout.BeginArea(new Rect(r.x + 12, r.y + 8, r.width - 20, r.height - 14));
            GUILayout.Label("ATIVIDADE DE POPULACAO", _rotulo);
            foreach (var linha in linhas)
            {
                var partes = linha.Split('|');
                float v = float.Parse(partes[1]);
                // Barra de populacao: os neuronios sem morfologia exportada
                // contribuem aqui, e NAO num ponto inventado do cerebro.
                int blocos = Mathf.Clamp(Mathf.RoundToInt(v / 3f), 0, 28);
                GUILayout.Label($"{partes[0]}  ({partes[2]} sem morfologia individual)", _mono);
                GUILayout.Label("  " + new string('#', blocos), _mono);
            }
            GUILayout.EndArea();
        }

        private void DesenhaInspector()
        {
            long sel = _brain != null ? _brain.SelectedBodyId : -1;
            var r = new Rect(Screen.width - 340, Screen.height - 232, 330, 222);
            GUI.Box(r, GUIContent.none);
            GUILayout.BeginArea(new Rect(r.x + 12, r.y + 8, r.width - 20, r.height - 14));
            GUILayout.Label("INSPECTOR", _rotulo);
            if (sel < 0)
            {
                // Sem selecao, o espaco vira o aviso de amostragem. Afirmar ou
                // sugerir que a morfologia mostrada e a populacao toda seria
                // falso, e este e o lugar onde o usuario olharia.
                foreach (var c in _brain.Coverage)
                {
                    GUILayout.Label(c.group, _mono);
                    GUILayout.Label($"  Simulated: {c.total_simulated} neurons", _mono);
                    GUILayout.Label($"  3D morphology shown: {c.total_visualized} representative", _mono);
                    GUILayout.Label($"  Synaptic-weight coverage: {c.fraction_weight_covered * 100f:F0}%", _mono);
                    var antes = GUI.color;
                    GUI.color = ProvenanceUtil.Color(Provenance.Data);
                    GUILayout.Label("  DATA - morphology subset", _mono);
                    GUI.color = antes;
                }
                GUILayout.Label("botao direito seleciona um neuronio", _mono);
                GUILayout.EndArea();
                return;
            }
            if (sel != _ultimoSelecionado)
            {
                _ultimoSelecionado = sel;
                var d = _brain.Describe(sel);
                _infoSelecionado.Clear();
                if (d != null) foreach (var kv in d) _infoSelecionado[kv.Key] = kv.Value;
            }
            // cada campo com sua procedencia: e o ponto do projeto inteiro
            Badge.Linha("bodyId", Get("bodyId"), Provenance.Data, _mono);
            Badge.Linha("type", Get("type"), Provenance.Data, _mono);
            Badge.Linha("group", Get("group"), Provenance.Data, _mono);
            Badge.Linha("side", Get("side"), Provenance.Data, _mono);
            Badge.Linha("neurotransmitter", Get("neurotransmitter"), Provenance.Data, _mono);
            Badge.Linha("polarity", Get("polarity"), Provenance.Model, _mono);
            Badge.Linha("atividade", Get("recentActivity"), Provenance.Model, _mono);
            if (_grafo != null)
            {
                var (entra, sai, nE, nS) = _grafo.Resumo(sel);
                Badge.Linha("sinapses in", $"{entra} ({nE} parceiros)", Provenance.Data, _mono);
                Badge.Linha("sinapses out", $"{sai} ({nS} parceiros)", Provenance.Data, _mono);
            }
            GUILayout.EndArea();
        }

        private string Get(string k) => _infoSelecionado.TryGetValue(k, out var v) ? v : "-";

        private static string Explica(Provenance p)
        {
            switch (p)
            {
                case Provenance.Data: return "conectoma medido";
                case Provenance.Model: return "modelo de Shiu et al.";
                case Provenance.Assumption: return "suposicao nossa";
                default: return "";
            }
        }

        void Update()
        {
            _wallTime = Time.realtimeSinceStartup;
            if (Input.GetMouseButtonDown(1) && _camBrain != null && _brain != null)
            {
                // botao direito seleciona; o esquerdo ja gira a camera
                long achou = _brain.Pick(_camBrain.ScreenPointToRay(Input.mousePosition));
                if (achou >= 0) Debug.Log($"[lab] selecionado neuron_{achou}");
            }
            if (Input.GetKeyDown(KeyCode.Escape) && _brain != null) _brain.ClearSelection();
            if (Input.GetKeyDown(KeyCode.C) && _grafo != null) _grafo.show = !_grafo.show;
            if (Input.GetKeyDown(KeyCode.F) && _grafo != null)
                _grafo.filtro = (ConnectivityGraph.Filtro)
                    (((int)_grafo.filtro + 1) % System.Enum.GetValues(typeof(ConnectivityGraph.Filtro)).Length);
            if (Input.GetKeyDown(KeyCode.P)) presentationMode = !presentationMode;
            if (Input.GetKeyDown(KeyCode.N)) _brain.showNeurons = !_brain.showNeurons;
            if (Input.GetKeyDown(KeyCode.B))
            {
                showShell = !showShell;
                foreach (var r in _shellRenderers) if (r != null) r.enabled = showShell;
            }
        }
    }

    /// <summary>Camera de orbita: arrastar gira, scroll aproxima.</summary>
    public class OrbitCamera : MonoBehaviour
    {
        public Vector3 target = Vector3.zero;
        public float distance = 18f;
        public float yaw = 0f, pitch = 12f;

        void LateUpdate()
        {
            if (Input.GetMouseButton(0))
            {
                yaw += Input.GetAxis("Mouse X") * 3f;
                pitch = Mathf.Clamp(pitch - Input.GetAxis("Mouse Y") * 2f, -85f, 85f);
            }
            distance = Mathf.Clamp(distance - Input.mouseScrollDelta.y * 1.5f, 2f, 120f);
            var rot = Quaternion.Euler(pitch, yaw, 0f);
            transform.position = target + rot * new Vector3(0, 0, -distance);
            transform.rotation = rot;
        }
    }
}
