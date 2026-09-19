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

        IEnumerator Start()
        {
            BuildScene();

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
                    break;

                case "retina":
                    if (msg["derived"] is JObject der)
                    {
                        foreach (var kv in der)
                        {
                            if (!(kv.Value is JObject lr)) continue;
                            _retinaLabel = kv.Key;
                            _retinaDerivadaL = new[] { (float)lr["L"] };
                            _retinaDerivadaR = new[] { (float)lr["R"] };
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
                    GUILayout.Label($"  {kv.Key,-10} {kv.Value,4}", _mono);

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
            }

            GUILayout.EndArea();

            if (!presentationMode) DesenhaLegendaProcedencia();
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
