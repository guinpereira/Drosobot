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
        [Tooltip("Canal de controle: e por onde a interface escolhe o experimento. " +
                 "Socket separado do de telemetria de proposito -- ver ControlClient.cs.")]
        public int controlPort = 8766;

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
        private ControlClient _ctl;
        private ExperimentSelector _seletor;
        private BrainActivity _brain;
        private Transform _cnsRoot;
        private Transform _fly;
        private Camera _camBrain;
        private readonly List<Renderer> _shellRenderers = new List<Renderer>();
        private ConnectivityGraph _grafo;
        private FlyBody _mosca;

        // O cerebro ganhou camera propria. Antes CNS e mosca dividiam a mesma
        // camera; quando ela passou a seguir a mosca (que anda), o CNS saia de
        // quadro e sumia. Agora cada um tem a sua, e o cerebro e desenhado numa
        // caixa no painel lateral -- da pra ver o corpo andando e as sinapses
        // pulsando ao mesmo tempo.
        private Camera _camCns;
        private RenderTexture _rtCns;
        private const int CAMADA_CNS = 30;      // layer dedicada ao CNS
        public int larguraCaixaCns = 300;
        public int alturaCaixaCns = 300;
        private float _cnsYaw = 0f;
        [Tooltip("Graus por segundo que a caixa do cerebro gira sozinha. " +
                 "0 = parada.")]
        public float giroCns = 8f;

        // gate do Giant Fiber: dado REAL vindo do runtime, nao animacao
        private float _gfExc, _gfInib, _gfLiquido, _gfVmin;
        private int _gfSpikes;
        private readonly List<string> _gfTopInib = new List<string>();
        private string _limitacaoTexto = "";
        private float _limitacaoLimiar = 0f;
        private bool _limitacaoDisparou;

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

        // runtime: quem produziu o dado que esta na tela. Vem do campo
        // `runtime` do experiment_info, que o sim/drosobot_lab.py publica.
        // Nunca inferido aqui -- se o runtime nao disser, fica em branco.
        private readonly Dictionary<string, string> _runtime = new Dictionary<string, string>();
        private string _escopoSimulado = "", _escopoSensorimotor = "", _escopoNota = "";
        // custo por etapa, em ms de relogio por SEGUNDO SIMULADO. E a unica base
        // em que as etapas se somam -- ver sim/profiler.py.
        private readonly Dictionary<string, float> _profile = new Dictionary<string, float>();
        private readonly Dictionary<string, int> _popAtividade = new Dictionary<string, int>();

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

            _ctl = gameObject.AddComponent<ControlClient>();
            _ctl.host = host;
            _ctl.port = controlPort;
            _seletor = gameObject.AddComponent<ExperimentSelector>();
            _seletor.Ligar(_ctl);

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
            _camBrain.transform.position = new Vector3(0f, 2f, -15.5f);
            _camBrain.transform.LookAt(Vector3.zero);
            camGo.AddComponent<OrbitCamera>();

            // camera do cerebro: renderiza SO a layer do CNS, numa textura
            var cnsGo = new GameObject("CnsCamera");
            _camCns = cnsGo.AddComponent<Camera>();
            _camCns.clearFlags = CameraClearFlags.SolidColor;
            _camCns.backgroundColor = new Color(0.04f, 0.045f, 0.06f);
            _camCns.cullingMask = 1 << CAMADA_CNS;
            _rtCns = new RenderTexture(larguraCaixaCns, alturaCaixaCns, 16);
            _camCns.targetTexture = _rtCns;
            // e a camera principal deixa de desenhar o CNS, senao ele aparece
            // duas vezes -- uma na caixa e outra por cima da arena
            _camBrain.cullingMask &= ~(1 << CAMADA_CNS);

            // Tres luzes: a mosca tem 3 mm, e escura e fica num fundo escuro.
            // Com uma luz so ela virava silhueta. Ver FlyAppearance.Ilumina.
            var luzGo = new GameObject("Luzes");
            FlyAppearance.Ilumina(luzGo.transform);

            // A mosca de verdade: malhas do NeuroMechFly exportadas do modelo
            // COMPILADO que a fisica roda (tools/export_fly_mesh.py). Se elas
            // nao estiverem geradas, cai num marcador -- mas rotulado como tal,
            // porque anatomia inventada e pior que marcador honesto.
            _mosca = gameObject.AddComponent<FlyBody>();
            if (_mosca.Montar())
            {
                _fly = _mosca.Raiz;
                // a camera segue o TORAX, nao a raiz: a raiz fica na origem e
                // quem se move sao os segmentos
                var torax = _mosca.Raiz.Find("c_thorax");
                var orb = _camBrain.GetComponent<OrbitCamera>();
                if (torax != null && orb != null)
                {
                    orb.seguir = torax;
                    orb.distance = 6f;
                    orb.pitch = 20f;
                    orb.yaw = 35f;
                }
            }
            else
            {
                var flyGo = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                flyGo.name = "FlyMarker (placeholder -- rode tools/export_fly_mesh.py)";
                flyGo.transform.localScale = new Vector3(0.6f, 1.2f, 0.6f);
                Destroy(flyGo.GetComponent<Collider>());
                _fly = flyGo.transform;
            }

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
            // tudo do CNS (malhas e arestas) vai pra layer dedicada
            foreach (var t in _cnsRoot.GetComponentsInChildren<Transform>(true))
                t.gameObject.layer = CAMADA_CNS;

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

            // Idade da ultima mensagem: `connected` continua true com o socket
            // aberto mesmo se a simulacao travar ou terminar. Sem a idade nao
            // da pra distinguir "rodando" de "parou e o socket ficou".
            _ultimaMensagem = Time.realtimeSinceStartup;
            if (_seletor != null) _seletor.AoReceber(msg);

            switch ((string)msg["type"])
            {
                case "experiment_info":
                    // limitacao declarada do modelo. Fica guardada e so aparece
                    // quando o runtime avisar que o limiar foi cruzado -- nao
                    // como alerta permanente.
                    if (msg["model_limitations"] is JArray ml && ml.Count > 0)
                    {
                        _limitacaoTexto = (string)ml[0]["text"] ?? "";
                        _limitacaoLimiar = (float?)ml[0]["warning_threshold_mV"] ?? 0f;
                    }
                    if (msg["runtime"] is JObject rt)
                    {
                        _runtime.Clear();
                        foreach (var kv in rt)
                            _runtime[kv.Key] = kv.Value?.ToString() ?? "";
                    }
                    if (msg["scope"] is JObject esc)
                    {
                        _escopoSimulado = (string)esc["simulated"] ?? "";
                        _escopoSensorimotor = (string)esc["sensorimotor_model"] ?? "";
                        _escopoNota = (string)esc["note"] ?? "";
                    }
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
                        // Conversao de eixo: `MujocoFrame`, como todo o resto.
                        // Isto e APRESENTACAO. A pose autoritativa continua sendo
                        // a do MuJoCo; nada volta pra la.
                        _flyPos = MujocoFrame.Pos((float)pos[0], (float)pos[1],
                                                  (float)pos[2]);
                        // So o MARCADOR e movido por aqui. Quando a mosca real
                        // esta montada, cada segmento ja chega com pose de
                        // MUNDO em `body_pose`; mover a raiz tambem somaria o
                        // deslocamento duas vezes e jogaria a mosca pra longe.
                        bool temMoscaReal = _mosca != null && _mosca.segmentosMontados > 0;
                        if (_fly != null && !temMoscaReal) _fly.position = _flyPos;
                    }
                    var d = msg["drive"] as JArray;
                    if (d != null && d.Count >= 2) _drive = new[] { (float)d[0], (float)d[1] };
                    if (msg["profile"] is JObject pf)
                    {
                        _profile.Clear();
                        foreach (var kv in pf) _profile[kv.Key] = (float)kv.Value;
                    }
                    // pose dos segmentos: chega a 30 Hz, nao por passo de fisica
                    if (msg["body_pose"] is JObject bp && _mosca != null)
                        _mosca.AplicarPose(bp);
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
                            if (lr["L"] == null || lr["R"] == null) continue;
                            _retinaLabel = kv.Key;
                            _retinaDerivadaL = new[] { (float)lr["L"] };
                            _retinaDerivadaR = new[] { (float)lr["R"] };
                            _gSensorL.Push(_retinaDerivadaL[0]);
                            _gSensorR.Push(_retinaDerivadaR[0]);
                            break;
                        }
                    }
                    break;

                case "statistics":
                    // atividade agregada das populacoes SEM morfologia 3D. O
                    // runtime nunca manda 164 mil estados por quadro.
                    if (msg["values"]?["population_activity"] is JObject pa)
                    {
                        _popAtividade.Clear();
                        foreach (var kv in pa) _popAtividade[kv.Key] = (int)kv.Value;
                    }
                    if (msg["values"]?["gf_gate"] is JObject gg)
                    {
                        _gfExc = (float)gg["exc_mV"];
                        _gfInib = (float)gg["inib_mV"];
                        _gfLiquido = (float)gg["liquido_mV"];
                        _gfVmin = (float)gg["v_min_mV"];
                        _gfSpikes = (int)gg["spikes_gf"];
                        _gfTopInib.Clear();
                        if (gg["top_inib"] is JArray ti)
                            foreach (var e in ti)
                                _gfTopInib.Add($"{(long)e["body_id"]}  {(float)e["mV"]:F1} mV");
                    }
                    break;

                case "event":
                    if ((string)msg["kind"] == "model_limitation")
                        _limitacaoDisparou = true;
                    _eventos.Insert(0, $"{(double)msg["sim_time"]:F2}s  {(string)msg["kind"]}");
                    // so os mais recentes: a lista alimenta um painel de altura
                    // medida, entao cada evento a mais empurra o resto pra baixo
                    if (_eventos.Count > 5) _eventos.RemoveAt(_eventos.Count - 1);
                    break;
            }
        }

        // --------------------------------------------------------------- HUD
        // IMGUI de proposito: sem Editor nao da pra autorar prefabs de UI de
        // forma confiavel. E funcional e legivel; trocar por uGUI/UI Toolkit e
        // polimento, nao arquitetura.

        private GUIStyle _titulo, _rotulo, _mono;

        // Colunas do HUD. A posicao de cada painel sai daqui, nao de coordenada
        // escrita a mao -- ver LabLayout.cs pro motivo (duas caixas chegaram a se
        // sobrepor porque um x fixo nao acompanhou a largura do vizinho).
        // Tres colunas, nao quatro. A do meio saiu de proposito: o CNS pode ser
        // girado e ampliado com o mouse, e painel no centro da tela disputa
        // espaco justamente com o objeto que o usuario esta manipulando.
        //
        //    esquerda   estado e sinais
        //    centro     o cerebro, sem nada por cima
        //    direita    inspecao e controles
        private Coluna _colEsquerda, _colDireita, _colRodape;
        private readonly Painel _pPrincipal = new Painel("principal");
        private readonly Painel _pExperimento = new Painel("experimento");
        private readonly Painel _pPopulacao = new Painel("populacao");
        private readonly Painel _pProcedencia = new Painel("procedencia");
        private readonly Painel _pRetina = new Painel("retina");
        private readonly Painel _pInspector = new Painel("inspector");
        private readonly Painel _pSinais = new Painel("sinais");
        private readonly Painel _pRuntime = new Painel("runtime");
        private readonly Painel _pGate = new Painel("gate");
        private readonly Painel _pCns3D = new Painel("cns3d");
        private readonly Painel _pCorpo = new Painel("corpo");
        private float _ultimaMensagem = -1f;

        void OnGUI()
        {
            if (_titulo == null)
            {
                _titulo = new GUIStyle(GUI.skin.label) { fontSize = 17, fontStyle = FontStyle.Bold };
                _titulo.normal.textColor = new Color(0.92f, 0.94f, 0.98f);
                _rotulo = new GUIStyle(GUI.skin.label) { fontSize = 12 };
                _rotulo.normal.textColor = new Color(0.72f, 0.76f, 0.82f);
                // wordWrap explicito: e o que faz uma linha longa virar duas em
                // vez de sumir na borda, e o Painel mede a altura ja com a quebra.
                _mono = new GUIStyle(_rotulo) { fontSize = 12, alignment = TextAnchor.UpperLeft,
                                                wordWrap = true };
                _rotulo.wordWrap = true;
                _titulo.wordWrap = true;
            }

            // Guarda propria, e nao a do bloco acima: os estilos sobrevivem ao
            // recarregamento de dominio da Unity e as colunas, sendo campos
            // novos, voltam nulas. Compartilhar o `if` deixava OnGUI lancando
            // NullReference depois de todo hot reload.
            if (_colEsquerda == null)
            {
                // Larguras por coluna. A da direita e a maior porque e onde mora
                // o texto corrido (procedencia, inspector), que era o que estava
                // sendo cortado.
                _colEsquerda = new Coluna(Coluna.Ancora.SuperiorEsquerda, 352f);
                // A direita e mais larga porque passou a acumular procedencia,
                // retina, inspector E os controles. Quando nao couber na altura,
                // a Coluna rola sozinha em vez de comprimir os paineis.
                _colDireita = new Coluna(Coluna.Ancora.SuperiorDireita, 384f);
                _colRodape = new Coluna(Coluna.Ancora.InferiorEsquerda, 300f);
            }

            GUI.color = new Color(1, 1, 1, 0.93f);

            MontaPrincipal();
            _colEsquerda.Limpar();
            _colEsquerda.Adiciona(_pPrincipal);

            _colDireita.Limpar();
            _colRodape.Limpar();
            if (!presentationMode)
            {
                // atividade de populacao e sinal, entao mora com os outros
                // sinais, na esquerda
                MontaPopulacao();
                _colEsquerda.Adiciona(_pPopulacao);

                MontaProcedencia();
                MontaRuntime();
                MontaRetina();
                MontaInspector();
                MontaExperimento();
                MontaSinais();
                // Ordem pedida: da leitura passiva pra acao. O EXPERIMENTO fica
                // logo abaixo do INSPECTOR pra dar pra olhar o neuronio
                // selecionado e trocar de corrida sem atravessar a tela.
                MontaGate();
                MontaCns3D();
                MontaCorpo();
                _colDireita.Adiciona(_pCns3D);
                // CORPO logo abaixo do cerebro: sao os dois inspetores do que
                // esta sendo simulado, um de cada lado da fronteira.
                _colDireita.Adiciona(_pCorpo);
                _colDireita.Adiciona(_pProcedencia);
                _colDireita.Adiciona(_pRuntime);
                _colDireita.Adiciona(_pGate);
                _colDireita.Adiciona(_pRetina);
                _colDireita.Adiciona(_pInspector);
                _colDireita.Adiciona(_pExperimento);
                _colRodape.Adiciona(_pSinais);
            }

            // rotulos vao ANTES dos paineis: sao da cena, e os paineis
            // tem que ficar por cima deles
            DesenhaRotulosCorpo();

            float usadoEsquerda = _colEsquerda.Desenhar();
            _colDireita.Desenhar();
            // o rodape so ocupa o que sobra abaixo da coluna de cima
            _colRodape.Desenhar(0f, usadoEsquerda + Espaco.Vao);
        }

        // ------------------------------------------------------------ paineis

        private void MontaPrincipal()
        {
            _pPrincipal.Limpar();
            _pPrincipal.Texto("DROSOBOT LAB", _titulo);
            _pPrincipal.Texto(_expName, _rotulo);
            _pPrincipal.Espacador(6f);

            // SIM/WALL/RTF em destaque: a simulacao roda ~25x abaixo de tempo
            // real e isso NAO e travamento. Deixar explicito evita o mal-entendido.
            _pPrincipal.Texto($"SIM   {_simTime:F2} s", _mono);
            _pPrincipal.Texto($"WALL  {_wallTime:F1} s", _mono);
            _pPrincipal.Texto($"RTF   {_rtf:F3}x   (lento de proposito)", _mono);
            _pPrincipal.Texto($"conexao: {(_tel != null && _tel.connected ? "ligada" : "aguardando simulacao")}", _mono);
            if (presentationMode) return;

            _pPrincipal.Espacador(8f);
            _pPrincipal.Texto("ATIVIDADE (spikes na janela)", _rotulo);
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
                _pPrincipal.Texto($"  {kv.Key,-10} {kv.Value,4}{sufixo}", _mono);
            }

            // Os dois arrays sao indexados aqui, entao os dois precisam ser
            // checados. O guard antigo so olhava L e estourava quando chegava um
            // `derived` sem o par -- ou quando o runtime nao publica retina,
            // como o laco novo, que manda looming_hz em `statistics`.
            if (_retinaDerivadaL != null && _retinaDerivadaL.Length > 0
                && _retinaDerivadaR != null && _retinaDerivadaR.Length > 0)
            {
                _pPrincipal.Espacador(6f);
                _pPrincipal.Texto($"VISAO ({_retinaLabel})", _rotulo);
                _pPrincipal.Texto($"  L {_retinaDerivadaL[0],7:F1}   R {_retinaDerivadaR[0],7:F1}", _mono);
            }

            _pPrincipal.Espacador(6f);
            _pPrincipal.Texto("EVENTOS", _rotulo);
            foreach (var e in _eventos) _pPrincipal.Texto("  " + e, _mono);

            _pPrincipal.Espacador(6f);
            _pPrincipal.Texto("B casca  N neuronios  P apresentacao", _mono);
            _pPrincipal.Texto("botao direito seleciona  ESC limpa", _mono);
            if (_grafo != null)
            {
                string estado = _grafo.show
                    ? _grafo.visibleEdges + "/" + _grafo.edgeCount
                    : "off";
                _pPrincipal.Texto($"C conexoes ({estado})  F filtro: {_grafo.filtro}", _mono);
            }
        }

        private void MontaExperimento()
        {
            _pExperimento.Limpar();
            if (_seletor != null) _seletor.Preenche(_pExperimento, _rotulo, _mono);
        }

        private void MontaProcedencia()
        {
            // A separacao DATA / MODEL / ASSUMPTION e uma das razoes de o projeto
            // existir. Fica sempre visivel, nao escondida num menu.
            _pProcedencia.Limpar();
            _pProcedencia.Texto("PROCEDENCIA", _rotulo);
            foreach (Provenance p in new[] { Provenance.Data, Provenance.Model, Provenance.Assumption })
                _pProcedencia.Texto($"  {ProvenanceUtil.Label(p)}  {Explica(p)}", _mono,
                                    ProvenanceUtil.Color(p));
        }

        // ------------------------------------------------------------- corpo
        //
        // Painel de INSPECAO do corpo: nao mexe na simulacao, so em como a
        // mosca e desenhada e de onde a camera olha.

        private static readonly string[] RotulosModo =
            { "Normal", "Bind Pose", "Eixos", "Rotulos" };
        private static readonly string[] RotulosVista =
            { "Persp", "Lado", "Topo", "Frente" };

        private void MontaCorpo()
        {
            _pCorpo.Limpar();
            _pCorpo.Texto("CORPO", _rotulo);

            if (_mosca == null || _mosca.segmentosMontados == 0)
            {
                _pCorpo.Texto("malhas nao geradas -- marcador no lugar", _mono,
                              ProvenanceUtil.Color(Provenance.Assumption));
                _pCorpo.Texto(".venv-flygym2\\Scripts\\python tools\\export_fly_mesh.py",
                              _mono);
                return;
            }

            // --- estado da conexao ---
            // Quatro estados de verdade, nao "ligada/desligada": o socket pode
            // estar aberto com a simulacao parada, e isso parecia "conectado".
            string estado, detalhe;
            // _wallTime e double (vem do relogio da simulacao); a idade so
            // precisa de precisao de milissegundo
            float idade = _ultimaMensagem < 0f ? -1f
                                               : (float)_wallTime - _ultimaMensagem;
            if (_tel == null || !_tel.connected)
            {
                estado = "DESCONECTADO";
                detalhe = _ultimaMensagem < 0f
                    ? "suba: python sim/lab_runner.py"
                    : $"a simulacao caiu ou terminou ha {idade:F0} s";
            }
            else if (_ultimaMensagem < 0f)
            {
                estado = "CONECTANDO"; detalhe = "socket aberto, nada chegou ainda";
            }
            else if (idade > 2f)
            {
                estado = "PARADO";
                detalhe = $"ultima telemetria ha {idade:F1} s";
            }
            else
            {
                estado = "RODANDO";
                detalhe = $"ultima telemetria ha {idade * 1000f:F0} ms";
            }
            var corEstado = estado == "RODANDO"
                ? new Color(0.45f, 0.85f, 0.55f)
                : estado == "CONECTANDO" ? new Color(0.90f, 0.80f, 0.40f)
                                         : new Color(0.90f, 0.45f, 0.45f);
            _pCorpo.Texto($"  {estado}", _mono, corEstado);
            _pCorpo.Texto($"  {detalhe}", _mono);
            _pCorpo.Espacador();

            // --- modo de depuracao ---
            _pCorpo.Desenho(24f, r =>
            {
                float w = (r.width - 6f) / 4f;
                for (int i = 0; i < 4; i++)
                {
                    var b = new Rect(r.x + i * (w + 2f), r.y, w, 22f);
                    var antes = GUI.color;
                    GUI.color = (int)_mosca.modo == i
                        ? new Color(0.45f, 0.80f, 1.00f)
                        : new Color(0.78f, 0.80f, 0.84f);
                    if (GUI.Button(b, RotulosModo[i]))
                    {
                        _mosca.modo = (FlyBody.Modo)i;
                        // Bind Pose e pra olhar o corpo inteiro parado: a
                        // camera para de perseguir e enquadra tudo, senao o
                        // modo "congelado" continua com a camera andando.
                        var o = _camBrain.GetComponent<OrbitCamera>();
                        if (o != null)
                        {
                            o.seguir = _mosca.modo == FlyBody.Modo.BindPose
                                ? null : _mosca.Raiz.Find("c_thorax");
                            o.Focar(_mosca.Extensao());
                        }
                    }
                    GUI.color = antes;
                }
            });

            if (_mosca.modo == FlyBody.Modo.BindPose)
                _pCorpo.Texto("pose de repouso do MODELO (juntas em zero). " +
                              "Nao e a postura de andar nem telemetria.", _mono,
                              ProvenanceUtil.Color(Provenance.Model));

            // --- vistas ---
            _pCorpo.Espacador();
            var orb = _camBrain != null ? _camBrain.GetComponent<OrbitCamera>() : null;
            _pCorpo.Desenho(24f, r =>
            {
                float w = (r.width - 6f) / 4f;
                for (int i = 0; i < 4; i++)
                {
                    var b = new Rect(r.x + i * (w + 2f), r.y, w, 22f);
                    var antes = GUI.color;
                    GUI.color = orb != null && (int)orb.preset == i
                        ? new Color(0.45f, 0.80f, 1.00f)
                        : new Color(0.78f, 0.80f, 0.84f);
                    if (GUI.Button(b, RotulosVista[i]) && orb != null)
                        orb.Aponta((OrbitCamera.Preset)i);
                    GUI.color = antes;
                }
            });
            _pCorpo.Desenho(24f, r =>
            {
                if (GUI.Button(new Rect(r.x, r.y, r.width, 22f), "Focar mosca")
                    && orb != null)
                    orb.Focar(_mosca.Extensao());
            });
            _pCorpo.Texto("arrastar gira   scroll aproxima   meio empurra", _mono);

            // --- aparencia ---
            // O modelo nao traz cor: os 69 geoms sao cinza 0,5. O esquema
            // realista e invencao nossa e tem que aparecer como tal, senao uma
            // captura de tela vira "foto" de uma coisa que ninguem mediu.
            _pCorpo.Espacador();
            _pCorpo.Desenho(24f, r =>
            {
                float w = (r.width - 2f) / 2f;
                for (int i = 0; i < 2; i++)
                {
                    var b = new Rect(r.x + i * (w + 2f), r.y, w, 22f);
                    var antes = GUI.color;
                    GUI.color = (int)_mosca.aparencia == i
                        ? new Color(0.45f, 0.80f, 1.00f)
                        : new Color(0.78f, 0.80f, 0.84f);
                    if (GUI.Button(b, i == 0 ? "Clay" : "Realista"))
                        _mosca.aparencia = (Aparencia)i;
                    GUI.color = antes;
                }
            });
            _pCorpo.Texto(_mosca.aparencia == Aparencia.Clay
                    ? "  cinza 0,5 -- a unica cor que o modelo carrega"
                    : "  paleta do flybody; atribuicao por anatomia e nossa",
                _mono,
                ProvenanceUtil.Color(_mosca.aparencia == Aparencia.Clay
                                     ? Provenance.Data : Provenance.Assumption));
            _pCorpo.Texto($"{_mosca.segmentosMontados} segmentos   " +
                          $"{_mosca.segmentosRecebidos} na pose", _mono);
        }

        // Rotulos dos segmentos: desenhados por cima da cena, projetados pela
        // camera. Mais simples que TextMesh em mundo e legivel em qualquer zoom.
        private void DesenhaRotulosCorpo()
        {
            if (_mosca == null || _mosca.modo != FlyBody.Modo.Rotulos) return;
            if (_camBrain == null) return;
            var antes = GUI.color;
            GUI.color = new Color(0.85f, 0.92f, 1f, 0.9f);
            foreach (var kv in _mosca.Segmentos())
            {
                var s = _camBrain.WorldToScreenPoint(kv.Value);
                if (s.z <= 0f) continue;                  // atras da camera
                GUI.Label(new Rect(s.x + 3f, Screen.height - s.y - 8f, 160f, 16f),
                          kv.Key, _mono);
            }
            GUI.color = antes;
        }

        private void MontaCns3D()
        {
            _pCns3D.Limpar();
            if (_rtCns == null || _cnsRoot == null) return;

            _pCns3D.Texto("MALE CNS", _rotulo);
            int lado = Mathf.Min(larguraCaixaCns, (int)_colDireita.largura - 24);
            _pCns3D.Desenho(lado, r =>
                GUI.DrawTexture(new Rect(r.x, r.y, lado, lado), _rtCns,
                                ScaleMode.ScaleToFit));

            // legenda do que esta pulsando: as arestas acendem quando o
            // pre-sinaptico dispara, e a cor e o SINAL (MODEL, regra de Dale)
            if (_grafo != null && _grafo.show)
            {
                _pCns3D.Texto($"sinapses  {_grafo.visibleEdges}/{_grafo.edgeCount} "
                              + $"visiveis   filtro {_grafo.filtro}", _mono);
                _pCns3D.Texto("  azul excitatoria   vermelho inibitoria", _mono,
                              ProvenanceUtil.Color(Provenance.Model));
            }
            else
            {
                _pCns3D.Texto("C liga o fluxo de sinapses", _mono);
            }
        }

        private void MontaGate()
        {
            _pGate.Limpar();
            if (_gfExc == 0f && _gfInib == 0f && _gfSpikes == 0) return;

            _pGate.Texto("GIANT FIBER -- GATE", _rotulo);
            var corExc = new Color(0.40f, 0.72f, 1.00f);
            var corInib = new Color(1.00f, 0.42f, 0.42f);
            _pGate.Texto($"excitacao {_gfExc,10:F1} mV", _mono, corExc);
            _pGate.Texto($"inibicao  {_gfInib,10:F1} mV", _mono, corInib);
            _pGate.Texto($"liquido   {_gfLiquido,10:F1} mV", _mono,
                         _gfLiquido >= 0 ? corExc : corInib);
            _pGate.Texto($"v min     {_gfVmin,10:F1} mV   (limiar -45,0)", _mono);
            _pGate.Texto($"spikes do GF {_gfSpikes}", _mono);

            if (_gfTopInib.Count > 0)
            {
                _pGate.Espacador();
                _pGate.Texto("maiores fontes de inibicao neste passo", _mono);
                foreach (var l in _gfTopInib) _pGate.Texto("  " + l, _mono, corInib);
            }

            // So aparece depois que o runtime avisou. Nao e alerta decorativo:
            // e o registro de que o modelo saiu da faixa fisiologica, e NAO ha
            // clamp -- o valor acima veio como o modelo produziu.
            if (_limitacaoDisparou && _limitacaoTexto.Length > 0)
            {
                _pGate.Espacador();
                _pGate.Texto($"LIMITACAO DO MODELO (v < {_limitacaoLimiar:F0} mV)",
                             _mono, ProvenanceUtil.Color(Provenance.Assumption));
                _pGate.Texto(_limitacaoTexto, _mono);
                _pGate.Texto("reportado, nunca corrigido por clamp", _mono);
            }
        }

        private void MontaRuntime()
        {
            _pRuntime.Limpar();
            if (_runtime.Count == 0) return;
            _pRuntime.Texto("RUNTIME", _rotulo);

            string Le(string k) => _runtime.TryGetValue(k, out var v) ? v : "-";
            _pRuntime.Texto($"OS       {Le("os")}", _mono);
            _pRuntime.Texto($"Physics  {Le("physics_backend")}", _mono);
            _pRuntime.Texto($"Neural   {Le("neural_backend")}", _mono);
            _pRuntime.Texto($"Device   {Le("neural_device")}", _mono);

            // SIMULADO e MODELADO sao coisas diferentes e ficam separados de
            // proposito: o conectoma inteiro participar da dinamica nao quer
            // dizer que exista modelo sensorimotor completo.
            if (_escopoSimulado.Length > 0)
            {
                _pRuntime.Espacador();
                _pRuntime.Texto($"Escopo simulado    {_escopoSimulado}", _mono,
                                ProvenanceUtil.Color(Provenance.Data));
                _pRuntime.Texto($"Modelo sensorimotor {_escopoSensorimotor}", _mono,
                                ProvenanceUtil.Color(Provenance.Assumption));
                if (_escopoNota.Length > 0) _pRuntime.Texto(_escopoNota, _mono);
            }

            if (_runtime.ContainsKey("neurons_simulated"))
            {
                _pRuntime.Espacador();
                _pRuntime.Texto($"Neurons  {Le("neurons_simulated")}", _mono);
                _pRuntime.Texto($"Edges    {Le("edges_simulated")}", _mono);
                _pRuntime.Texto($"VRAM     {Le("vram_mib")} MiB", _mono);
            }

            if (_profile.Count > 0)
            {
                _pRuntime.Espacador();
                _pRuntime.Texto("CUSTO (ms de relogio por segundo simulado)", _rotulo);
                foreach (var kv in _profile)
                    _pRuntime.Texto($"  {kv.Key,-10} {kv.Value,8:F0} ms", _mono);
            }
            _pRuntime.Espacador();
            _pRuntime.Texto($"RTF      {_rtf:F4}x", _mono);
        }

        private void MontaSinais()
        {
            _pSinais.Limpar();
            _pSinais.Texto("SINAIS (janela rolante)", _rotulo);
            foreach (var g in _graficos)
            {
                var grafico = g;   // capturado pelo lambda do desenho
                _pSinais.Texto(grafico.Rotulo, _mono, ProvenanceUtil.Color(grafico.procedencia));
                _pSinais.Desenho(34f, r =>
                    GUI.DrawTexture(new Rect(r.x, r.y + 2f, r.width, r.height - 4f),
                                    grafico.Desenhar(), ScaleMode.StretchToFill));
                _pSinais.Espacador(8f);
            }
        }

        private void MontaRetina()
        {
            _pRetina.Limpar();
            if (_retinaL == null && _retinaR == null) return;
            float lw = _retL.Largura, lh = _retL.Altura;
            _pRetina.Texto("VISAO  (721 omatideos/olho)", _rotulo);
            // Intensidade CRUA do omatideo, so limitada a 0-1 -- nao e
            // renormalizada por quadro. Dizer "normalizado" faria o usuario ler
            // contraste onde ha brilho absoluto.
            _pRetina.Texto("escala: 0 escuro -> 1 claro (valor cru, limitado a 0-1)", _mono);
            _pRetina.Texto("LEFT EYE", _mono);
            _pRetina.Desenho(lh, r => GUI.DrawTexture(new Rect(r.x, r.y, lw, lh),
                                                      _retL.Desenhar(_retinaL),
                                                      ScaleMode.StretchToFill));
            _pRetina.Texto("RIGHT EYE", _mono);
            _pRetina.Desenho(lh, r => GUI.DrawTexture(new Rect(r.x, r.y, lw, lh),
                                                      _retR.Desenhar(_retinaR),
                                                      ScaleMode.StretchToFill));
        }

        private void MontaPopulacao()
        {
            _pPopulacao.Limpar();
            if (_brain == null || _brain.AtividadeAgregada.Count == 0) return;

            bool algum = false;
            foreach (var kv in _brain.AtividadeAgregada)
            {
                if (!_brain.SemGeometriaPorCamada.TryGetValue(kv.Key, out int semGeo) || semGeo <= 0)
                    continue;
                if (!algum) { _pPopulacao.Texto("ATIVIDADE DE POPULACAO", _rotulo); algum = true; }
                // Barra de populacao: os neuronios sem morfologia exportada
                // contribuem aqui, e NAO num ponto inventado do cerebro.
                int blocos = Mathf.Clamp(Mathf.RoundToInt(kv.Value / 3f), 0, 28);
                _pPopulacao.Texto($"{kv.Key}  ({semGeo} sem morfologia individual)", _mono);
                _pPopulacao.Texto("  " + new string('#', blocos), _mono);
            }
        }

        private void MontaInspector()
        {
            _pInspector.Limpar();
            long sel = _brain != null ? _brain.SelectedBodyId : -1;
            _pInspector.Texto("INSPECTOR", _rotulo);

            if (sel < 0)
            {
                // Sem selecao, o espaco vira o aviso de amostragem. Afirmar ou
                // sugerir que a morfologia mostrada e a populacao toda seria
                // falso, e este e o lugar onde o usuario olharia.
                if (_brain != null)
                    foreach (var c in _brain.Coverage)
                    {
                        _pInspector.Texto(c.group, _mono);
                        _pInspector.Texto($"  Simulated: {c.total_simulated} neurons", _mono);
                        _pInspector.Texto($"  3D morphology shown: {c.total_visualized} representative", _mono);
                        _pInspector.Texto($"  Synaptic-weight coverage: {c.fraction_weight_covered * 100f:F0}%", _mono);
                        _pInspector.Texto("  DATA - morphology subset", _mono,
                                          ProvenanceUtil.Color(Provenance.Data));
                    }
                _pInspector.Texto("botao direito seleciona um neuronio", _mono);
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
            Campo("bodyId", Get("bodyId"), Provenance.Data);
            Campo("type", Get("type"), Provenance.Data);
            Campo("group", Get("group"), Provenance.Data);
            Campo("side", Get("side"), Provenance.Data);
            Campo("neurotransmitter", Get("neurotransmitter"), Provenance.Data);
            Campo("polarity", Get("polarity"), Provenance.Model);
            Campo("atividade", Get("recentActivity"), Provenance.Model);
            if (_grafo != null)
            {
                var (entra, sai, nE, nS) = _grafo.Resumo(sel);
                Campo("sinapses in", $"{entra} ({nE} parceiros)", Provenance.Data);
                Campo("sinapses out", $"{sai} ({nS} parceiros)", Provenance.Data);
            }
        }

        private void Campo(string chave, string valor, Provenance p)
        {
            _pInspector.Desenho(17f, r => Badge.Desenha(r, chave, valor, p, _mono));
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

            // a caixa do cerebro gira sozinha, devagar: sem isso ela fica uma
            // silhueta chapada e nao da pra ler a profundidade das arestas
            if (_camCns != null && _cnsRoot != null)
            {
                _cnsYaw += giroCns * Time.deltaTime;
                var alvo = _cnsRoot.GetComponent<Renderer>() != null
                    ? _cnsRoot.position : _cnsRoot.position;
                var rot = Quaternion.Euler(14f, _cnsYaw, 0f);
                _camCns.transform.position = alvo + rot * new Vector3(0, 0, -16f);
                _camCns.transform.rotation = rot;
            }
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

    /// <summary>Camera de orbita: arrastar gira, scroll aproxima, meio empurra.</summary>
    public class OrbitCamera : MonoBehaviour
    {
        /// <summary>Pontos de vista fixos pra inspecionar o corpo.</summary>
        public enum Preset { Perspectiva, Lateral, Topo, Frente }

        public Vector3 target = Vector3.zero;
        [Tooltip("Deslocamento manual do alvo (botao do meio). Zerado ao focar.")]
        public Vector3 pan = Vector3.zero;
        // 15.5 em vez de 18: com o centro da tela livre de painel, o CNS cabe
        // ~16% maior. E aproximacao de camera, nao escala do objeto -- a
        // geometria e as posicoes que o grafo de conectividade calcula ficam
        // exatamente como estavam.
        public float distance = 15.5f;
        public float yaw = 0f, pitch = 12f;

        [Tooltip("Segue este transform. A mosca ANDA -- sem isto ela sai de " +
                 "quadro em poucos segundos e a cena parece vazia.")]
        public Transform seguir;
        public float suavizacaoSeguir = 6f;

        /// <summary>
        /// Aponta a camera de um dos lados. So muda angulo -- a distancia
        /// continua sendo a que `Focar` calculou, senao trocar de vista
        /// desenquadra a mosca toda vez.
        /// </summary>
        public void Aponta(Preset p)
        {
            // A camera fica em `target + Euler(pitch,yaw,0) * (0,0,-d)`, entao
            // yaw 0 poe a camera em -Z e olhando pra +Z. Na Unity, depois da
            // conversao, +X e a frente da mosca e +Z e o lado dela -- por isso
            // "lateral" e yaw 0 e nao 90. Ver docs/UNITY_BODY_COORDINATES.md.
            switch (p)
            {
                case Preset.Lateral: yaw = 0f; pitch = 0f; break;
                case Preset.Topo: yaw = 0f; pitch = 89f; break;
                case Preset.Frente: yaw = -90f; pitch = 0f; break;
                default: yaw = -40f; pitch = 22f; break;   // 3/4 pela frente
            }
            preset = p;
        }

        public Preset preset = Preset.Perspectiva;

        /// <summary>
        /// Enquadra o que cabe dentro de `b`, com folga. Usado pelo botao
        /// "Focar mosca": a mosca tem ~3 mm de corpo mas ~5 mm com as pernas
        /// abertas, e chutar a distancia corta perna em metade das vistas.
        /// </summary>
        public void Focar(Bounds b)
        {
            pan = Vector3.zero;
            target = b.center;
            float raio = b.extents.magnitude;
            var cam = GetComponent<Camera>();
            float fov = cam != null ? cam.fieldOfView : 60f;
            // raio / sin(fov/2) enquadra a esfera que contem o corpo; o 1.25
            // e a folga pra nao encostar nas bordas
            distance = Mathf.Clamp(
                1.25f * raio / Mathf.Sin(fov * 0.5f * Mathf.Deg2Rad), 1.5f, 120f);
        }

        void LateUpdate()
        {
            if (seguir != null)
            {
                // Segue o torax, mas NAO colado: `pan` fica por fora do lerp,
                // entao o que o usuario empurrou continua valendo enquanto a
                // mosca anda.
                target = Vector3.Lerp(target, seguir.position,
                                      1f - Mathf.Exp(-suavizacaoSeguir * Time.deltaTime));
            }
            var rot = Quaternion.Euler(pitch, yaw, 0f);
            if (Input.GetMouseButton(0))
            {
                yaw += Input.GetAxis("Mouse X") * 3f;
                pitch = Mathf.Clamp(pitch - Input.GetAxis("Mouse Y") * 2f, -89f, 89f);
            }
            if (Input.GetMouseButton(2))
            {
                // empurrar no plano da tela, proporcional a distancia: perto
                // move pouco, longe move muito -- senao pan fica inutil em zoom
                float k = distance * 0.0015f;
                pan -= rot * new Vector3(Input.GetAxis("Mouse X") * k,
                                         Input.GetAxis("Mouse Y") * k, 0f);
            }
            distance = Mathf.Clamp(distance - Input.mouseScrollDelta.y * 1.5f, 1.5f, 120f);
            rot = Quaternion.Euler(pitch, yaw, 0f);
            transform.position = target + pan + rot * new Vector3(0, 0, -distance);
            transform.rotation = rot;
        }
    }
}
