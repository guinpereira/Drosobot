// A mosca do NeuroMechFly na Unity -- a de verdade, nao um marcador.
//
// As malhas sao as do modelo que a FISICA roda: exportadas do modelo compilado
// do FlyGym 2.x por tools/export_fly_mesh.py, com o transform local de cada
// geom. O que aparece aqui corresponde ao corpo que esta sendo simulado.
//
// ## Esta classe nao simula nada
//
// Ela reproduz a pose que chega pela telemetria (`body_pose` no `frame`). A
// fisica e do MuJoCo. A Unity pode INTERPOLAR entre dois quadros pra suavizar,
// porque a pose chega a 30 Hz e a interface desenha a 60+ -- mas nada do que
// ela calcula volta pra simulacao.
//
// ## Conversao de eixos
//
// Nao acontece aqui. Toda troca de eixo passa por `MujocoFrame`, um lugar so.
// Ver docs/UNITY_BODY_COORDINATES.md.
//
// ## Pose em MUNDO
//
// Cada segmento chega com transform de MUNDO, entao a raiz visual fica na
// identidade. Mover a raiz E aplicar pose de mundo soma o deslocamento duas
// vezes -- ja aconteceu. `MujocoFrame.RaizNeutra` verifica isso todo quadro.

using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Drosobot.Lab
{
    public class FlyBody : MonoBehaviour
    {
        /// <summary>O que o Lab esta mostrando do corpo.</summary>
        public enum Modo
        {
            /// <summary>Pose viva, vinda da telemetria.</summary>
            Normal,
            /// <summary>Pose de repouso do MODELO, parada. Nao usa telemetria.</summary>
            BindPose,
            /// <summary>Pose viva + eixos locais nos segmentos principais.</summary>
            Eixos,
            /// <summary>Pose viva + nome de cada segmento.</summary>
            Rotulos,
        }

        [Header("Estado")]
        public int segmentosMontados;
        public int segmentosRecebidos;
        public bool poseRecebida;
        public Modo modo = Modo.Normal;

        [Header("Exibicao")]
        [Tooltip("Escala de mundo. O modelo esta em mm; 1 unidade Unity = 1 mm.")]
        public float escala = 1f;
        [Tooltip("Suavizacao entre quadros de pose. Puramente visual. Alta de " +
                 "proposito: a pose chega a 30 Hz e a mosca se desloca rapido, " +
                 "entao suavizacao baixa faz os segmentos ficarem pra tras e a " +
                 "mosca aparecer desmontada.")]
        public float suavizacao = 60f;
        [Tooltip("Comprimento dos eixos locais no modo Eixos, em mm.")]
        public float tamanhoEixo = 0.35f;
        [Tooltip("Esquema de cor. O modelo NAO traz cor -- os 69 geoms sao " +
                 "cinza 0,5. Ver FlyAppearance.cs.")]
        public Aparencia aparencia = Aparencia.Realista;

        private class Seg
        {
            public Transform t;
            public Vector3 alvoPos;
            public Quaternion alvoRot;
            public bool temAlvo;
            public Vector3 bindPos;      // pose de repouso do modelo, em MUNDO
            public Quaternion bindRot;
            public string nome;
        }

        private readonly Dictionary<string, Seg> _porNome = new Dictionary<string, Seg>();
        private readonly List<Seg> _ordem = new List<Seg>();
        private Transform _raiz;
        // um renderer por geom, com o nome do segmento: e o que permite trocar
        // o esquema de cor sem remontar a mosca
        private readonly List<KeyValuePair<string, Renderer>> _pintura =
            new List<KeyValuePair<string, Renderer>>();
        private Aparencia _aparenciaAplicada = (Aparencia)(-1);
        private GameObject _eixos;
        private bool _avisouRaiz;

        /// <summary>Monta a mosca a partir de Resources/Fly/fly_body.json.</summary>
        public bool Montar()
        {
            var meta = Resources.Load<TextAsset>("Fly/fly_body");
            if (meta == null)
            {
                Debug.LogWarning("[fly] Resources/Fly/fly_body.json ausente. " +
                                 "Gere com: .venv-flygym2\\Scripts\\python " +
                                 "tools\\export_fly_mesh.py");
                return false;
            }

            var raizGo = new GameObject("NeuroMechFly");
            raizGo.transform.SetParent(transform, false);
            _raiz = raizGo.transform;

            var doc = JObject.Parse(meta.text);
            foreach (var g in doc["geoms"])
            {
                // O modelo compilado prefixa os corpos com o nome da mosca
                // (`fly/c_thorax`); a pose chega com o nome do segmento
                // (`c_thorax`). Normalizamos dos DOIS lados, aqui e no
                // AplicarPose, senao nada casa e a mosca fica parada.
                string body = Normaliza((string)g["body"]);
                string mesh = (string)g["mesh"];
                var malha = CarregaMalha(mesh.Replace('/', '_'));
                if (malha == null) continue;

                // um GameObject por segmento; o geom entra como filho com o
                // transform LOCAL dele, que e o que monta a peca no lugar certo
                if (!_porNome.TryGetValue(body, out var seg))
                {
                    var go = new GameObject(body);
                    go.transform.SetParent(_raiz, false);
                    seg = new Seg { t = go.transform, nome = body };
                    _porNome[body] = seg;
                    _ordem.Add(seg);
                }

                var filho = new GameObject(mesh);
                filho.transform.SetParent(seg.t, false);
                var mf = filho.AddComponent<MeshFilter>();
                mf.sharedMesh = malha;
                var mr = filho.AddComponent<MeshRenderer>();
                mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                _pintura.Add(new KeyValuePair<string, Renderer>(body, mr));

                filho.transform.localPosition = MujocoFrame.Pos(Tres(g["pos"]));
                filho.transform.localRotation = MujocoFrame.Quat(Quatro(g["quat_wxyz"]));
            }

            AplicaAparencia();
            LeBindPose(doc["pose_repouso"] as JObject);
            AplicaBindPose();   // nasce em repouso; a telemetria assume depois

            segmentosMontados = _porNome.Count;
            Debug.Log($"[fly] {segmentosMontados} segmentos montados a partir das " +
                      "malhas reais do NeuroMechFly");
            return segmentosMontados > 0;
        }

        // ------------------------------------------------------- bind pose
        //
        // Vem PRONTA do exportador, em MUNDO: sao os xpos/xquat que o MuJoCo
        // tem depois de mj_forward na qpos padrao do modelo.
        //
        // Ja foi calculada aqui, compondo `body_pos`/`body_quat` da arvore de
        // corpos, e estava ERRADA: essa arvore e a cinematica com as juntas em
        // ZERO, e as 73 qpos do modelo sao todas nao-nulas (postura de pe).
        // Compor ignorando as juntas erra ate 1,77 mm numa mosca de 2,7 mm --
        // as pernas descem coladas na linha media, e a mosca aparece
        // desmontada. Isso apareceu no render, nao nos numeros: a cadeia
        // coxa > femur > tibia > tarso continuava descendo e simetrica, entao
        // as verificacoes numericas passavam.
        //
        // Sendo MUNDO, e a mesma grandeza que a telemetria manda: a pose de
        // repouso e a pose viva percorrem o mesmo caminho de codigo.
        private void LeBindPose(JObject repouso)
        {
            if (repouso == null) return;
            var segs = repouso["segments"] as JArray;
            var pos = repouso["pos"] as JArray;
            var quat = repouso["quat_wxyz"] as JArray;
            if (segs == null || pos == null || quat == null) return;

            int n = Mathf.Min(segs.Count, Mathf.Min(pos.Count, quat.Count));
            for (int i = 0; i < n; i++)
            {
                if (!_porNome.TryGetValue(Normaliza((string)segs[i]), out var seg))
                    continue;
                seg.bindPos = MujocoFrame.Pos(Tres(pos[i])) * escala;
                seg.bindRot = MujocoFrame.Quat(Quatro(quat[i]));
            }
        }

        private void AplicaBindPose()
        {
            foreach (var seg in _ordem)
            {
                seg.t.localPosition = seg.bindPos;
                seg.t.localRotation = seg.bindRot;
            }
        }

        /// <summary>Centro e extensao da mosca na pose atual, pra enquadrar camera.</summary>
        public Bounds Extensao()
        {
            if (_ordem.Count == 0) return new Bounds(Vector3.zero, Vector3.one);
            var b = new Bounds(_ordem[0].t.position, Vector3.zero);
            foreach (var seg in _ordem) b.Encapsulate(seg.t.position);
            b.Expand(0.4f);   // as malhas passam um pouco da origem do segmento
            return b;
        }

        /// <summary>Aplica a pose que veio no `body_pose` do frame.</summary>
        public void AplicarPose(JObject pose)
        {
            var segs = pose["segments"] as JArray;
            var pos = pose["pos"] as JArray;
            var quat = pose["quat"] as JArray;
            if (segs == null || pos == null || quat == null) return;

            int n = Mathf.Min(segs.Count, Mathf.Min(pos.Count, quat.Count));
            segmentosRecebidos = n;
            int casou = 0;
            for (int i = 0; i < n; i++)
            {
                if (!_porNome.TryGetValue(Normaliza((string)segs[i]), out var seg))
                    continue;
                casou++;
                seg.alvoPos = MujocoFrame.Pos(Tres(pos[i])) * escala;
                seg.alvoRot = MujocoFrame.Quat(Quatro(quat[i]));
                if (!seg.temAlvo)
                {
                    // Primeira pose: assenta direto. Interpolar a partir da
                    // bind pose faria os segmentos atravessarem a cena ate
                    // alcancar o corpo.
                    seg.t.localPosition = seg.alvoPos;
                    seg.t.localRotation = seg.alvoRot;
                }
                seg.temAlvo = true;
            }
            poseRecebida = casou > 0;
        }

        /// <summary>
        /// Pinta cada geom pelo nome do segmento. E APARENCIA: o modelo nao
        /// traz cor e nenhuma destas cores codifica grandeza. Ver
        /// FlyAppearance.cs.
        /// </summary>
        public void AplicaAparencia()
        {
            if (_aparenciaAplicada == aparencia) return;
            _aparenciaAplicada = aparencia;
            // um material por CATEGORIA, nao por geom: sao 69 geoms e oito
            // categorias, e 69 materiais iguais so gastariam draw call
            var cache = new Dictionary<string, Material>();
            foreach (var kv in _pintura)
            {
                string cat = FlyAppearance.Categoria(kv.Key);
                if (!cache.TryGetValue(cat, out var mat))
                {
                    mat = FlyAppearance.Material(kv.Key, aparencia);
                    cache[cat] = mat;
                }
                kv.Value.sharedMaterial = mat;
            }
        }

        void Update()
        {
            AplicaAparencia();   // barato: sai na primeira linha se nao mudou

            // A pose que chega e de MUNDO. Se alguem mover a raiz, cada
            // segmento sai deslocado pelo valor dela -- o bug do offset
            // duplicado. Avisa uma vez, alto, em vez de deixar a mosca
            // "estranha" sem explicacao.
            if (!_avisouRaiz && _raiz != null &&
                !MujocoFrame.RaizNeutra(_raiz, EspacoPose.Mundo, out string erro))
            {
                _avisouRaiz = true;
                Debug.LogError("[fly] " + erro);
            }

            if (modo == Modo.BindPose)
            {
                AplicaBindPose();       // congelado: telemetria ignorada
                DesenhaEixos(false);
                return;
            }

            if (poseRecebida)
            {
                // Interpolacao PURAMENTE visual: a pose chega a 30 Hz e a
                // interface desenha a 60+. Nada disto volta pra fisica.
                float k = 1f - Mathf.Exp(-suavizacao * Time.deltaTime);
                foreach (var seg in _ordem)
                {
                    if (!seg.temAlvo) continue;
                    seg.t.localPosition = Vector3.Lerp(seg.t.localPosition,
                                                       seg.alvoPos, k);
                    seg.t.localRotation = Quaternion.Slerp(seg.t.localRotation,
                                                           seg.alvoRot, k);
                }
            }

            DesenhaEixos(modo == Modo.Eixos);
        }

        // --------------------------------------------------------- eixos
        //
        // Segmentos escolhidos: torax e a cadeia de cada perna. Desenhar os 69
        // vira um novelo e nao se le nada.
        // Os nomes sao os do modelo COMPILADO, minusculos e com underscore
        // (`lf_coxa`), nao os nomes de artigo (`LFCoxa`). O femur e o
        // `trochanterfemur`: no NeuroMechFly trocanter e femur sao um corpo so.
        private static readonly string[] AlvosEixo = {
            "c_thorax",
            "lf_coxa", "lf_trochanterfemur", "lf_tibia", "lf_tarsus1",
            "rf_coxa", "rf_trochanterfemur", "rf_tibia", "rf_tarsus1",
            "lm_coxa", "lm_trochanterfemur", "lm_tibia", "lm_tarsus1",
            "rm_coxa", "rm_trochanterfemur", "rm_tibia", "rm_tarsus1",
            "lh_coxa", "lh_trochanterfemur", "lh_tibia", "lh_tarsus1",
            "rh_coxa", "rh_trochanterfemur", "rh_tibia", "rh_tarsus1",
        };

        private readonly List<LineRenderer> _linhas = new List<LineRenderer>();

        private void DesenhaEixos(bool ligado)
        {
            if (!ligado)
            {
                if (_eixos != null) _eixos.SetActive(false);
                return;
            }
            if (_eixos == null) CriaEixos();
            _eixos.SetActive(true);

            int i = 0;
            foreach (var nome in AlvosEixo)
            {
                if (!_porNome.TryGetValue(nome, out var seg)) { i += 3; continue; }
                var p = seg.t.position;
                var r = seg.t.rotation;
                Ponta(i++, p, p + r * Vector3.right * tamanhoEixo);
                Ponta(i++, p, p + r * Vector3.up * tamanhoEixo);
                Ponta(i++, p, p + r * Vector3.forward * tamanhoEixo);
            }
        }

        private void Ponta(int i, Vector3 a, Vector3 b)
        {
            if (i >= _linhas.Count) return;
            _linhas[i].SetPosition(0, a);
            _linhas[i].SetPosition(1, b);
        }

        private void CriaEixos()
        {
            _eixos = new GameObject("SegmentAxes");
            _eixos.transform.SetParent(transform, false);
            var sh = Shader.Find("Sprites/Default") ?? Shader.Find("Unlit/Color");
            // X vermelho, Y verde, Z ciano -- eixos da UNITY, ja convertidos.
            // Nao sao os eixos do MuJoCo; ver docs/UNITY_BODY_COORDINATES.md.
            Color[] cores = { Color.red, Color.green, Color.cyan };
            for (int s = 0; s < AlvosEixo.Length; s++)
            {
                for (int e = 0; e < 3; e++)
                {
                    var go = new GameObject($"{AlvosEixo[s]}_{e}");
                    go.transform.SetParent(_eixos.transform, false);
                    var lr = go.AddComponent<LineRenderer>();
                    lr.material = new Material(sh) { color = cores[e] };
                    lr.startColor = lr.endColor = cores[e];
                    lr.widthMultiplier = 0.02f;
                    lr.positionCount = 2;
                    lr.useWorldSpace = true;
                    lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    _linhas.Add(lr);
                }
            }
        }

        /// <summary>Nome e posicao de mundo de cada segmento, pros rotulos.</summary>
        public IEnumerable<KeyValuePair<string, Vector3>> Segmentos()
        {
            foreach (var seg in _ordem)
                yield return new KeyValuePair<string, Vector3>(seg.nome, seg.t.position);
        }

        /// <summary>Tira o prefixo do modelo: `fly/c_thorax` -> `c_thorax`.</summary>
        private static string Normaliza(string nome)
        {
            if (string.IsNullOrEmpty(nome)) return "";
            int i = nome.LastIndexOf('/');
            return i >= 0 ? nome.Substring(i + 1) : nome;
        }

        private static float[] Tres(JToken t)
            => new[] { (float)t[0], (float)t[1], (float)t[2] };

        private static float[] Quatro(JToken t)
            => new[] { (float)t[0], (float)t[1], (float)t[2], (float)t[3] };

        // Unity importa OBJ como GameObject; a Mesh e SUB-asset, e
        // Resources.Load<Mesh> nao a encontra pelo caminho do arquivo. LoadAll
        // devolve o conjunto e a gente pega a malha.
        private readonly Dictionary<string, Mesh> _cacheMalha = new Dictionary<string, Mesh>();

        private Mesh CarregaMalha(string nome)
        {
            if (_cacheMalha.TryGetValue(nome, out var m)) return m;
            m = Resources.Load<Mesh>($"Fly/{nome}");
            if (m == null)
            {
                foreach (var o in Resources.LoadAll($"Fly/{nome}"))
                {
                    if (o is Mesh mm) { m = mm; break; }
                    if (o is GameObject go)
                    {
                        var mf = go.GetComponentInChildren<MeshFilter>();
                        if (mf != null && mf.sharedMesh != null) { m = mf.sharedMesh; break; }
                    }
                }
            }
            _cacheMalha[nome] = m;
            return m;
        }

        public Transform Raiz => _raiz;
        public bool TemSegmento(string nome) => _porNome.ContainsKey(nome);
    }
}
