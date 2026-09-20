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
// ## Conversao de eixos e de quaternio
//
// O exportador entrega os numeros como o MuJoCo os tem, e a conversao acontece
// aqui, num lugar so:
//
//     MuJoCo   Z-up, destro,   quaternio (w, x, y, z),  milimetros
//     Unity    Y-up, canhoto,  quaternio (x, y, z, w)
//
// Converter nos dois lados seria o jeito mais facil de aplicar a rotacao duas
// vezes e passar semanas procurando o erro.

using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Drosobot.Lab
{
    public class FlyBody : MonoBehaviour
    {
        [Header("Estado")]
        public int segmentosMontados;
        public int segmentosRecebidos;
        public bool poseRecebida;

        [Header("Exibicao")]
        [Tooltip("Escala de mundo. O modelo esta em mm; 1 unidade Unity = 1 mm.")]
        public float escala = 1f;
        [Tooltip("Suavizacao entre quadros de pose. Puramente visual. Alta de " +
                 "proposito: a pose chega a 30 Hz e a mosca se desloca rapido, " +
                 "entao suavizacao baixa faz os segmentos ficarem pra tras e a " +
                 "mosca aparecer desmontada.")]
        public float suavizacao = 60f;

        private class Seg
        {
            public Transform t;
            public Vector3 alvoPos;
            public Quaternion alvoRot;
            public bool temAlvo;
        }

        private readonly Dictionary<string, Seg> _porNome = new Dictionary<string, Seg>();
        private Transform _raiz;
        private Material _mat;

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

            var sh = Shader.Find("Standard") ?? Shader.Find("Diffuse");
            _mat = new Material(sh) { color = new Color(0.72f, 0.66f, 0.52f) };

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
                    seg = new Seg { t = go.transform };
                    _porNome[body] = seg;
                }

                var filho = new GameObject(mesh);
                filho.transform.SetParent(seg.t, false);
                var mf = filho.AddComponent<MeshFilter>();
                mf.sharedMesh = malha;
                var mr = filho.AddComponent<MeshRenderer>();
                mr.sharedMaterial = _mat;
                mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

                var p = g["pos"];
                var q = g["quat_wxyz"];
                filho.transform.localPosition = DeMujoco(
                    (float)p[0], (float)p[1], (float)p[2]);
                filho.transform.localRotation = DeMujocoQuat(
                    (float)q[0], (float)q[1], (float)q[2], (float)q[3]);
            }

            segmentosMontados = _porNome.Count;
            Debug.Log($"[fly] {segmentosMontados} segmentos montados a partir das " +
                      "malhas reais do NeuroMechFly");
            return segmentosMontados > 0;
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
                var p = pos[i];
                var q = quat[i];
                seg.alvoPos = DeMujoco((float)p[0], (float)p[1], (float)p[2]) * escala;
                seg.alvoRot = DeMujocoQuat((float)q[0], (float)q[1],
                                           (float)q[2], (float)q[3]);
                if (!seg.temAlvo)
                {
                    // Primeira pose: assenta direto. Interpolar a partir da
                    // origem faria os segmentos atravessarem a cena inteira
                    // ate alcancar o corpo.
                    seg.t.localPosition = seg.alvoPos;
                    seg.t.localRotation = seg.alvoRot;
                }
                seg.temAlvo = true;
            }
            poseRecebida = casou > 0;
        }

        void Update()
        {
            if (!poseRecebida) return;
            // Interpolacao PURAMENTE visual: a pose chega a 30 Hz e a interface
            // desenha a 60+. Nada disto volta pra fisica.
            float k = 1f - Mathf.Exp(-suavizacao * Time.deltaTime);
            foreach (var seg in _porNome.Values)
            {
                if (!seg.temAlvo) continue;
                seg.t.localPosition = Vector3.Lerp(seg.t.localPosition, seg.alvoPos, k);
                seg.t.localRotation = Quaternion.Slerp(seg.t.localRotation,
                                                       seg.alvoRot, k);
            }
        }

        /// <summary>Tira o prefixo do modelo: `fly/c_thorax` -> `c_thorax`.</summary>
        private static string Normaliza(string nome)
        {
            if (string.IsNullOrEmpty(nome)) return "";
            int i = nome.LastIndexOf('/');
            return i >= 0 ? nome.Substring(i + 1) : nome;
        }

        // MuJoCo Z-up destro -> Unity Y-up canhoto: troca Y e Z.
        private static Vector3 DeMujoco(float x, float y, float z)
            => new Vector3(x, z, y);

        // MuJoCo (w,x,y,z) -> Unity (x,y,z,w), com a mesma troca de eixos e a
        // inversao de sinal que a mudanca de quiralidade exige.
        private static Quaternion DeMujocoQuat(float w, float x, float y, float z)
            => new Quaternion(-x, -z, -y, w);

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
