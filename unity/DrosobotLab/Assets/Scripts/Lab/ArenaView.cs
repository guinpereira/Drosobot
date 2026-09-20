// A arena e o estimulo, desenhados a partir do que a simulacao manda.
//
// Ate agora a mosca aparecia flutuando no vazio: o chao existia no MuJoCo e o
// estimulo tambem, mas nenhum dos dois tinha representacao na tela. Ver uma
// mosca andando sobre nada e pior que uma tela vazia -- da a impressao de que
// ela esta voando, e esconde justamente o objeto que dispara a fuga.
//
// ## A Unity nao calcula a trajetoria
//
// A posicao da esfera chega MEDIDA, quadro a quadro, de quem a move no MuJoCo
// (`PhysicsAdapter.estado_estimulo`). Reproduzir a formula aqui seria facil --
// o `scene_info` manda ciclo, distancia inicial e final -- e seria a interface
// simulando. O raio tambem vem de la, do mesmo `RAIO_ESTIMULO` que constroi o
// mundo, pra que a esfera na tela tenha o tamanho que a retina de fato viu.

using UnityEngine;

namespace Drosobot.Lab
{
    public class ArenaView : MonoBehaviour
    {
        [Header("Estado")]
        public bool temEstimulo;
        public float distanciaMm;

        private Transform _chao;
        private Transform _esfera;
        private Renderer _rEsfera;
        private Material _matEsfera;
        private Vector3 _alvoEsfera;
        private bool _primeiraPose = true;

        /// <summary>Monta chao e esfera. Ambos comecam escondidos.</summary>
        public void Montar(Transform pai)
        {
            var sh = Shader.Find("Standard") ?? Shader.Find("Diffuse");

            // Chao: um quad grande e escuro, so pra dar plano de referencia. A
            // arena do MuJoCo e um plano infinito; desenhar "infinito" nao da,
            // entao e um pedaco grande o bastante pra mosca nunca sair dele na
            // escala de tempo do experimento.
            var chao = GameObject.CreatePrimitive(PrimitiveType.Plane);
            chao.name = "Arena";
            Destroy(chao.GetComponent<Collider>());   // a Unity nao simula nada
            chao.transform.SetParent(pai, false);
            chao.transform.localScale = new Vector3(12f, 1f, 12f);  // 120 mm
            var matChao = new Material(sh)
            {
                color = new Color(0.115f, 0.125f, 0.145f),
            };
            matChao.SetFloat("_Glossiness", 0.08f);
            matChao.SetFloat("_Metallic", 0f);
            chao.GetComponent<Renderer>().sharedMaterial = matChao;
            chao.GetComponent<Renderer>().shadowCastingMode =
                UnityEngine.Rendering.ShadowCastingMode.Off;
            _chao = chao.transform;

            var esfera = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            esfera.name = "EstimuloLooming";
            Destroy(esfera.GetComponent<Collider>());
            esfera.transform.SetParent(pai, false);
            _matEsfera = new Material(sh) { color = new Color(0.62f, 0.16f, 0.16f) };
            _matEsfera.SetFloat("_Glossiness", 0.25f);
            _matEsfera.SetFloat("_Metallic", 0f);
            _rEsfera = esfera.GetComponent<Renderer>();
            _rEsfera.sharedMaterial = _matEsfera;
            _rEsfera.shadowCastingMode =
                UnityEngine.Rendering.ShadowCastingMode.Off;
            _esfera = esfera.transform;
            _esfera.gameObject.SetActive(false);
        }

        /// <summary>Chao visivel? `flat` e `looming` tem chao; nada mais tem.</summary>
        public void DefineArena(string tipo)
        {
            if (_chao != null)
                _chao.gameObject.SetActive(tipo == "flat" || tipo == "looming");
        }

        /// <summary>
        /// Posicao e raio medidos, em coordenadas do MuJoCo. A conversao de
        /// eixos passa por `MujocoFrame`, como todo o resto.
        /// </summary>
        public void AplicaEstimulo(float x, float y, float z, float raio)
        {
            if (_esfera == null) return;
            if (!_esfera.gameObject.activeSelf)
            {
                _esfera.gameObject.SetActive(true);
                _primeiraPose = true;
            }
            temEstimulo = true;
            _alvoEsfera = MujocoFrame.Pos(x, y, z);
            // o primitivo da Unity tem 1 unidade de DIAMETRO, entao a escala e
            // o raio dobrado
            _esfera.localScale = Vector3.one * (raio * 2f);
            if (_primeiraPose)
            {
                _esfera.localPosition = _alvoEsfera;
                _primeiraPose = false;
            }
        }

        public void SemEstimulo()
        {
            temEstimulo = false;
            if (_esfera != null) _esfera.gameObject.SetActive(false);
        }

        void Update()
        {
            if (_esfera == null || !temEstimulo) return;
            // A posicao chega a 30 Hz junto com a pose; interpolar e so
            // suavizacao visual, como na mosca. Nada disto volta pra fisica.
            float k = 1f - Mathf.Exp(-30f * Time.deltaTime);
            _esfera.localPosition = Vector3.Lerp(_esfera.localPosition,
                                                 _alvoEsfera, k);
        }
    }
}
