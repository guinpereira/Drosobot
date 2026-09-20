// Renderiza a mosca em bind pose, das vistas de inspecao, sem abrir o Editor.
//
//   Unity.exe -batchmode -quit ^
//     -projectPath unity\DrosobotLab ^
//     -executeMethod Drosobot.EditorTools.CaptureBodyShots.Capturar ^
//     -logFile unity\capture.log
//
// (sem -nographics: e preciso GPU pra renderizar)
//
// POR QUE EM EDIT MODE: a bind pose nao depende de telemetria nenhuma -- e a
// pose de repouso do modelo, montada por FlyBody.Montar(). Capturar assim
// separa duas perguntas que estavam juntas: "a montagem e a conversao de eixos
// estao certas?" (isto aqui responde) e "a pose que chega esta certa?" (so a
// corrida ao vivo responde). Se a mosca parece errada nestas imagens, o
// problema NAO e da simulacao.

using System.IO;
using UnityEditor;
using UnityEngine;
using Drosobot.Lab;

namespace Drosobot.EditorTools
{
    public static class CaptureBodyShots
    {
        private const int Lado = 1200;
        private static readonly string Saida =
            Path.GetFullPath(Path.Combine(Application.dataPath, "../../../docs/images"));

        [MenuItem("Drosobot/Capturar Bind Pose")]
        public static void Capturar()
        {
            var raiz = new GameObject("CaptureRig");
            var mosca = raiz.AddComponent<FlyBody>();
            if (!mosca.Montar())
            {
                Debug.LogError("[capture] malhas ausentes -- rode tools/export_fly_mesh.py");
                Object.DestroyImmediate(raiz);
                return;
            }

            // mesma iluminacao do Lab, pra que a referencia e a tela batam
            var luzGo = new GameObject("Luzes");
            FlyAppearance.Ilumina(luzGo.transform);

            var camGo = new GameObject("CaptureCam");
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.06f, 0.07f, 0.09f);
            cam.fieldOfView = 40f;
            cam.nearClipPlane = 0.05f;

            // Diagnostico: o que o importador de OBJ da Unity fez com os
            // vertices. Comparar com os limites que o MuJoCo tem pra mesma
            // malha diz se houve troca de eixo ou negacao na importacao.
            foreach (var nome in new[] { "fly_lf_tarsus3", "fly_lf_tibia", "fly_c_thorax" })
            {
                Mesh malha = null;
                foreach (var o in Resources.LoadAll("Fly/" + nome))
                {
                    if (o is Mesh mm) { malha = mm; break; }
                    if (o is GameObject go)
                    {
                        var mf = go.GetComponentInChildren<MeshFilter>();
                        if (mf != null) { malha = mf.sharedMesh; break; }
                    }
                }
                if (malha != null)
                    Debug.Log($"[capture] malha {nome} centro {malha.bounds.center:F4} " +
                              $"tamanho {malha.bounds.size:F4}");
            }

            var b = mosca.Extensao();
            Debug.Log($"[capture] centro {b.center}  extensao {b.size}  " +
                      $"{mosca.segmentosMontados} segmentos");

            // Mesmos angulos dos presets do Lab (OrbitCamera.Aponta), pra que a
            // imagem de referencia e o que se ve na tela sejam a mesma coisa.
            // Os DOIS esquemas de cor: o clay serve de referencia neutra e e o
            // "antes" na comparacao.
            mosca.aparencia = Aparencia.Clay;
            mosca.AplicaAparencia();
            Tira(cam, mosca, b, -40f, 22f, "fly_clay_perspectiva");
            Tira(cam, mosca, b, 0f, 0f, "fly_clay_lateral");

            mosca.aparencia = Aparencia.Flybody;
            mosca.AplicaAparencia();
            Tira(cam, mosca, b, -40f, 22f, "fly_flybody_perspectiva");
            Tira(cam, mosca, b, 0f, 0f, "fly_flybody_lateral");

            mosca.aparencia = Aparencia.Drosophila;
            mosca.AplicaAparencia();
            Tira(cam, mosca, b, -40f, 22f, "fly_bindpose_perspectiva");
            Tira(cam, mosca, b, 0f, 0f, "fly_bindpose_lateral");
            Tira(cam, mosca, b, 0f, 89f, "fly_bindpose_topo");
            Tira(cam, mosca, b, -90f, 0f, "fly_bindpose_frente");

            Object.DestroyImmediate(camGo);
            Object.DestroyImmediate(luzGo);
            Object.DestroyImmediate(raiz);
        }

        private static void Tira(Camera cam, FlyBody mosca, Bounds b,
                                 float yaw, float pitch, string nome)
        {
            // mesmo enquadramento de OrbitCamera.Focar: a esfera que contem o
            // corpo, com 25% de folga
            float raio = b.extents.magnitude;
            float dist = 1.25f * raio / Mathf.Sin(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
            var rot = Quaternion.Euler(pitch, yaw, 0f);
            cam.transform.position = b.center + rot * new Vector3(0, 0, -dist);
            cam.transform.rotation = rot;

            var rt = new RenderTexture(Lado, Lado, 24);
            cam.targetTexture = rt;
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(Lado, Lado, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, Lado, Lado), 0, 0);
            tex.Apply();
            RenderTexture.active = null;
            cam.targetTexture = null;

            Directory.CreateDirectory(Saida);
            string caminho = Path.Combine(Saida, nome + ".png");
            File.WriteAllBytes(caminho, tex.EncodeToPNG());
            Debug.Log($"[capture] {caminho}  (yaw {yaw} pitch {pitch} dist {dist:F2} mm)");

            Object.DestroyImmediate(tex);
            rt.Release();
            Object.DestroyImmediate(rt);
        }
    }
}
