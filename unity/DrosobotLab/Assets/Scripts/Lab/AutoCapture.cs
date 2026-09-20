// Captura a sessao AO VIVO, sem ninguem olhando.
//
// Existe porque "a Unity compila" e "a telemetria entrega os campos" nao
// respondem a pergunta que importa: **a tela esta certa?** Os dois bugs mais
// caros desta rodada -- a malha em eixos do MuJoCo e a bind pose com as juntas
// em zero -- passavam em toda verificacao numerica e so apareceram na imagem.
//
// Ligado por linha de comando, pra que o player normal nunca faca isto:
//
//     DrosobotLab.exe -autocapture 40 -capturedir C:\...\docs\images\live
//
// Espera a telemetria chegar, deixa a simulacao rodar, e tira uma sequencia de
// fotos em tempos fixos. No fim, fecha sozinho.

using System.Collections;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace Drosobot.Lab
{
    public class AutoCapture : MonoBehaviour
    {
        [Tooltip("Segundos de sessao antes de encerrar. 0 = desligado.")]
        public float duracao = 0f;
        public string pasta = "";

        private LabBootstrap _lab;

        // momentos de captura, em segundos desde o Play, e o que cada um
        // deveria mostrar
        private static readonly (float t, string nome)[] Roteiro = {
            (6f,  "01_conectando"),
            (14f, "02_rodando"),
            (20f, "03_vistalateral"),
            (26f, "04_topo"),
            (32f, "05_eixos"),
            (38f, "06_clay"),
        };

        void Start()
        {
            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-autocapture" && i + 1 < args.Length)
                    float.TryParse(args[i + 1], NumberStyles.Float,
                                   CultureInfo.InvariantCulture, out duracao);
                if (args[i] == "-capturedir" && i + 1 < args.Length)
                    pasta = args[i + 1];
            }
            if (duracao <= 0f) { enabled = false; return; }
            if (string.IsNullOrEmpty(pasta))
                pasta = Path.Combine(Application.dataPath, "..", "capturas");
            Directory.CreateDirectory(pasta);
            _lab = FindObjectOfType<LabBootstrap>();
            StartCoroutine(Roda());
        }

        private IEnumerator Roda()
        {
            Debug.Log($"[capture] sessao de {duracao:F0} s -> {pasta}");
            float t0 = Time.realtimeSinceStartup;
            int proximo = 0;

            while (Time.realtimeSinceStartup - t0 < duracao)
            {
                float t = Time.realtimeSinceStartup - t0;
                if (proximo < Roteiro.Length && t >= Roteiro[proximo].t)
                {
                    // Antes de cada foto, mexe no Lab pelo mesmo caminho que um
                    // humano usaria -- assim a imagem prova o controle, nao so
                    // o render.
                    Prepara(Roteiro[proximo].nome);
                    // um quadro pra o estado novo aparecer antes da foto
                    yield return new WaitForEndOfFrame();
                    yield return new WaitForEndOfFrame();
                    Tira(Roteiro[proximo].nome);
                    proximo++;
                }
                yield return null;
            }

            Debug.Log("[capture] fim da sessao");
            Application.Quit();
        }

        private void Prepara(string etapa)
        {
            if (_lab == null) return;
            switch (etapa)
            {
                case "03_vistalateral": _lab.CapturaVista(1); break;
                case "04_topo": _lab.CapturaVista(2); break;
                case "05_eixos": _lab.CapturaVista(0); _lab.CapturaModo(2); break;
                case "06_clay": _lab.CapturaModo(0); _lab.CapturaPaleta(0); break;
            }
        }

        private void Tira(string nome)
        {
            string caminho = Path.Combine(pasta, $"live_{nome}.png");
            ScreenCapture.CaptureScreenshot(caminho);
            Debug.Log($"[capture] {caminho}");
        }
    }
}
