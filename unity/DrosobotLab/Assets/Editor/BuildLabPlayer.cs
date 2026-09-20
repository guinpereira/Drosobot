// Constroi o player standalone do Drosobot Lab por linha de comando.
//
//   Unity.exe -batchmode -quit -projectPath unity\DrosobotLab ^
//     -executeMethod Drosobot.EditorTools.BuildLabPlayer.Build ^
//     -logFile unity\buildplayer.log
//
// Existe pra validar a sessao AO VIVO sem ninguem clicando em Play. O modo
// Play em batchmode e instavel -- o Editor nao garante quadro nenhum sem foco,
// e ja deu `Time.frameCount` travado em 2. Um player de verdade renderiza de
// verdade, e e ele quem tira as fotos (ver AutoCapture.cs).

using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Drosobot.EditorTools
{
    public static class BuildLabPlayer
    {
        private const string Cena = "Assets/Scenes/DrosobotLab.unity";
        private static readonly string Destino =
            Path.GetFullPath(Path.Combine(Application.dataPath, "../Build/DrosobotLab.exe"));

        [MenuItem("Drosobot/Build Player")]
        public static void Build()
        {
            if (!File.Exists(Cena))
            {
                Debug.Log("[build] cena ausente; montando antes");
                BuildLabScene.Build();
            }

            Directory.CreateDirectory(Path.GetDirectoryName(Destino));
            // Janela, nao tela cheia: o player roda ao lado da simulacao e
            // tela cheia atrapalharia qualquer uso interativo depois.
            PlayerSettings.defaultIsNativeResolution = false;
            PlayerSettings.defaultScreenWidth = 1920;
            PlayerSettings.defaultScreenHeight = 1080;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.runInBackground = true;
            PlayerSettings.resizableWindow = true;

            var opcoes = new BuildPlayerOptions
            {
                scenes = new[] { Cena },
                locationPathName = Destino,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.Development,
            };
            var relatorio = BuildPipeline.BuildPlayer(opcoes);
            var r = relatorio.summary;
            Debug.Log($"[build] {r.result}  {r.totalSize / 1048576} MiB  "
                      + $"{r.totalTime.TotalSeconds:F0} s  -> {Destino}");
            if (r.result != BuildResult.Succeeded)
                EditorApplication.Exit(1);
        }
    }
}
