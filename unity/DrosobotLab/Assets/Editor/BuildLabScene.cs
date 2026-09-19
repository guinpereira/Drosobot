// Cria e salva a cena do Drosobot Lab sem abrir o Editor.
//
//   Unity.exe -batchmode -nographics -quit ^
//     -projectPath unity\DrosobotLab ^
//     -executeMethod Drosobot.EditorTools.BuildLabScene.Build ^
//     -logFile unity\build.log
//
// Existe porque este projeto foi escrito sem integracao MCP com a Unity: sem um
// jeito de dirigir o Editor, ou a cena nasce por linha de comando ou alguem
// monta na mao toda vez. Assim `git clone` + um comando ja da uma cena aberta.
//
// A cena e deliberadamente MINIMA -- um GameObject com LabBootstrap. Todo o
// resto (camera, luz, CNS, HUD) e montado em runtime por aquele componente, pra
// que a configuracao viva em C# revisavel no diff e nao em YAML de cena.

using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Drosobot.Lab;

namespace Drosobot.EditorTools
{
    public static class BuildLabScene
    {
        private const string Pasta = "Assets/Scenes";
        private const string Caminho = Pasta + "/DrosobotLab.unity";

        [MenuItem("Drosobot/Rebuild Lab Scene")]
        public static void Build()
        {
            var cena = EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var go = new GameObject("DrosobotLab");
            var boot = go.AddComponent<LabBootstrap>();
            boot.host = "127.0.0.1";
            boot.port = 8765;
            boot.cnsResource = "CNS/cns";
            boot.metadataResource = "CNS/neuron_metadata";

            if (!AssetDatabase.IsValidFolder(Pasta))
                AssetDatabase.CreateFolder("Assets", "Scenes");

            bool ok = EditorSceneManager.SaveScene(cena, Caminho);
            Debug.Log(ok
                ? $"[build] cena salva em {Caminho}"
                : $"[build] FALHA ao salvar {Caminho}");

            // deixa a cena como a que abre por padrao, pra bastar dar Play
            var atuais = EditorBuildSettings.scenes;
            bool ja = false;
            foreach (var s in atuais) if (s.path == Caminho) ja = true;
            if (!ja)
            {
                var nova = new EditorBuildSettingsScene[atuais.Length + 1];
                nova[0] = new EditorBuildSettingsScene(Caminho, true);
                for (int i = 0; i < atuais.Length; i++) nova[i + 1] = atuais[i];
                EditorBuildSettings.scenes = nova;
                Debug.Log("[build] cena registrada em Build Settings");
            }

            AssetDatabase.SaveAssets();
        }
    }
}
