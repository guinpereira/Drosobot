// Cor da mosca na tela.
//
// ## Isto e ASSUMPTION, nao dado
//
// O modelo do NeuroMechFly nao traz cor: os 69 geoms tem `rgba` (0,5 0,5 0,5 1)
// -- cinza uniforme. Tudo que este arquivo faz e aparencia inventada por nos
// pra que a mosca pareca uma mosca.
//
// Por isso existem DOIS esquemas e o Lab diz qual esta ligado. `Modelo` mostra
// o cinza que o modelo de fato carrega; `Realista` mostra o nosso. Nenhuma cor
// aqui codifica grandeza nenhuma -- nao ha contato, ativacao nem forca sendo
// pintada. Se um dia houver, tem que ser em outro canal visual e rotulado como
// tal, senao vira grafico disfarcado de foto.
//
// A geometria continua sendo a do modelo compilado que a fisica roda. So o que
// reflete luz e escolha nossa.

using UnityEngine;

namespace Drosobot.Lab
{
    public enum Aparencia
    {
        /// <summary>O cinza que o modelo carrega de verdade.</summary>
        Modelo,
        /// <summary>Cores inventadas por nos. Sem significado quantitativo.</summary>
        Realista,
    }

    public static class FlyAppearance
    {
        // Cores de referencia: Drosophila melanogaster selvagem. Cuticula
        // marrom-ambar, olhos vermelhos, asas quase transparentes com leve
        // iridescencia. Escolhidas a olho, nao medidas.
        private static readonly Color Cuticula = new Color(0.34f, 0.22f, 0.11f);
        private static readonly Color Torax = new Color(0.42f, 0.29f, 0.16f);
        private static readonly Color Abdomen = new Color(0.26f, 0.17f, 0.09f);
        private static readonly Color Olho = new Color(0.62f, 0.09f, 0.06f);
        private static readonly Color Asa = new Color(0.80f, 0.84f, 0.88f, 0.22f);
        private static readonly Color Haltere = new Color(0.72f, 0.62f, 0.30f);
        private static readonly Color Perna = new Color(0.45f, 0.33f, 0.18f);
        private static readonly Color Tarso = new Color(0.30f, 0.22f, 0.12f);
        private static readonly Color Neutro = new Color(0.5f, 0.5f, 0.5f);

        /// <summary>
        /// Em que categoria visual cai um segmento. Publico pra que quem pinta
        /// possa reusar UM material por categoria em vez de um por segmento:
        /// sao 68 segmentos e oito categorias.
        /// </summary>
        public static string Categoria(string segmento)
        {
            string s = segmento.ToLowerInvariant();
            if (s.Contains("eye")) return "olho";
            if (s.Contains("wing")) return "asa";
            if (s.Contains("haltere")) return "haltere";
            if (s.Contains("tarsus") || s.Contains("arista")) return "tarso";
            if (s.Contains("coxa") || s.Contains("femur") || s.Contains("tibia"))
                return "perna";
            if (s.Contains("abdomen")) return "abdomen";
            if (s.Contains("thorax")) return "torax";
            return "cuticula";
        }

        /// <summary>Material pra um segmento, pelo nome dele.</summary>
        public static Material Material(string segmento, Aparencia modo)
        {
            var sh = Shader.Find("Standard") ?? Shader.Find("Diffuse");
            var m = new Material(sh);

            if (modo == Aparencia.Modelo)
            {
                m.color = Neutro;
                m.SetFloat("_Glossiness", 0.1f);
                return m;
            }

            string s = Categoria(segmento);
            if (s == "olho")
            {
                // olho composto: liso e brilhante, e o que mais denuncia uma
                // mosca de verdade numa imagem
                m.color = Olho;
                m.SetFloat("_Glossiness", 0.75f);
                m.SetFloat("_Metallic", 0.10f);
            }
            else if (s == "asa")
            {
                m.color = Asa;
                m.SetFloat("_Glossiness", 0.85f);
                Transparente(m);
            }
            else if (s == "haltere")
            {
                m.color = Haltere;
                m.SetFloat("_Glossiness", 0.35f);
            }
            else if (s == "tarso")
            {
                m.color = Tarso;
                m.SetFloat("_Glossiness", 0.25f);
            }
            else if (s == "perna")
            {
                m.color = Perna;
                m.SetFloat("_Glossiness", 0.32f);
            }
            else if (s == "abdomen")
            {
                m.color = Abdomen;
                m.SetFloat("_Glossiness", 0.28f);
            }
            else if (s == "torax")
            {
                m.color = Torax;
                m.SetFloat("_Glossiness", 0.38f);
            }
            else
            {
                m.color = Cuticula;      // cabeca, rostro, antenas, o resto
                m.SetFloat("_Glossiness", 0.30f);
            }
            return m;
        }

        // Transparencia no Standard do Built-in nao sai so mudando _Mode:
        // precisa dos blend modes, de desligar o ZWrite E das keywords.
        private static void Transparente(Material m)
        {
            m.SetFloat("_Mode", 3f);
            m.SetFloat("_Surface", 1f);
            m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetInt("_ZWrite", 0);
            m.DisableKeyword("_ALPHATEST_ON");
            m.EnableKeyword("_ALPHABLEND_ON");
            m.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        }
    }
}
