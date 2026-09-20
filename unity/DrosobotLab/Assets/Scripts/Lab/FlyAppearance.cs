// Cor e material da mosca na tela.
//
// ## De onde vem a cor, e o que e nosso
//
// O modelo que a FISICA roda nao tem cor: os 69 geoms do NeuroMechFly saem do
// compilador com `rgba` (0,5 0,5 0,5 1) e `matid = -1`. Verificado, nao
// suposto -- `nmat = 1` no modelo inteiro, e esse material e o `grid` do chao
// da arena.
//
// Os VALORES usados aqui nao sao inventados por nos: sao a paleta do modelo
// `flybody` que vem no mesmo pacote de assets do FlyGym
// (`flygym/assets/model/flybody/fruitfly.xml`, Vaxenburg et al.), que e um
// modelo de Drosophila autorado com material por regiao.
//
// A ATRIBUICAO e nossa. O `fruitfly.xml` divide o corpo em outros geoms
// (`thorax_black`, `head_red`, `wing_left_membrane`...), 28 com material, e nao
// da pra casar um a um com os nossos 69 segmentos. Entao mapeamos por anatomia,
// e isso e ASSUMPTION -- fica dito no painel CORPO.
//
// Nenhuma cor aqui codifica grandeza. Nao ha contato, ativacao nem forca sendo
// pintada no corpo. Se um dia houver, tem que ser em outro canal visual e
// rotulado como tal, senao vira grafico disfarcado de foto.
//
// A geometria continua sendo a do modelo compilado que a fisica roda. So o que
// reflete luz e escolha nossa.
//
// ## O que NAO da pra fazer com este modelo
//
// Nervuras de asa. No `fruitfly.xml` elas sao um geom separado
// (`wing_left_brown`) por cima da membrana. No modelo do NeuroMechFly a asa e
// UMA malha so, sem nervura na geometria e sem UV. Desenhar nervura aqui seria
// inventar anatomia numa imagem que as pessoas vao ler como o modelo. Fica sem.

using UnityEngine;

namespace Drosobot.Lab
{
    public enum Aparencia
    {
        /// <summary>
        /// O cinza 0,5 que o modelo carrega de verdade. Serve de neutro pra
        /// depuracao: sem cor competindo, forma e pose ficam mais legiveis.
        /// </summary>
        Clay,
        /// <summary>Paleta do flybody, atribuida por anatomia.</summary>
        Realista,
    }

    public static class FlyAppearance
    {
        // ------------------------------------------------- paleta do flybody
        //
        // Copiada de flygym/assets/model/flybody/fruitfly.xml. Os nomes sao os
        // de la, de proposito, pra dar pra conferir linha a linha.
        private static readonly Color Body = Rgb(0.674f, 0.350f, 0.143f);
        private static readonly Color Lower = Rgb(0.799f, 0.610f, 0.386f);
        private static readonly Color Brown = Rgb(0.202f, 0.0782f, 0.0262f);
        private static readonly Color Bristle = Rgb(0.06f, 0.04f, 0.03f);
        private static readonly Color Membrana = new Color(0.539f, 0.686f, 0.800f, 0.30f);
        private static readonly Color Neutro = new Color(0.5f, 0.5f, 0.5f);

        // O `red` do flybody e (0,8 0,028 0,0015): saturado demais, fica neon
        // no fundo escuro do Lab. Escurecido pra vinho, 45% na direcao do
        // `brown` da mesma paleta. Este ajuste e nosso.
        private static readonly Color Olho =
            Color.Lerp(Rgb(0.800f, 0.0279f, 0.00154f), Rgb(0.202f, 0.0782f, 0.0262f), 0.45f);

        private static Color Rgb(float r, float g, float b) => new Color(r, g, b, 1f);

        /// <summary>
        /// Em que categoria visual cai um segmento. Publico pra que quem pinta
        /// reuse UM material por categoria em vez de um por segmento.
        /// </summary>
        public static string Categoria(string segmento)
        {
            string s = segmento.ToLowerInvariant();
            if (s.Contains("eye")) return "olho";
            if (s.Contains("wing")) return "asa";
            if (s.Contains("haltere")) return "haltere";
            // arista, cerdas e antenas: finas, escuras e foscas
            if (s.Contains("arista") || s.Contains("pedicel") || s.Contains("funiculus"))
                return "cerda";
            if (s.Contains("tarsus")) return "tarso";
            if (s.Contains("coxa") || s.Contains("femur") || s.Contains("tibia"))
                return "perna";
            // o abdomen tem banda: cada segmento e um corpo separado no modelo,
            // entao a faixa sai da GEOMETRIA, nao de textura pintada
            if (s.Contains("abdomen")) return "abdomen" + Segmento(s);
            if (s.Contains("thorax")) return "torax";
            return "cuticula";
        }

        // c_abdomen12 -> 2, c_abdomen3 -> 3 ... o ultimo digito basta e e o
        // indice antero-posterior do terguito
        private static string Segmento(string s)
        {
            for (int i = s.Length - 1; i >= 0; i--)
                if (char.IsDigit(s[i])) return s[i].ToString();
            return "2";
        }

        /// <summary>Material pra um segmento, pelo nome dele.</summary>
        public static Material Material(string segmento, Aparencia modo)
        {
            var sh = Shader.Find("Standard") ?? Shader.Find("Diffuse");
            var m = new Material(sh);
            m.SetFloat("_Metallic", 0f);      // cuticula nao e metal em lugar nenhum

            if (modo == Aparencia.Clay)
            {
                Ajusta(m, Neutro, 0.12f);
                return m;
            }

            string cat = Categoria(segmento);

            if (cat == "olho")
            {
                // olho composto: liso, um pouco mais brilhante que a cuticula.
                // E o que mais denuncia uma mosca de verdade numa imagem.
                Ajusta(m, Olho, 0.55f);
            }
            else if (cat == "asa")
            {
                // membrana: quase transparente e com brilho especular alto, que
                // e o que da a leitura de "fina" sem nervura nenhuma
                Ajusta(m, Membrana, 0.80f);
                Transparente(m);
            }
            else if (cat == "haltere")
            {
                Ajusta(m, Lower, 0.35f);
            }
            else if (cat == "cerda")
            {
                Ajusta(m, Bristle, 0.10f);   // fosca de proposito
            }
            else if (cat == "tarso")
            {
                // extremidade mais escura: e assim na mosca e ajuda a leitura
                // de onde a perna termina contra o chao escuro
                Ajusta(m, Color.Lerp(Body, Brown, 0.55f), 0.30f);
            }
            else if (cat == "perna")
            {
                Ajusta(m, Color.Lerp(Body, Brown, 0.25f), 0.28f);
            }
            else if (cat.StartsWith("abdomen"))
            {
                // Banda do abdomen. Cada terguito escurece em direcao ao
                // posterior, do `lower` ao `brown` da paleta. Os limites das
                // faixas sao as juntas reais entre os corpos do modelo.
                int i = cat.Length > 7 ? cat[7] - '0' : 2;
                float t = Mathf.InverseLerp(2f, 6f, i);
                Ajusta(m, Color.Lerp(Lower, Color.Lerp(Body, Brown, 0.5f), t), 0.30f);
            }
            else if (cat == "torax")
            {
                // torax um tom mais quente que o resto
                Ajusta(m, Color.Lerp(Body, Lower, 0.25f), 0.38f);
            }
            else
            {
                Ajusta(m, Body, 0.30f);      // cabeca, rostro, o resto
            }
            return m;
        }

        private static void Ajusta(Material m, Color cor, float lisura)
        {
            m.color = cor;
            // `_Glossiness` no Standard do Built-in, `_Smoothness` no URP: os
            // dois sao setados porque o projeto ja rodou nos dois pipelines e
            // setar a propriedade que nao existe e inofensivo.
            m.SetFloat("_Glossiness", lisura);
            m.SetFloat("_Smoothness", lisura);
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

        /// <summary>
        /// Tres luzes pra ler um corpo pequeno e escuro contra fundo escuro:
        /// principal quente, preenchimento frio e fraco pra sombra nao fechar,
        /// e contraluz que separa a silhueta do fundo. Direcionais -- nao ha
        /// custo por pixel de sombra, o corpo nem projeta sombra.
        /// </summary>
        public static void Ilumina(Transform pai)
        {
            // Intensidades baixas de proposito. Com key 1,15 + fill 0,42 +
            // rim 0,70 e ambiente trilight a cuticula estourava: a mosca lia
            // como pessego claro em vez do ambar da paleta. A soma aqui fica
            // perto de 1,2 pra que a cor na tela seja a cor do material.
            Luz(pai, "Key", new Color(1.00f, 0.96f, 0.90f), 0.80f, 38f, 40f);
            Luz(pai, "Fill", new Color(0.62f, 0.72f, 0.90f), 0.22f, 12f, -120f);
            Luz(pai, "Rim", new Color(0.80f, 0.88f, 1.00f), 0.45f, -8f, 190f);
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.13f, 0.14f, 0.17f);
            RenderSettings.ambientEquatorColor = new Color(0.09f, 0.09f, 0.10f);
            RenderSettings.ambientGroundColor = new Color(0.05f, 0.04f, 0.04f);
        }

        private static void Luz(Transform pai, string nome, Color cor,
                                float intensidade, float pitch, float yaw)
        {
            var go = new GameObject(nome);
            if (pai != null) go.transform.SetParent(pai, false);
            var l = go.AddComponent<Light>();
            l.type = LightType.Directional;
            l.color = cor;
            l.intensity = intensidade;
            l.shadows = LightShadows.None;
            go.transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
        }
    }
}
