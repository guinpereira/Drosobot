// Cor e material da mosca na tela.
//
// ## O modelo nao traz cor
//
// O modelo que a FISICA roda sai do compilador sem material nenhum: `nmat = 1`
// no modelo inteiro, e esse material e o `grid` do chao da arena. Os 69 geoms
// tem `matid = -1` e `rgba` (0,5 0,5 0,5 1). Verificado, nao suposto.
//
// Entao toda cor daqui e escolha de apresentacao, e o painel CORPO diz qual
// esquema esta ligado e de onde ele veio. Sao tres:
//
//   Clay        o cinza 0,5 que o modelo de fato carrega. DATA.
//   Flybody     paleta do modelo `flybody` que vem no mesmo pacote de assets
//               do FlyGym (assets/model/flybody/fruitfly.xml, Vaxenburg et
//               al.), autorada com material por regiao. Os VALORES vem de la;
//               a atribuicao aos nossos segmentos e nossa.
//   Drosophila  paleta escolhida a olho pra leitura no fundo escuro do Lab.
//               Nao tem procedencia de medicao nenhuma.
//
// Os dois esquemas de cor ficam lado a lado de proposito. O Flybody e o unico
// com origem rastreavel, e apagar ele pra deixar so o que "parece melhor"
// jogaria fora a unica referencia externa que existe.
//
// Nenhuma cor aqui codifica grandeza. Nao ha contato, ativacao nem forca sendo
// pintada no corpo. Se um dia houver, tem que ser em outro canal visual e
// rotulado como tal, senao vira grafico disfarcado de foto.
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
        /// <summary>Valores da paleta do flybody, atribuidos por anatomia.</summary>
        Flybody,
        /// <summary>Paleta escolhida a olho pra leitura no fundo escuro.</summary>
        Drosophila,
    }

    public static class FlyAppearance
    {
        /// <summary>
        /// As cores base de um esquema. So as BASE: banda do abdomen e
        /// escurecimento distal das pernas sao derivados delas, nao listados um
        /// a um -- assim os dois esquemas ganham a mesma estrutura anatomica e
        /// so a cor muda.
        /// </summary>
        private struct Paleta
        {
            public Color cuticula;   // cabeca, rostro, o que nao tem regra propria
            public Color torax;
            public Color abdomen;    // BASE do abdomen; a banda sai daqui
            public Color perna;      // femur/tibia/coxa; o tarso escurece
            public Color olho;
            public Color asa;        // com alfa
            public Color haltere;
        }

        // Paleta do flybody: flygym/assets/model/flybody/fruitfly.xml. Os nomes
        // de la ficam no comentario pra dar pra conferir linha a linha.
        private static readonly Paleta DoFlybody = new Paleta
        {
            cuticula = Rgb(0.674f, 0.350f, 0.143f),                  // body
            torax = Color.Lerp(Rgb(0.674f, 0.350f, 0.143f),
                               Rgb(0.799f, 0.610f, 0.386f), 0.25f),  // body -> lower
            abdomen = Rgb(0.799f, 0.610f, 0.386f),                   // lower
            perna = Color.Lerp(Rgb(0.674f, 0.350f, 0.143f),
                               Rgb(0.202f, 0.0782f, 0.0262f), 0.25f), // body -> brown
            // o `red` de la e (0,8 0,028 0,0015): saturado demais, fica neon no
            // fundo escuro. Escurecido 45% pro `brown` da mesma paleta.
            olho = Color.Lerp(Rgb(0.800f, 0.0279f, 0.00154f),
                              Rgb(0.202f, 0.0782f, 0.0262f), 0.45f),
            asa = new Color(0.539f, 0.686f, 0.800f, 0.30f),          // membrane
            haltere = Rgb(0.799f, 0.610f, 0.386f),                   // lower
        };

        // Paleta Drosophila: tons claros de cuticula, olho tijolo, asa quase
        // incolor. Mais clara que a do flybody, que no fundo escuro do Lab
        // tende a fechar. Escolhida a olho.
        private static readonly Paleta DeDrosophila = new Paleta
        {
            cuticula = Rgb(0.78f, 0.56f, 0.31f),   // #C68E4F
            torax = Rgb(0.78f, 0.56f, 0.31f),      // #C68E4F
            abdomen = Rgb(0.87f, 0.70f, 0.44f),    // #DEB887
            perna = Rgb(0.93f, 0.86f, 0.51f),      // #EEDC82
            olho = Rgb(0.65f, 0.11f, 0.11f),       // #A61C1C
            asa = new Color(0.92f, 0.94f, 0.95f, 0.30f),  // #EAF0F1
            haltere = Rgb(0.93f, 0.86f, 0.51f),
        };

        // Para onde as partes escuras puxam: cerdas, tarsos, banda posterior do
        // abdomen. Um so, pros dois esquemas, pra que a estrutura seja a mesma.
        private static readonly Color Escuro = Rgb(0.10f, 0.05f, 0.03f);
        private static readonly Color Neutro = new Color(0.5f, 0.5f, 0.5f);

        private static Color Rgb(float r, float g, float b) => new Color(r, g, b, 1f);

        private static Paleta Escolhe(Aparencia modo)
            => modo == Aparencia.Flybody ? DoFlybody : DeDrosophila;

        /// <summary>De onde veio a cor deste esquema, pro painel dizer.</summary>
        public static string Procedencia(Aparencia modo)
        {
            switch (modo)
            {
                case Aparencia.Clay:
                    return "cinza 0,5 -- a unica cor que o modelo carrega";
                case Aparencia.Flybody:
                    return "valores do flybody (assets do FlyGym); atribuicao nossa";
                default:
                    return "escolhida a olho; sem procedencia de medicao";
            }
        }

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

            var p = Escolhe(modo);
            string cat = Categoria(segmento);

            if (cat == "olho")
            {
                // olho composto: liso, mais brilhante que a cuticula. E o que
                // mais denuncia uma mosca de verdade numa imagem.
                Ajusta(m, p.olho, 0.78f);
            }
            else if (cat == "asa")
            {
                // membrana: quase transparente e com especular alto, que e o
                // que da a leitura de "fina" sem nervura nenhuma
                Ajusta(m, p.asa, 0.88f);
                Transparente(m);
            }
            else if (cat == "haltere")
            {
                Ajusta(m, p.haltere, 0.35f);
            }
            else if (cat == "cerda")
            {
                Ajusta(m, Escuro, 0.10f);    // fosca de proposito
            }
            else if (cat == "tarso")
            {
                // extremidade mais escura: e assim na mosca e ajuda a leitura
                // de onde a perna termina contra o chao escuro
                Ajusta(m, Color.Lerp(p.perna, Escuro, 0.35f), 0.30f);
            }
            else if (cat == "perna")
            {
                Ajusta(m, p.perna, 0.22f);
            }
            else if (cat.StartsWith("abdomen"))
            {
                // Banda do abdomen. Cada terguito escurece em direcao ao
                // posterior. Os limites das faixas sao as juntas reais entre os
                // corpos do modelo, entao a faixa sai da geometria.
                int i = cat.Length > 7 ? cat[7] - '0' : 2;
                float t = Mathf.InverseLerp(2f, 6f, i);
                Ajusta(m, Color.Lerp(p.abdomen, Escuro, 0.42f * t), 0.30f);
            }
            else if (cat == "torax")
            {
                Ajusta(m, p.torax, 0.38f);
            }
            else
            {
                Ajusta(m, p.cuticula, 0.30f);   // cabeca, rostro, o resto
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
