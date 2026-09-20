// Conversao MuJoCo -> Unity. UM lugar so.
//
// Espalhar troca de eixo por varios scripts e o jeito mais confiavel de aplicar
// a rotacao duas vezes e passar dias procurando o erro. Toda conversao de
// coordenada do Drosobot passa por aqui.
//
// Ver docs/UNITY_BODY_COORDINATES.md pra derivacao.

using UnityEngine;

namespace Drosobot.Lab
{
    /// <summary>
    /// Em que espaco a telemetria entrega os transforms dos segmentos.
    ///
    /// Isto existe pra tornar IMPOSSIVEL o bug que ja aconteceu: mover a raiz
    /// da mosca pela posicao global E aplicar pose de mundo em cada segmento,
    /// somando o deslocamento duas vezes.
    /// </summary>
    public enum EspacoPose
    {
        /// <summary>
        /// Cada segmento chega com transform de MUNDO. A raiz visual fica
        /// neutra (identidade) e NAO pode ser movida por ninguem.
        /// </summary>
        Mundo,

        /// <summary>
        /// Cada segmento chega relativo ao pai. A raiz recebe o transform
        /// global e os filhos os locais.
        /// </summary>
        Local,
    }

    public static class MujocoFrame
    {
        // ---------------------------------------------------------- eixos
        //
        //   MuJoCo   destro, Z pra cima, milimetros
        //   Unity    canhoto, Y pra cima
        //
        // Trocar Y e Z leva um sistema no outro e ja inverte a quiralidade,
        // que e o que a mudanca de destro pra canhoto exige.

        /// <summary>Posicao MuJoCo (x, y, z) -> Unity. Milimetros nos dois.</summary>
        public static Vector3 Pos(float x, float y, float z) => new Vector3(x, z, y);

        public static Vector3 Pos(float[] p) => Pos(p[0], p[1], p[2]);

        /// <summary>
        /// Quaternio MuJoCo (w, x, y, z) -> Unity (x, y, z, w).
        ///
        /// A troca de eixo e uma reflexao (determinante -1), entao alem de
        /// permutar a parte vetorial e preciso negar: (x, y, z) -> (-x, -z, -y),
        /// com w intacto. Sem a negacao a mosca gira ao contrario, e isso so
        /// aparece nos segmentos muito rotacionados -- as pernas.
        /// </summary>
        public static Quaternion Quat(float w, float x, float y, float z)
            => new Quaternion(-x, -z, -y, w);

        public static Quaternion Quat(float[] q) => Quat(q[0], q[1], q[2], q[3]);

        /// <summary>
        /// Guarda contra o transform duplicado.
        ///
        /// Com pose de MUNDO, a raiz visual tem que ficar na identidade. Se
        /// alguem a mover, cada segmento sai deslocado pelo valor da raiz e a
        /// mosca aparece longe de onde esta. Ja aconteceu; agora avisa.
        /// </summary>
        public static bool RaizNeutra(Transform raiz, EspacoPose espaco,
                                      out string problema)
        {
            problema = null;
            if (espaco != EspacoPose.Mundo || raiz == null) return true;
            bool ok = raiz.position.sqrMagnitude < 1e-6f
                      && Quaternion.Angle(raiz.rotation, Quaternion.identity) < 0.01f;
            if (!ok)
            {
                problema = $"pose em MUNDO exige raiz neutra, mas a raiz esta em "
                           + $"{raiz.position} / {raiz.rotation.eulerAngles}. "
                           + "O deslocamento seria somado duas vezes.";
            }
            return ok;
        }
    }
}
