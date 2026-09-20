// Paineis do Drosobot Lab: graficos de timeline, retina e inspector.
//
// Tudo desenhado em texturas pequenas atualizadas por quadro. E barato (a retina
// sao 1442 pixels) e nao depende de autorar prefabs de UI -- o projeto e montado
// por codigo, entao a UI tambem.
//
// Nenhum destes paineis calcula nada de ciencia. Todos mostram valores que
// chegaram prontos pela telemetria.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace Drosobot.Lab
{
    /// <summary>Janela rolante de um sinal, desenhada numa textura.</summary>
    public class Sparkline
    {
        public readonly string nome;
        public readonly Color cor;
        public readonly Provenance procedencia;
        private readonly float[] _buf;
        private int _cursor;
        private int _preenchidos;
        private Texture2D _tex;
        private Color32[] _pixels;
        private readonly int _w, _h;

        public float Ultimo { get; private set; }

        public Sparkline(string nome, Color cor, Provenance proc,
                         int amostras = 240, int w = 240, int h = 42)
        {
            this.nome = nome; this.cor = cor; this.procedencia = proc;
            _buf = new float[amostras];
            _w = w; _h = h;
        }

        public void Push(float v)
        {
            Ultimo = v;
            _buf[_cursor] = v;
            _cursor = (_cursor + 1) % _buf.Length;
            if (_preenchidos < _buf.Length) _preenchidos++;
        }

        public Texture2D Desenhar()
        {
            if (_tex == null)
            {
                _tex = new Texture2D(_w, _h, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point };
                _pixels = new Color32[_w * _h];
            }

            var fundo = new Color32(18, 20, 25, 235);
            var grade = new Color32(40, 44, 52, 255);
            for (int i = 0; i < _pixels.Length; i++) _pixels[i] = fundo;
            for (int x = 0; x < _w; x++) _pixels[(_h / 2) * _w + x] = grade;

            // escala automatica: sinais aqui vao de fracao de Hz a centenas
            float max = 1e-6f;
            for (int i = 0; i < _preenchidos; i++) max = Mathf.Max(max, Mathf.Abs(_buf[i]));

            var c32 = (Color32)cor;
            for (int x = 0; x < _w; x++)
            {
                int idx = _cursor - _preenchidos + (int)((float)x / _w * _preenchidos);
                if (idx < 0) idx += _buf.Length;
                idx = ((idx % _buf.Length) + _buf.Length) % _buf.Length;
                float v = _buf[idx] / max;
                int y = Mathf.Clamp(Mathf.RoundToInt((v * 0.5f + 0.5f) * (_h - 1)), 0, _h - 1);
                _pixels[y * _w + x] = c32;
                // preenche ate a linha de base pra leitura ficar mais facil
                int baseY = _h / 2;
                int passo = y > baseY ? -1 : 1;
                for (int yy = y; yy != baseY; yy += passo)
                    _pixels[yy * _w + x] = new Color32(c32.r, c32.g, c32.b, 70);
            }

            _tex.SetPixels32(_pixels);
            _tex.Apply(false);
            return _tex;
        }

        public string Rotulo => $"{nome}  {Ultimo:F1}";
    }

    /// <summary>
    /// Os 721 omatideos de um olho num mapa 2D.
    ///
    /// A retina da mosca e uma superficie curva e a projecao em grade NAO
    /// preserva a geometria do olho -- e uma leitura de intensidade por
    /// omatideo, na ordem em que o FlyGym entrega, nao um mapa retinotopico.
    /// Dito aqui pra ninguem ler posicao espacial onde nao ha.
    /// </summary>
    public class RetinaView
    {
        private const int N = 721;
        private const int Colunas = 31;          // 31 x 24 = 744 >= 721
        private const int Linhas = 24;
        // 3 e nao 5: com 5 os dois olhos somavam 320 px e empurravam o
        // painel de sinais pra fora da tela em 1080p. A grade continua
        // legivel -- e mapa de intensidade, nao imagem pra ler detalhe.
        private const int Escala = 3;            // pixels por omatideo

        private Texture2D _tex;
        private Color32[] _pixels;
        private readonly int _w = Colunas * Escala, _h = Linhas * Escala;

        public Texture2D Desenhar(IList<float> valores)
        {
            if (_tex == null)
            {
                _tex = new Texture2D(_w, _h, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point };
                _pixels = new Color32[_w * _h];
            }
            var vazio = new Color32(14, 16, 20, 235);
            for (int i = 0; i < _pixels.Length; i++) _pixels[i] = vazio;

            int n = valores == null ? 0 : Mathf.Min(N, valores.Count);
            for (int k = 0; k < n; k++)
            {
                float v = Mathf.Clamp01(valores[k]);
                var c = new Color32((byte)(v * 235), (byte)(v * 240), (byte)(v * 255), 255);
                int cx = (k % Colunas) * Escala, cy = (k / Colunas) * Escala;
                for (int dy = 0; dy < Escala - 1; dy++)
                    for (int dx = 0; dx < Escala - 1; dx++)
                    {
                        int px = cx + dx, py = _h - 1 - (cy + dy);
                        if (px < _w && py >= 0) _pixels[py * _w + px] = c;
                    }
            }
            _tex.SetPixels32(_pixels);
            _tex.Apply(false);
            return _tex;
        }

        public int Largura => _w;
        public int Altura => _h;
    }

    /// <summary>Etiqueta DATA / MODEL / ASSUMPTION ao lado de um valor.</summary>
    public static class Badge
    {
        /// <summary>
        /// Desenha uma linha `chave | etiqueta | valor` dentro de um retangulo.
        ///
        /// Recebe o retangulo pronto, em vez de usar GUILayout, porque quem
        /// posiciona os paineis agora e LabLayout.cs -- ele mede a altura de cada
        /// bloco antes de desenhar, e GUILayout so sabe a altura depois.
        /// </summary>
        public static void Desenha(Rect r, string chave, string valor, Provenance p,
                                   GUIStyle estilo, float larguraChave = 116f)
        {
            float larguraEtiqueta = 78f;
            GUI.Label(new Rect(r.x, r.y, larguraChave, r.height), chave, estilo);
            var antes = GUI.color;
            GUI.color = ProvenanceUtil.Color(p);
            GUI.Label(new Rect(r.x + larguraChave, r.y, larguraEtiqueta, r.height),
                      ProvenanceUtil.Label(p), estilo);
            GUI.color = antes;
            float x = r.x + larguraChave + larguraEtiqueta;
            GUI.Label(new Rect(x, r.y, Mathf.Max(0f, r.xMax - x), r.height), valor, estilo);
        }
    }
}
