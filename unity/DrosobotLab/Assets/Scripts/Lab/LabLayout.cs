// Layout dos paineis do HUD: posicao e altura CALCULADAS, nunca chutadas.
//
// ## Por que isto existe
//
// Os paineis eram desenhados em coordenadas escritas a mao:
//
//     GUI.Box(new Rect(10, 10, 360, 330), ...)        // painel principal
//     GUI.Box(new Rect(290, 10, 330, ...), ...)       // atividade de populacao
//     GUI.Box(new Rect(Screen.width - 250, 10, 240, 92), ...)   // procedencia
//
// Dois defeitos saem direto dai, e os dois apareceram na tela:
//
//   SOBREPOSICAO -- o painel principal ocupa x de 10 a 370 e o de populacao
//   comecava em x=290. O numero 290 so estava certo pra uma largura antiga do
//   painel da esquerda. Coordenada escrita a mao nao acompanha o vizinho.
//
//   TEXTO CORTADO -- altura fixa (92) com conteudo que quebra linha. "ASSUMPTION
//   escolha nossa de modelagem" nao cabe na largura, o GUIStyle quebra em duas
//   linhas, e a quarta linha cai fora da caixa. O texto estava sendo desenhado;
//   a caixa e que era curta demais pra ele.
//
// ## Como resolve
//
// Um painel declara o CONTEUDO. A altura sai de GUIStyle.CalcHeight sobre a
// largura real da coluna, entao quebra de linha vira altura em vez de corte. Uma
// coluna empilha paineis com um vao fixo e cada `x` vem da coluna anterior, nao
// de um numero digitado -- duas caixas nao tem como se sobrepor porque nenhuma
// escolhe a propria posicao.
//
// ## Sobre uGUI
//
// Isto e o equivalente, em IMGUI, de ancora + VerticalLayoutGroup +
// ContentSizeFitter. O HUD do Lab e IMGUI (OnGUI) desde o comeco, e trocar por
// Canvas/TextMeshPro seria reescrever a interface inteira -- o que o pedido
// deste ajuste dizia pra nao fazer. As garantias pedidas (nada sobreposto, nada
// cortado, altura seguindo o conteudo, respiro consistente) estao aqui; o que
// muda e que a medicao e explicita em vez de delegada ao Canvas.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace Drosobot.Lab
{
    /// <summary>Espacamentos do HUD. Um lugar so, pra todos os paineis.</summary>
    public static class Espaco
    {
        public const float Margem = 12f;   // da borda da tela ate a primeira caixa
        public const float Vao = 10f;      // entre caixas
        public const float Pad = 12f;      // da borda da caixa ate o conteudo
        public const float Linha = 4f;     // respiro entre blocos dentro da caixa
    }

    /// <summary>Um painel do HUD: conteudo declarado, altura medida.</summary>
    public class Painel
    {
        private enum Tipo { Texto, Espacador, Desenho }

        private struct Item
        {
            public Tipo tipo;
            public string texto;
            public GUIStyle estilo;
            public Color cor;
            public bool temCor;
            public float altura;          // espacador e desenho
            public Action<Rect> desenhar;
        }

        private readonly List<Item> _itens = new List<Item>();

        public string Nome { get; }
        public bool Vazio => _itens.Count == 0;

        public Painel(string nome) { Nome = nome; }

        public void Limpar() => _itens.Clear();

        public Painel Texto(string t, GUIStyle estilo)
        {
            _itens.Add(new Item { tipo = Tipo.Texto, texto = t, estilo = estilo });
            return this;
        }

        public Painel Texto(string t, GUIStyle estilo, Color cor)
        {
            _itens.Add(new Item { tipo = Tipo.Texto, texto = t, estilo = estilo,
                                  cor = cor, temCor = true });
            return this;
        }

        public Painel Espacador(float h = Espaco.Linha)
        {
            _itens.Add(new Item { tipo = Tipo.Espacador, altura = h });
            return this;
        }

        /// <summary>Bloco de altura fixa: sparkline, retina, barra.</summary>
        public Painel Desenho(float altura, Action<Rect> desenhar)
        {
            _itens.Add(new Item { tipo = Tipo.Desenho, altura = altura, desenhar = desenhar });
            return this;
        }

        /// <summary>
        /// Altura necessaria pra este conteudo nesta largura.
        ///
        /// CalcHeight e o ponto do arquivo: ele ja conta a quebra de linha. Uma
        /// linha que nao cabe vira duas, e a caixa cresce -- que e o oposto do
        /// que acontecia com altura fixa.
        /// </summary>
        public float Medir(float larguraCaixa)
        {
            float util = larguraCaixa - Espaco.Pad * 2f;
            float h = Espaco.Pad * 2f;
            foreach (var it in _itens)
            {
                switch (it.tipo)
                {
                    case Tipo.Texto:
                        h += it.estilo.CalcHeight(new GUIContent(it.texto), util);
                        break;
                    default:
                        h += it.altura;
                        break;
                }
            }
            return Mathf.Ceil(h);
        }

        public void Desenhar(Rect caixa)
        {
            GUI.Box(caixa, GUIContent.none);
            float util = caixa.width - Espaco.Pad * 2f;
            float y = caixa.y + Espaco.Pad;

            foreach (var it in _itens)
            {
                if (it.tipo == Tipo.Texto)
                {
                    float h = it.estilo.CalcHeight(new GUIContent(it.texto), util);
                    var antes = GUI.color;
                    if (it.temCor) GUI.color = it.cor;
                    GUI.Label(new Rect(caixa.x + Espaco.Pad, y, util, h), it.texto, it.estilo);
                    GUI.color = antes;
                    y += h;
                }
                else if (it.tipo == Tipo.Desenho)
                {
                    it.desenhar?.Invoke(new Rect(caixa.x + Espaco.Pad, y, util, it.altura));
                    y += it.altura;
                }
                else
                {
                    y += it.altura;
                }
            }
        }
    }

    /// <summary>
    /// Uma pilha de paineis ancorada num canto da tela.
    ///
    /// Empilha de cima pra baixo (ancora superior) ou de baixo pra cima
    /// (inferior). O `x` vem do canto e da largura, entao colunas vizinhas nunca
    /// se encostam: e so pedir a proxima a partir do `Direita` da anterior.
    /// </summary>
    public class Coluna
    {
        public enum Ancora { SuperiorEsquerda, SuperiorDireita, InferiorEsquerda, InferiorDireita }

        private readonly List<Painel> _paineis = new List<Painel>();
        private Vector2 _rolagem;

        public Ancora ancora;
        public float largura;
        public float x;             // resolvido em Resolver()
        public float Direita => x + largura;

        public Coluna(Ancora ancora, float largura)
        {
            this.ancora = ancora;
            this.largura = largura;
        }

        public void Limpar() { _paineis.Clear(); }

        public void Adiciona(Painel p)
        {
            if (p != null && !p.Vazio) _paineis.Add(p);
        }

        /// <summary>
        /// Posiciona e desenha; devolve a altura ocupada.
        ///
        /// `deslocamentoX` move a coluna pro lado, pra colunas lado a lado.
        /// `topoOcupado` e quanto do alto ja esta tomado por outra coluna do
        /// mesmo lado -- e o que impede um bloco ancorado embaixo de subir por
        /// cima de um ancorado em cima numa janela baixa.
        /// </summary>
        public float Desenhar(float deslocamentoX = 0f, float topoOcupado = 0f)
        {
            bool direita = ancora == Ancora.SuperiorDireita || ancora == Ancora.InferiorDireita;
            bool baixo = ancora == Ancora.InferiorEsquerda || ancora == Ancora.InferiorDireita;

            x = direita
                ? Screen.width - Espaco.Margem - largura - deslocamentoX
                : Espaco.Margem + deslocamentoX;

            var alturas = new float[_paineis.Count];
            float total = 0f;
            for (int i = 0; i < _paineis.Count; i++)
            {
                alturas[i] = _paineis[i].Medir(largura);
                total += alturas[i] + (i > 0 ? Espaco.Vao : 0f);
            }

            float disponivel = Screen.height - Espaco.Margem * 2f - topoOcupado;

            // Nao cabe na tela: a coluna rola em vez de cortar o ultimo painel.
            // So acontece em janela baixa -- em 1920x1080 nenhuma coluna chega
            // perto disso -- mas cortar em silencio e exatamente o defeito que
            // este arquivo veio corrigir.
            if (total > disponivel)
            {
                var area = new Rect(x, Espaco.Margem + topoOcupado, largura + 16f, disponivel);
                _rolagem = GUI.BeginScrollView(area, _rolagem,
                                               new Rect(0, 0, largura, total));
                float yr = 0f;
                for (int i = 0; i < _paineis.Count; i++)
                {
                    _paineis[i].Desenhar(new Rect(0, yr, largura, alturas[i]));
                    yr += alturas[i] + Espaco.Vao;
                }
                GUI.EndScrollView();
                return disponivel;
            }

            float y = baixo
                ? Screen.height - Espaco.Margem - total
                : Espaco.Margem + topoOcupado;
            for (int i = 0; i < _paineis.Count; i++)
            {
                _paineis[i].Desenhar(new Rect(x, y, largura, alturas[i]));
                y += alturas[i] + Espaco.Vao;
            }
            return total;
        }
    }
}
