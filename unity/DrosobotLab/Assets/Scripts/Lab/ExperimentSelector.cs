// Seletor de experimentos: escolher qual roda, com que semente, e start/pause/reset.
//
// O catalogo e o estado NAO sao inventados aqui. Os dois chegam pela telemetria
// (`experiment_list` e `run_state`), que e escrita pelo runner em Python. Esta
// classe nunca decide que algo esta rodando: ela mostra o que o runner disse.
// Se o runner cair, a interface passa a dizer "aguardando", e nao continua
// desenhando um experimento que nao existe mais.
//
// Os botoes escrevem pelo ControlClient, no socket de controle. O que eles podem
// mandar e curto e fechado: escolher experimento, semente, start, pause, resume,
// reset, stop. Nada de peso sinaptico, limiar ou comando motor -- isso continua
// sendo do conectoma e do MuJoCo, e o proprio runner recusa o resto.
//
// ## Resultado negativo
//
// O `obstacle_field` reproduz um resultado NEGATIVO: o circuito nao guia desvio
// de obstaculo. O experimento declara isso no `known_result` e o painel mostra
// antes e durante a corrida. Nao e ressalva de rodape -- quem abre o Lab precisa
// saber que esta vendo um negativo reproduzido, nao um desvio que quase deu.

using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using Drosobot.Telemetry;

namespace Drosobot.Lab
{
    public class ExperimentSelector : MonoBehaviour
    {
        public class Entrada
        {
            public string id, name, description;
        }

        [Header("Estado vindo do runner (somente leitura)")]
        public string estado = "desconhecido";
        public string experimentoAtual = "";
        public int seed;

        private readonly List<Entrada> _catalogo = new List<Entrada>();
        private ControlClient _ctl;
        private int _escolhido = -1;
        private int _seedEditada;
        private bool _seedTocada;
        private string _mensagemEstado = "";
        private string _resultadoConhecido = "";
        private string _causaConhecida = "";

        public bool Conectado => _ctl != null && _ctl.connected;
        public bool TemCatalogo => _catalogo.Count > 0;

        public void Ligar(ControlClient ctl)
        {
            _ctl = ctl;
        }

        // ------------------------------------------------- entrada de dados

        /// <summary>Uma mensagem de telemetria ja decodificada.</summary>
        public void AoReceber(JObject msg)
        {
            switch ((string)msg["type"])
            {
                case "experiment_list":
                    _catalogo.Clear();
                    foreach (var e in msg["experiments"])
                        _catalogo.Add(new Entrada
                        {
                            id = (string)e["id"],
                            name = (string)e["name"],
                            description = (string)e["description"],
                        });
                    var atual = (string)msg["current"];
                    if (!string.IsNullOrEmpty(atual)) SelecionaPorId(atual);
                    if (_escolhido < 0 && _catalogo.Count > 0) _escolhido = 0;
                    break;

                case "run_state":
                    estado = (string)msg["state"] ?? "?";
                    experimentoAtual = (string)msg["experiment_id"] ?? "";
                    seed = (int?)msg["seed"] ?? 0;
                    // Alinha o campo editavel com a corrida na PRIMEIRA vez que
                    // o estado chega. Sem isso o campo mostra 0 enquanto o
                    // rodape diz "seed 7", e um Reset mandaria 0 sem querer.
                    // Depois disso o campo e do usuario e nao e mais mexido.
                    if (!_seedTocada) { _seedEditada = seed; _seedTocada = true; }
                    _mensagemEstado = (string)msg["detail"]?["message"] ?? "";
                    if (!string.IsNullOrEmpty(experimentoAtual)) SelecionaPorId(experimentoAtual);
                    break;

                case "experiment_info":
                    // `known_result` so existe em experimento que JA sabe seu
                    // desfecho -- hoje, o campo de obstaculos.
                    var kr = msg["known_result"];
                    _resultadoConhecido = kr == null ? "" : (string)kr["summary"];
                    _causaConhecida = kr == null ? "" : (string)kr["cause"];
                    break;
            }
        }

        private void SelecionaPorId(string id)
        {
            for (int i = 0; i < _catalogo.Count; i++)
                if (_catalogo[i].id == id) { _escolhido = i; return; }
        }

        // ------------------------------------------------------------ painel

        public void Preenche(Painel p, GUIStyle rotulo, GUIStyle mono)
        {
            p.Texto("EXPERIMENTO", rotulo);

            if (!Conectado)
            {
                p.Texto("canal de controle desligado", mono);
                p.Texto("suba: python sim/lab_runner.py", mono);
                return;
            }
            if (!TemCatalogo)
            {
                p.Texto("aguardando o catalogo do runner...", mono);
                return;
            }

            // --- escolha ---
            float alturaLista = _catalogo.Count * 24f;
            p.Desenho(alturaLista, r =>
            {
                for (int i = 0; i < _catalogo.Count; i++)
                {
                    var linha = new Rect(r.x, r.y + i * 24f, r.width, 22f);
                    bool marcado = i == _escolhido;
                    var antes = GUI.color;
                    // Tinge o botao inteiro do selecionado. So um ">" no comeco
                    // da linha se perde numa lista de tres itens parecidos.
                    GUI.color = marcado ? new Color(0.45f, 0.80f, 1.00f)
                                        : new Color(0.78f, 0.80f, 0.84f);
                    if (GUI.Button(linha, (marcado ? "▸ " : "  ") + _catalogo[i].name))
                        _escolhido = i;
                    GUI.color = antes;
                }
            });

            var sel = _escolhido >= 0 && _escolhido < _catalogo.Count ? _catalogo[_escolhido] : null;
            if (sel != null)
            {
                p.Espacador();
                // O estilo quebra linha e o painel mede a altura, entao uma
                // descricao longa cresce a caixa em vez de sumir na borda.
                p.Texto(sel.description, mono);
            }

            // --- semente ---
            p.Espacador(8f);
            p.Desenho(22f, r =>
            {
                GUI.Label(new Rect(r.x, r.y, 52f, 20f), "seed", mono);
                var txt = GUI.TextField(new Rect(r.x + 54f, r.y, 60f, 20f),
                                        _seedEditada.ToString());
                if (int.TryParse(txt, out int v) && v != _seedEditada)
                { _seedEditada = Mathf.Max(0, v); _seedTocada = true; }
                if (GUI.Button(new Rect(r.x + 120f, r.y, 30f, 20f), "+1"))
                { _seedEditada++; _seedTocada = true; }
                if (GUI.Button(new Rect(r.x + 154f, r.y, 30f, 20f), "-1"))
                { _seedEditada = Mathf.Max(0, _seedEditada - 1); _seedTocada = true; }
            });

            // --- comandos ---
            p.Espacador(8f);
            bool rodando = estado == "running";
            bool pausado = estado == "paused";
            bool montando = estado == "loading";
            // 2x2, nao uma fila de quatro: a coluna e estreita e quatro botoes
            // lado a lado ficariam com rotulo espremido.
            p.Desenho(54f, r =>
            {
                float w = (r.width - 8f) / 2f;
                float h = 24f;
                var antes = GUI.enabled;

                GUI.enabled = !montando && sel != null;
                if (GUI.Button(new Rect(r.x, r.y, w, h), "Start"))
                    _ctl.Iniciar(sel.id, _seedEditada);

                GUI.enabled = rodando || pausado;
                if (GUI.Button(new Rect(r.x + w + 8f, r.y, w, h), pausado ? "Resume" : "Pause"))
                {
                    if (pausado) _ctl.Retomar(); else _ctl.Pausar();
                }

                GUI.enabled = !montando && (rodando || pausado);
                if (GUI.Button(new Rect(r.x, r.y + h + 6f, w, h), "Reset"))
                    _ctl.Reiniciar(_seedEditada);

                if (GUI.Button(new Rect(r.x + w + 8f, r.y + h + 6f, w, h), "Stop"))
                    _ctl.Parar();

                GUI.enabled = antes;
            });

            // --- estado, como o runner reportou ---
            p.Espacador();
            p.Texto($"estado: {estado}" + (string.IsNullOrEmpty(_mensagemEstado)
                                           ? "" : "  -- " + _mensagemEstado), mono);
            if (!string.IsNullOrEmpty(experimentoAtual))
                p.Texto($"rodando: {experimentoAtual}  seed {seed}", mono);

            // --- resultado negativo, quando for o caso ---
            if (!string.IsNullOrEmpty(_resultadoConhecido))
            {
                p.Espacador();
                var cor = ProvenanceUtil.Color(Provenance.Assumption);
                p.Texto("RESULTADO CONHECIDO: NEGATIVO", mono, cor);
                p.Texto(_resultadoConhecido, mono, cor);
                if (!string.IsNullOrEmpty(_causaConhecida))
                    p.Texto("causa: " + _causaConhecida, mono);
            }
        }
    }
}
