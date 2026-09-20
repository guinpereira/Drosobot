// Cliente do canal de controle: escolher experimento e apertar start/pause/reset.
//
// Este e o unico lugar do projeto onde a Unity ESCREVE alguma coisa pra
// simulacao, e por isso ele e um arquivo separado do TelemetryClient em vez de
// um metodo dentro dele. A separacao e pra ser obvia na leitura: quem quiser
// conferir que a visualizacao nao interfere no experimento le
// TelemetryClient.cs e nao encontra nenhum Write.
//
// O que cabe aqui e curto e fechado (ver sim/telemetry/control.py):
//
//     list  select  start  pause  resume  reset  stop  quit
//
// Nao existe comando pra mexer em peso sinaptico, limiar, taxa de disparo ou
// drive motor -- nem aqui nem do lado Python, que recusa o que nao esta na
// lista. A interface escolhe QUAL experimento roda; o conectoma continua sendo a
// autoridade do comportamento e o MuJoCo a da fisica.
//
// O estado de verdade NAO vem das respostas daqui: vem do `run_state` que chega
// pela telemetria. Um ack so diz "recebi" -- e no caso do `start` ele volta
// antes da montagem terminar, porque montar arena, mosca e circuito leva
// segundos.

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace Drosobot.Telemetry
{
    public class ControlClient : MonoBehaviour
    {
        [Header("Conexao")]
        public string host = "127.0.0.1";
        public int port = 8766;
        public bool autoConnect = true;
        public float retrySeconds = 2f;

        [Header("Estado (somente leitura)")]
        public bool connected;
        public int commandsSent;
        public string lastError = "";

        /// <summary>Ack cru do runner, como JSON.</summary>
        public event Action<string> OnAck;

        private readonly ConcurrentQueue<string> _saida = new ConcurrentQueue<string>();
        private readonly ConcurrentQueue<string> _acks = new ConcurrentQueue<string>();
        private Thread _thread;
        private volatile bool _parar;
        private TcpClient _tcp;

        void Start()
        {
            if (autoConnect) Connect();
        }

        public void Connect()
        {
            if (_thread != null && _thread.IsAlive) return;
            _parar = false;
            _thread = new Thread(Loop) { IsBackground = true, Name = "DrosobotControl" };
            _thread.Start();
        }

        public void Disconnect()
        {
            _parar = true;
            try { _tcp?.Close(); } catch { }
            _tcp = null;
            connected = false;
        }

        void OnDestroy() => Disconnect();
        void OnApplicationQuit() => Disconnect();

        void Update()
        {
            while (_acks.TryDequeue(out var linha))
            {
                try { OnAck?.Invoke(linha); }
                catch (Exception e) { Debug.LogError($"[control] ack ilegivel: {e.Message}"); }
            }
        }

        // ------------------------------------------------------------ comandos

        public void Listar() => Enviar("{\"command\":\"list\"}");

        public void Selecionar(string experimentId, int seed) =>
            Enviar($"{{\"command\":\"select\",\"experiment_id\":\"{Escapa(experimentId)}\",\"seed\":{seed}}}");

        public void Iniciar(string experimentId, int seed) =>
            Enviar($"{{\"command\":\"start\",\"experiment_id\":\"{Escapa(experimentId)}\",\"seed\":{seed}}}");

        public void Pausar() => Enviar("{\"command\":\"pause\"}");
        public void Retomar() => Enviar("{\"command\":\"resume\"}");
        public void Parar() => Enviar("{\"command\":\"stop\"}");

        public void Reiniciar(int seed) =>
            Enviar($"{{\"command\":\"reset\",\"seed\":{seed}}}");

        private static string Escapa(string s) =>
            (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");

        /// <summary>Enfileira. Nunca bloqueia a thread principal da Unity.</summary>
        private void Enviar(string json)
        {
            if (!connected)
            {
                lastError = "runner nao conectado";
                Debug.LogWarning($"[control] {lastError}: {json}");
                return;
            }
            _saida.Enqueue(json);
            commandsSent++;
        }

        // -------------------------------------------------------------- thread

        private void Loop()
        {
            while (!_parar)
            {
                try
                {
                    _tcp = new TcpClient();
                    _tcp.Connect(host, port);
                    _tcp.NoDelay = true;
                    connected = true;
                    lastError = "";
                    Debug.Log($"[control] conectado em {host}:{port}");

                    using (var stream = _tcp.GetStream())
                    using (var reader = new StreamReader(stream, Encoding.UTF8))
                    {
                        while (!_parar && _tcp.Connected)
                        {
                            while (_saida.TryDequeue(out var json))
                            {
                                var bytes = Encoding.UTF8.GetBytes(json + "\n");
                                stream.Write(bytes, 0, bytes.Length);
                                stream.Flush();
                            }
                            // Le os acks que ja chegaram sem bloquear: um
                            // ReadLine parado aqui impediria o proximo comando de
                            // sair enquanto o runner monta a cena.
                            while (stream.DataAvailable)
                            {
                                var linha = reader.ReadLine();
                                if (!string.IsNullOrEmpty(linha)) _acks.Enqueue(linha);
                            }
                            Thread.Sleep(20);
                        }
                    }
                }
                catch (Exception e)
                {
                    if (!_parar)
                    {
                        lastError = e.GetType().Name;
                        Debug.Log($"[control] sem conexao ({lastError}); tentando de novo");
                    }
                }
                finally
                {
                    connected = false;
                    try { _tcp?.Close(); } catch { }
                }

                if (_parar) break;
                Thread.Sleep((int)(retrySeconds * 1000));
            }
        }
    }
}
