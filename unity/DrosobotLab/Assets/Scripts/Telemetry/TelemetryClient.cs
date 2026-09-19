// Cliente de telemetria do Drosobot Lab.
//
// Le o stream TCP que sim/telemetry/server.py publica: uma mensagem JSON por
// linha. A leitura roda numa thread separada e enfileira; o MonoBehaviour drena
// a fila no Update, porque a API da Unity so pode ser tocada na thread principal.
//
// DIRECAO UNICA. Este cliente so LE. Nao existe caminho de volta pra simulacao:
// a Unity nao decide se a mosca virou, se escapou ou se um neuronio disparou.
// MuJoCo continua sendo a autoridade da fisica e o circuito a do comportamento.
//
// Se a simulacao estiver 25x mais lenta que tempo real (e esta), a interface
// continua a 60 fps -- ela so desenha o ultimo estado que chegou.

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace Drosobot.Telemetry
{
    public class TelemetryClient : MonoBehaviour
    {
        [Header("Conexao")]
        public string host = "127.0.0.1";
        public int port = 8765;
        public bool autoConnect = true;
        [Tooltip("Segundos entre tentativas quando a simulacao ainda nao subiu")]
        public float retrySeconds = 2f;

        [Header("Estado (somente leitura)")]
        public bool connected;
        public int messagesReceived;
        public int messagesDropped;

        /// <summary>Uma mensagem crua por linha, ainda como JSON.</summary>
        public event Action<string> OnMessage;

        // Limite da fila: se a interface travar por um instante, e melhor perder
        // quadro de visualizacao do que estourar a memoria. A simulacao nunca
        // espera por isto de qualquer forma -- ela nem sabe que existimos.
        private const int MaxQueue = 512;

        private readonly ConcurrentQueue<string> _fila = new ConcurrentQueue<string>();
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
            _thread = new Thread(Loop) { IsBackground = true, Name = "DrosobotTelemetry" };
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
            // Drena tudo que chegou desde o quadro anterior. Processar na thread
            // principal e obrigatorio: a API da Unity nao e thread-safe.
            int processadas = 0;
            while (processadas < MaxQueue && _fila.TryDequeue(out var linha))
            {
                processadas++;
                messagesReceived++;
                try { OnMessage?.Invoke(linha); }
                catch (Exception e) { Debug.LogError($"[telemetry] mensagem descartada: {e.Message}"); }
            }
        }

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
                    Debug.Log($"[telemetry] conectado em {host}:{port}");

                    using (var stream = _tcp.GetStream())
                    using (var reader = new StreamReader(stream, System.Text.Encoding.UTF8))
                    {
                        string linha;
                        while (!_parar && (linha = reader.ReadLine()) != null)
                        {
                            if (linha.Length == 0) continue;
                            if (_fila.Count >= MaxQueue)
                            {
                                // descarta a mais antiga: quadro velho nao serve
                                _fila.TryDequeue(out _);
                                messagesDropped++;
                            }
                            _fila.Enqueue(linha);
                        }
                    }
                }
                catch (Exception e)
                {
                    if (!_parar) Debug.Log($"[telemetry] sem conexao ({e.GetType().Name}); tentando de novo");
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
