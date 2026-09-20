// Propagacao sinaptica esparsa sobre o Male CNS real, medida na RX 6700 XT.
//
// Este e o experimento que decide se whole-CNS em tempo real e possivel.
//
// O passo LIF denso ja se mostrou barato (17,9 us para 166k neuronios). O que
// faltava era: propagar por 25,5M arestas custa quanto?
//
// Orcamento: dt neural = 0,5 ms. Pra tempo real, o passo neural inteiro
// (LIF + compactacao + scatter) tem que caber em 500 us.
//
// Rodar:
//   unity cmd eval --code "Drosobot.Compute.SparseBenchmark.RodarTudo()"
//
// Dados: data/male-cns/csr/*.bin, gerados por benchmarks/neural/build_csr.py.
// Nao sao versionados (195 MiB); o script os reconstroi.

using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Drosobot.Compute
{
    public static class SparseBenchmark
    {
        // Conductancia acumulada em ponto fixo: HLSL SM 5.0 nao tem
        // InterlockedAdd em float. 1024 passos por mV deixa o degrau em
        // ~0,001 mV, uma ordem abaixo do erro de fp32 que ja medimos (6e-05 mV)
        // -- entao a quantizacao nao e o termo dominante.
        const float ESCALA = 1024f;
        const int GRUPO = 64;

        // taxas de disparo a medir. 100% e teste de estresse, nao regime
        // biologico -- em atividade fisiologica a fracao que dispara por passo
        // de 0,5 ms e bem menor que 1%.
        static readonly float[] TAXAS = { 0.001f, 0.005f, 0.01f, 0.05f, 0.10f, 1.00f };
        const int PASSOS = 100;

        static string DirCsr()
        {
            return Path.GetFullPath(Path.Combine(
                Application.dataPath, "..", "..", "..", "data", "male-cns", "csr"));
        }

        // BlockCopy em vez de BinaryReader elemento a elemento: sao 25,5M
        // valores por array, e ler um por um levava a chamada a estourar o
        // timeout do Editor antes de comecar a medir.
        static int[] LeInts(string caminho)
        {
            var bytes = File.ReadAllBytes(caminho);
            var saida = new int[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, saida, 0, bytes.Length);
            return saida;
        }

        static float[] LeFloats(string caminho)
        {
            var bytes = File.ReadAllBytes(caminho);
            var saida = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, saida, 0, bytes.Length);
            return saida;
        }

        // ---- estado entre chamadas ----
        //
        // O servidor do Editor limita CADA chamada a 5 s na thread principal, e
        // carregar 195 MiB de CSR mais medir seis taxas passa disso com folga.
        // Entao o benchmark virou etapas, guardando os buffers num estatico:
        //
        //   Preparar()      carrega o CSR e sobe pra GPU
        //   MedirTaxa(r)    mede UMA taxa de disparo
        //   Finalizar()     escreve o JSON e libera
        static ComputeShader _cs;
        static ComputeBuffer _bOff, _bTgt, _bW, _bSpike, _bAcum, _bFront, _bCont, _bArgs;
        static int _kLimpa, _kComp, _kFront, _kDenso;
        static int _n, _e;
        static double _vramMB;
        static int[] _offsetsCache;
        static readonly StringBuilder _linhas = new StringBuilder();

        public static string Preparar()
        {
            var dir = DirCsr();
            string fOff = Path.Combine(dir, "row_offsets_full.bin");
            string fTgt = Path.Combine(dir, "targets_full.bin");
            string fW = Path.Combine(dir, "weights_full.bin");
            if (!File.Exists(fOff))
                return "ERRO: falta " + fOff + ". Rode benchmarks/neural/build_csr.py";

            var crono = Stopwatch.StartNew();
            var rowOffsets = LeInts(fOff);
            var targets = LeInts(fTgt);
            var weights = LeFloats(fW);
            crono.Stop();

            _n = rowOffsets.Length - 1;
            _e = targets.Length;

            _cs = CarregaShader("Assets/Compute/SynapticScatter.compute");
            if (_cs == null) return "ERRO: SynapticScatter.compute nao encontrado";
            _kLimpa = _cs.FindKernel("LimpaAcum");
            _kComp = _cs.FindKernel("CompactaSpikes");
            _kFront = _cs.FindKernel("ScatterFrontier");
            _kDenso = _cs.FindKernel("ScatterDenso");

            Liberar();
            _bOff = new ComputeBuffer(_n + 1, 4);
            _bTgt = new ComputeBuffer(_e, 4);
            _bW = new ComputeBuffer(_e, 4);
            _bSpike = new ComputeBuffer(_n, 4);
            _bAcum = new ComputeBuffer(_n, 4);
            _bFront = new ComputeBuffer(_n, 4);
            _bCont = new ComputeBuffer(1, 4);
            _bArgs = new ComputeBuffer(3, 4, ComputeBufferType.IndirectArguments);
            _bOff.SetData(rowOffsets);
            _bTgt.SetData(targets);
            _bW.SetData(weights);

            _vramMB = ((double)(_n + 1) * 4 + (double)_e * 4 * 2
                       + (double)_n * 4 * 4) / (1024 * 1024);
            _linhas.Clear();
            _offsetsCache = rowOffsets;

            return string.Format(
                "device={0} api={1} | grafo: {2:N0} neuronios, {3:N0} arestas, "
                + "{4:F1} MiB (carregado em {5} ms)",
                SystemInfo.graphicsDeviceName, SystemInfo.graphicsDeviceType,
                _n, _e, _vramMB, crono.ElapsedMilliseconds);
        }

        public static string MedirTaxa(float taxa)
        {
            if (_bOff == null) return "ERRO: chame Preparar() antes";

            // semente derivada da taxa: as duas estrategias veem o MESMO padrao
            var rng = new System.Random((int)(taxa * 1000000) + 7);
            var spike = new uint[_n];
            int ativos = 0;
            for (int i = 0; i < _n; i++)
            {
                bool f = rng.NextDouble() < taxa;
                spike[i] = f ? 1u : 0u;
                if (f) ativos++;
            }
            _bSpike.SetData(spike);

            long arestas = 0;
            for (int i = 0; i < _n; i++)
                if (spike[i] != 0) arestas += _offsetsCache[i + 1] - _offsetsCache[i];

            double msFront = Medir(true);
            double msDenso = Medir(false);
            double melhor = Math.Min(msFront, msDenso);
            double eventos = melhor > 0 ? arestas / (melhor / 1000.0) : 0;

            var ci = CultureInfo.InvariantCulture;
            _linhas.Append("    {RATE_JSON}".Replace("RATE_JSON", string.Format(ci,
                "QUOTErateQUOTE: {0}, QUOTEactiveQUOTE: {1}, QUOTEedges_touchedQUOTE: {2}, "
                + "QUOTEfrontier_usQUOTE: {3:F2}, QUOTEdense_usQUOTE: {4:F2}, "
                + "QUOTEevents_per_sQUOTE: {5:F0}",
                taxa, ativos, arestas, msFront * 1000, msDenso * 1000, eventos)
                .Replace("QUOTE", "\"")) + ",\n");

            return string.Format(ci,
                "{0,6:F1}%  ativos {1,7:N0}  arestas {2,12:N0}  "
                + "FRONTIER {3,8:F1}us  DENSO {4,8:F1}us  {5,6:F2} G eventos/s",
                taxa * 100, ativos, arestas, msFront * 1000, msDenso * 1000,
                eventos / 1e9);
        }

        public static string Finalizar()
        {
            EscreveJson(_n, _e, _vramMB, _linhas.ToString());
            Liberar();
            return "json escrito, buffers liberados";
        }

        static void Liberar()
        {
            foreach (var b in new[] { _bOff, _bTgt, _bW, _bSpike, _bAcum,
                                      _bFront, _bCont, _bArgs })
                if (b != null) b.Release();
            _bOff = _bTgt = _bW = _bSpike = _bAcum = _bFront = _bCont = _bArgs = null;
        }

        static double Medir(bool usarFrontier)
        {
            foreach (int k in new[] { _kLimpa, _kComp, _kFront, _kDenso })
            {
                _cs.SetBuffer(k, "_RowOffsets", _bOff);
                _cs.SetBuffer(k, "_Targets", _bTgt);
                _cs.SetBuffer(k, "_Weights", _bW);
                _cs.SetBuffer(k, "_Spike", _bSpike);
                _cs.SetBuffer(k, "_GAcum", _bAcum);
                _cs.SetBuffer(k, "_Frontier", _bFront);
                _cs.SetBuffer(k, "_Contador", _bCont);
                _cs.SetBuffer(k, "_DispatchArgs", _bArgs);
            }
            _cs.SetInt("_N", _n);
            _cs.SetFloat("_Escala", ESCALA);
            _cs.SetInt("_GrupoScatter", GRUPO);
            int grupos = (_n + GRUPO - 1) / GRUPO;

            Action passo = () =>
            {
                _cs.Dispatch(_kLimpa, grupos, 1, 1);
                if (usarFrontier)
                {
                    _cs.Dispatch(_kComp, grupos, 1, 1);
                    // DispatchIndirect: o tamanho da frontier nunca passa pela CPU
                    _cs.DispatchIndirect(_kFront, _bArgs);
                }
                else
                {
                    _cs.Dispatch(_kDenso, grupos, 1, 1);
                }
            };

            for (int i = 0; i < 10; i++) passo();
            var descarte = new int[1];
            _bAcum.GetData(descarte, 0, 0, 1);

            var crono = Stopwatch.StartNew();
            for (int i = 0; i < PASSOS; i++) passo();
            _bAcum.GetData(descarte, 0, 0, 1);
            crono.Stop();
            return crono.Elapsed.TotalMilliseconds / PASSOS;
        }

        static void EscreveJson(int n, int e, double vramMB, string linhas)
        {
            var sb = new StringBuilder();
            sb.AppendLine("{");
            sb.AppendLine($"  \"probed_at\": \"{DateTime.Now:yyyy-MM-ddTHH:mm:ss}\",");
            sb.AppendLine($"  \"backend\": \"unity_computeshader_d3d12\",");
            sb.AppendLine($"  \"device\": \"{SystemInfo.graphicsDeviceName}\",");
            sb.AppendLine($"  \"neurons\": {n}, \"edges\": {e},");
            sb.AppendLine($"  \"graph_vram_mib\": {vramMB.ToString("F1", CultureInfo.InvariantCulture)},");
            sb.AppendLine($"  \"steps_per_measurement\": {PASSOS},");
            sb.AppendLine("  \"results\": [");
            sb.Append(linhas.TrimEnd('\n', ',').TrimEnd(',') + "\n");
            sb.AppendLine("  ]");
            sb.AppendLine("}");
            var dir = Path.GetFullPath(Path.Combine(
                Application.dataPath, "..", "..", "..", "benchmarks", "neural"));
            Directory.CreateDirectory(dir);
            var caminho = Path.Combine(
                dir, $"sparse_scatter_{DateTime.Now:yyyy-MM-dd_HHmmss}.json");
            File.WriteAllText(caminho, sb.ToString());
            Debug.Log($"[sparse-bench] salvo em {caminho}");
        }

        // `AssetDatabase` so existe no Editor. Sem a guarda, o codigo compila no
        // Editor e QUEBRA o build do player -- foi o que aconteceu na primeira
        // tentativa de gerar o executavel. Estes benchmarks sao ferramenta de
        // Editor; num player eles nao tem shader pra carregar, e dizem isso em
        // vez de estourar.
        private static ComputeShader CarregaShader(string caminho)
        {
#if UNITY_EDITOR
            return UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>(caminho);
#else
            Debug.LogWarning($"[benchmark] {caminho} so carrega no Editor");
            return null;
#endif
        }

    }
}
