// Microbenchmark: o kernel LIF rodando de verdade na GPU, nesta maquina.
//
// O que este arquivo tem que provar, e nada alem disso:
//
//   1. compute shader executa na RX 6700 XT, no Windows, via D3D12
//   2. o resultado numerico bate com sim/fast_lif.py (a referencia validada
//      contra o Brian2)
//   3. quanto custa, em varios tamanhos de rede, contra o Ryzen 7 5700X
//
// NAO prova que devemos adotar Unity ComputeShader como backend. Isso depende
// tambem do custo de readback e da propagacao sinaptica esparsa, que sao outro
// experimento. Ver docs/research/DROSOBOT_COMPUTE_ARCHITECTURE.md.
//
// Rodar pela linha de comando (o resultado sai no console e num JSON):
//
//   unity cmd eval --code "Drosobot.Compute.LifBenchmark.RodarTudo()"

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Drosobot.Compute
{
    public static class LifBenchmark
    {
        // Parametros de Shiu et al., os mesmos de sim/connectome_model.py.
        // Duplicados aqui de proposito e marcados: o teste de equivalencia em
        // Python compara contra a referencia e falha se divergirem.
        const float V_REST = -52f;
        const float V_RESET = -52f;
        const float V_TH = -45f;
        const float TAU_M = 20f;      // ms
        const float TAU_S = 5f;       // ms
        const float T_REF = 2.2f;     // ms
        const float DT = 0.5f;        // ms

        static readonly int[] TAMANHOS = { 1000, 10000, 50000, 166691 };
        const int PASSOS = 2000;

        public class Resultado
        {
            public int n;
            public int passos;
            public double gpuMs;
            public double gpuMsPorPasso;
            public double atualizacoesPorSegundo;
            public double readbackMs;
            public long spikesTotal;
            public double vramMB;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Aviso() { }

        public static string RodarTudo()
        {
            var cs = Resources.Load<ComputeShader>("LifStep");
            if (cs == null)
            {
                // O .compute vive em Assets/Compute/, que nao e Resources.
                // Carregamos pelo AssetDatabase no Editor; num build seria
                // referencia serializada.
#if UNITY_EDITOR
                cs = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>(
                    "Assets/Compute/LifStep.compute");
#endif
            }
            if (cs == null) return "ERRO: nao achei Assets/Compute/LifStep.compute";
            if (!SystemInfo.supportsComputeShaders) return "ERRO: sem compute shader";

            var sb = new StringBuilder();
            sb.AppendLine($"device={SystemInfo.graphicsDeviceName} " +
                          $"api={SystemInfo.graphicsDeviceType} " +
                          $"vram={SystemInfo.graphicsMemorySize}MB");

            var resultados = new List<Resultado>();
            foreach (int n in TAMANHOS)
            {
                var r = Rodar(cs, n, PASSOS);
                resultados.Add(r);
                sb.AppendLine(
                    $"n={r.n,7}  {r.gpuMsPorPasso * 1000.0,8:F1} us/passo  " +
                    $"{r.atualizacoesPorSegundo / 1e9,7:F2} G atualizacoes/s  " +
                    $"readback {r.readbackMs,6:F2} ms  spikes {r.spikesTotal}");
            }

            EscreveJson(resultados);
            var texto = sb.ToString();
            Debug.Log("[lif-bench]\n" + texto);
            return texto;
        }

        public static Resultado Rodar(ComputeShader cs, int n, int passos)
        {
            int k = cs.FindKernel("LifStep");

            // Coeficientes calculados AQUI, em double, exatamente como em
            // fast_lif.Camada.__init__, e so entao reduzidos a float. Calcula-los
            // dentro do shader em fp32 daria outro ultimo bit.
            double decV = Math.Exp(-DT / TAU_M);
            double decG = Math.Exp(-DT / TAU_S);
            double a = 1.0 / TAU_M - 1.0 / TAU_S;
            double acopla = (decG - decV) / (TAU_M * a);
            int refPassos = (int)Math.Round(T_REF / DT);   // 2.2/0.5 = 4.4 -> 4

            var v = new float[n];
            var g = new float[n];
            var refAte = new int[n];
            var entrada = new float[n];
            // Entrada deterministica: nao e ruido de Poisson, e um padrao fixo.
            // O objetivo aqui e MEDIR e COMPARAR, e pra comparar com a CPU o
            // estimulo tem que ser identico dos dois lados.
            //
            // O nivel nao e arbitrario. Com tau_s=5ms e dt=0.5ms o regime
            // estacionario da conductancia e entrada/(1-dec_g), e o potencial
            // resultante e g*acopla/(1-dec_v). Pra cruzar os 7 mV que separam
            // V_REST do limiar, a entrada tem que passar de ~0.7 mV/passo. Com
            // 0.30 o kernel rodava e NINGUEM disparava -- o caminho refratario
            // nunca era exercitado e o benchmark media meio kernel.
            for (int i = 0; i < n; i++)
            {
                v[i] = V_REST;
                g[i] = 0f;
                refAte[i] = -1;
                entrada[i] = 0.80f + 0.20f * (float)Math.Sin(i * 0.001);
            }

            var bV = new ComputeBuffer(n, sizeof(float));
            var bG = new ComputeBuffer(n, sizeof(float));
            var bR = new ComputeBuffer(n, sizeof(int));
            var bS = new ComputeBuffer(n, sizeof(uint));
            var bC = new ComputeBuffer(n, sizeof(uint));
            var bE = new ComputeBuffer(n, sizeof(float));
            bV.SetData(v); bG.SetData(g); bR.SetData(refAte); bE.SetData(entrada);
            bC.SetData(new uint[n]);

            cs.SetBuffer(k, "_V", bV);
            cs.SetBuffer(k, "_G", bG);
            cs.SetBuffer(k, "_RefAte", bR);
            cs.SetBuffer(k, "_Spike", bS);
            cs.SetBuffer(k, "_Contagem", bC);
            cs.SetBuffer(k, "_Entrada", bE);
            cs.SetFloat("_VRest", V_REST);
            cs.SetFloat("_VReset", V_RESET);
            cs.SetFloat("_VTh", V_TH);
            cs.SetFloat("_DecV", (float)decV);
            cs.SetFloat("_DecG", (float)decG);
            cs.SetFloat("_Acopla", (float)acopla);
            cs.SetInt("_RefPassos", refPassos);
            cs.SetInt("_N", n);

            int grupos = (n + 63) / 64;

            // Aquecimento: a primeira compilacao/dispatch do pipeline nao conta.
            for (int p = 0; p < 50; p++)
            {
                cs.SetInt("_Passo", p);
                cs.Dispatch(k, grupos, 1, 1);
            }
            // Forca a fila a esvaziar antes de comecar a cronometrar: sem isto
            // estariamos medindo enfileiramento, nao execucao.
            var descarte = new uint[1];
            bS.GetData(descarte, 0, 0, 1);

            var crono = Stopwatch.StartNew();
            for (int p = 0; p < passos; p++)
            {
                cs.SetInt("_Passo", p);
                cs.Dispatch(k, grupos, 1, 1);
            }
            // GetData e sincrono: bloqueia ate a GPU terminar tudo que foi
            // enfileirado. E o que fecha a janela de medicao.
            var spikes = new uint[n];
            bS.GetData(spikes);
            crono.Stop();
            double totalMs = crono.Elapsed.TotalMilliseconds;

            // readback medido separado, porque e o custo que decide se este
            // backend serve pro laco fechado com telemetria
            var cronoRb = Stopwatch.StartNew();
            bS.GetData(spikes);
            cronoRb.Stop();

            long soma = 0;
            for (int i = 0; i < n; i++) soma += spikes[i];

            double bytes = (double)n * (4 + 4 + 4 + 4 + 4 + 4);
            var r = new Resultado
            {
                n = n,
                passos = passos,
                gpuMs = totalMs,
                gpuMsPorPasso = totalMs / passos,
                atualizacoesPorSegundo = (double)n * passos / (totalMs / 1000.0),
                readbackMs = cronoRb.Elapsed.TotalMilliseconds,
                spikesTotal = soma,
                vramMB = bytes / (1024.0 * 1024.0),
            };

            bV.Release(); bG.Release(); bR.Release();
            bS.Release(); bC.Release(); bE.Release();
            return r;
        }

        /// <summary>
        /// Roda N passos e despeja o estado final, pra comparacao com
        /// sim/fast_lif.py.
        ///
        /// Um benchmark que so mede velocidade nao prova nada neste projeto: se
        /// o kernel estiver errado, ele so esta errado mais rapido. Este dump e
        /// o que o teste de equivalencia em Python le.
        /// </summary>
        public static string Validar(int n = 1000, int passos = 400)
        {
            var cs = CarregaShader("Assets/Compute/LifStep.compute");
            if (cs == null) return "ERRO: shader nao encontrado";
            int k = cs.FindKernel("LifStepConta");

            double decV = Math.Exp(-DT / TAU_M);
            double decG = Math.Exp(-DT / TAU_S);
            double a = 1.0 / TAU_M - 1.0 / TAU_S;
            double acopla = (decG - decV) / (TAU_M * a);
            int refPassos = (int)Math.Round(T_REF / DT);

            var v = new float[n]; var g = new float[n];
            var refAte = new int[n]; var entrada = new float[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = V_REST; g[i] = 0f; refAte[i] = -1;
                entrada[i] = 0.80f + 0.20f * (float)Math.Sin(i * 0.001);
            }

            var bV = new ComputeBuffer(n, 4); var bG = new ComputeBuffer(n, 4);
            var bR = new ComputeBuffer(n, 4); var bS = new ComputeBuffer(n, 4);
            var bC = new ComputeBuffer(n, 4); var bE = new ComputeBuffer(n, 4);
            bV.SetData(v); bG.SetData(g); bR.SetData(refAte);
            bE.SetData(entrada); bC.SetData(new uint[n]); bS.SetData(new uint[n]);

            cs.SetBuffer(k, "_V", bV); cs.SetBuffer(k, "_G", bG);
            cs.SetBuffer(k, "_RefAte", bR); cs.SetBuffer(k, "_Spike", bS);
            cs.SetBuffer(k, "_Contagem", bC); cs.SetBuffer(k, "_Entrada", bE);
            cs.SetFloat("_VRest", V_REST); cs.SetFloat("_VReset", V_RESET);
            cs.SetFloat("_VTh", V_TH); cs.SetFloat("_DecV", (float)decV);
            cs.SetFloat("_DecG", (float)decG); cs.SetFloat("_Acopla", (float)acopla);
            cs.SetInt("_RefPassos", refPassos); cs.SetInt("_N", n);

            int grupos = (n + 63) / 64;
            for (int p = 0; p < passos; p++)
            {
                cs.SetInt("_Passo", p);
                cs.Dispatch(k, grupos, 1, 1);
            }
            bV.GetData(v); bG.GetData(g); bR.GetData(refAte);
            var contagem = new uint[n]; bC.GetData(contagem);

            var ci = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            sb.AppendLine("{");
            sb.AppendLine($"  \"n\": {n}, \"steps\": {passos},");
            sb.AppendLine($"  \"device\": \"{SystemInfo.graphicsDeviceName}\",");
            sb.AppendLine($"  \"api\": \"{SystemInfo.graphicsDeviceType}\",");
            Action<string, Func<int, string>> arr = (nome, f) =>
            {
                sb.Append($"  \"{nome}\": [");
                for (int i = 0; i < n; i++)
                { sb.Append(f(i)); if (i < n - 1) sb.Append(","); }
                sb.AppendLine("],");
            };
            arr("v", i => v[i].ToString("R", ci));
            arr("g", i => g[i].ToString("R", ci));
            arr("ref_ate", i => refAte[i].ToString(ci));
            sb.Append("  \"contagem\": [");
            for (int i = 0; i < n; i++)
            { sb.Append(contagem[i]); if (i < n - 1) sb.Append(","); }
            sb.AppendLine("]");
            sb.AppendLine("}");

            var dir = Path.GetFullPath(Path.Combine(
                Application.dataPath, "..", "..", "..", "benchmarks", "neural"));
            Directory.CreateDirectory(dir);
            var caminho = Path.Combine(dir, "lif_gpu_estado.json");
            File.WriteAllText(caminho, sb.ToString());

            bV.Release(); bG.Release(); bR.Release();
            bS.Release(); bC.Release(); bE.Release();
            long soma = 0; foreach (var c in contagem) soma += c;
            return $"n={n} passos={passos} spikes_total={soma} -> {caminho}";
        }

        static void EscreveJson(List<Resultado> rs)
        {
            var ci = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            sb.AppendLine("{");
            sb.AppendLine($"  \"probed_at\": \"{DateTime.Now:yyyy-MM-ddTHH:mm:ss}\",");
            sb.AppendLine($"  \"backend\": \"unity_computeshader\",");
            sb.AppendLine($"  \"api\": \"{SystemInfo.graphicsDeviceType}\",");
            sb.AppendLine($"  \"device\": \"{SystemInfo.graphicsDeviceName}\",");
            sb.AppendLine($"  \"vram_mb\": {SystemInfo.graphicsMemorySize},");
            sb.AppendLine($"  \"steps\": {(rs.Count > 0 ? rs[0].passos : 0)},");
            sb.AppendLine("  \"results\": [");
            for (int i = 0; i < rs.Count; i++)
            {
                var r = rs[i];
                sb.Append("    {");
                sb.Append($"\"n\": {r.n}, ");
                sb.Append($"\"us_per_step\": {(r.gpuMsPorPasso * 1000).ToString("F3", ci)}, ");
                sb.Append($"\"updates_per_s\": {r.atualizacoesPorSegundo.ToString("F0", ci)}, ");
                sb.Append($"\"readback_ms\": {r.readbackMs.ToString("F3", ci)}, ");
                sb.Append($"\"state_mb\": {r.vramMB.ToString("F2", ci)}, ");
                sb.Append($"\"spikes_last_step\": {r.spikesTotal}");
                sb.AppendLine(i < rs.Count - 1 ? "}," : "}");
            }
            sb.AppendLine("  ]");
            sb.AppendLine("}");

            var dir = Path.Combine(Application.dataPath, "..", "..", "..",
                                   "benchmarks", "neural");
            dir = Path.GetFullPath(dir);
            Directory.CreateDirectory(dir);
            var caminho = Path.Combine(
                dir, $"lif_unity_compute_{DateTime.Now:yyyy-MM-dd_HHmmss}.json");
            File.WriteAllText(caminho, sb.ToString());
            Debug.Log($"[lif-bench] salvo em {caminho}");
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
