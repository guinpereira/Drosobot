/*
 * Propagacao sinaptica sobre o CSR do conectoma.
 *
 * Um work-item por NEURONIO; quem disparou percorre suas arestas e soma no
 * alvo. Estrategia DENSE.
 *
 * A alternativa -- compactar antes a lista de quem disparou (spike frontier) --
 * foi implementada e MEDIDA, e perdeu em todas as taxas de disparo testadas
 * (0,1% a 100%): a compactacao mais o dispatch indireto custam mais do que
 * economizam nesta escala. Ver docs/research/WHOLE_CNS_GPU_FEASIBILITY.md.
 * Contra-intuitivo, e e por isso que as duas foram escritas em vez de uma
 * escolhida no papel.
 *
 * `weights` ja vem COM SINAL, pela regra de Dale aplicada ao neurotransmissor
 * do pre-sinaptico (sim/connectome_model.py). O kernel nao conhece
 * neurotransmissor -- essa e uma regra cientifica e mora no host.
 *
 * Acumulacao em ponto fixo: OpenCL garante atomic_add em int, nao em float.
 */
__kernel void scatter(
    __global const uchar *spike,
    __global const int   *row_offsets,
    __global const int   *targets,
    __global const float *weights,
    __global int         *anel,
    const int n,
    const int cursor,
    const float escala)
{
    int i = get_global_id(0);
    if (i >= n) return;
    if (spike[i] == 0) return;

    int base = cursor * n;
    int ini = row_offsets[i];
    int fim = row_offsets[i + 1];
    for (int e = ini; e < fim; e++)
    {
        atomic_add(&anel[base + targets[e]], (int)(weights[e] * escala));
    }
}
