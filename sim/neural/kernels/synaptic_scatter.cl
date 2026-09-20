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


/* ---------------------------------------------------------------- frontier
 *
 * O `scatter` acima da uma thread por NEURONIO. Com 1.500 disparos em 164.451
 * neuronios, 99,1% das threads saem na primeira linha e as que sobram ficam
 * desbalanceadas: o grau de saida vai de 0 a 11.203, entao um hub que dispara
 * e percorrido por UMA thread enquanto 20 CUs esperam. Medido: 859 us de
 * atomics contra 19 us de piso (lancamento + varredura).
 *
 * Estes dois kernels dao uma thread por ARESTA, em dois passos:
 *
 *   compacta_spikes   junta os indices de quem disparou numa lista curta
 *   scatter_frontier  distribui as arestas dessa lista entre as threads
 *
 * O RESULTADO E IDENTICO, bit a bit. A soma e feita com `atomic_add` em INT, e
 * adicao inteira e associativa e comutativa -- a ordem em que as parcelas
 * chegam nao muda o total. (Seria diferente com float, onde a ordem importa;
 * e mais uma razao pro acumulador ser ponto fixo.)
 */

/* Largura: quantas threads dividem as arestas de um mesmo neuronio. 32 e a
 * wavefront da RDNA2; com isso o hub de 11.203 arestas vira 350 iteracoes por
 * thread em vez de 11.203. */
#define LARGURA_FRONTIER 32

__kernel void compacta_spikes(
    __global const uchar *spike,
    __global int         *frontier,
    __global int         *n_frontier,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    if (spike[i] == 0) return;
    int pos = atomic_inc(n_frontier);
    frontier[pos] = i;
}

__kernel void scatter_frontier(
    __global const int   *frontier,
    __global const int   *n_frontier,
    __global const int   *row_offsets,
    __global const int   *targets,
    __global const float *weights,
    __global int         *anel,
    const int n,
    const int cursor,
    const float escala)
{
    /* Tamanho de lancamento FIXO e laco com passo largo: descobrir quantos
     * dispararam exigiria trazer `n_frontier` pra CPU, e uma sincronizacao por
     * passo custaria mais que o desbalanceamento que viemos corrigir. */
    const int nf = n_frontier[0];
    const int total = nf * LARGURA_FRONTIER;
    const int passo_global = get_global_size(0);
    const int base = cursor * n;

    for (int j = get_global_id(0); j < total; j += passo_global)
    {
        int s = j / LARGURA_FRONTIER;
        int w = j - s * LARGURA_FRONTIER;
        int i = frontier[s];
        int fim = row_offsets[i + 1];
        for (int e = row_offsets[i] + w; e < fim; e += LARGURA_FRONTIER)
        {
            atomic_add(&anel[base + targets[e]], (int)(weights[e] * escala));
        }
    }
}
