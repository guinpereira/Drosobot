/*
 * Reducoes de leitura: o que a CPU recebe do cerebro.
 *
 * Com 164.249 neuronios, copiar o estado inteiro por passo desmonta o ganho de
 * manter tudo na GPU. Entao a CPU nunca le o cerebro: ela le
 *
 *   - a contagem de spike por neuronio (acumulada aqui, lida por janela);
 *   - a soma por GRUPO, pros neuronios sem morfologia 3D;
 *   - o estado de um subconjunto explicito (os com geometria + o selecionado).
 *
 * Tudo aqui e instrumentacao, e fica fora do caminho quente quando ninguem
 * esta olhando -- mesmo principio de fast_lif.roda().
 */

__kernel void acumula_contagem(
    __global const uchar *spike,
    __global int *contagem,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    if (spike[i]) contagem[i] += 1;
}

/* Atividade de populacao: grupo -1 = neuronio fora de qualquer grupo. */
__kernel void soma_por_grupo(
    __global const uchar *spike,
    __global const int   *grupo,
    __global int         *soma,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    if (spike[i] == 0) return;
    int gid = grupo[i];
    if (gid >= 0) atomic_add(&soma[gid], 1);
}

/* Estado de um subconjunto pequeno. O unico caminho de leitura por passo. */
__kernel void coleta_indices(
    __global const float *v,
    __global const float *g,
    __global const uchar *spike,
    __global const int   *indices,
    __global float *v_out,
    __global float *g_out,
    __global uchar *spike_out,
    const int n_sel)
{
    int k = get_global_id(0);
    if (k >= n_sel) return;
    int i = indices[k];
    v_out[k] = v[i];
    g_out[k] = g[i];
    spike_out[k] = spike[i];
}
