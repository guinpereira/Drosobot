/*
 * LIF exato -- o coracao do modelo de Shiu et al. 2024 (Nature 634:210).
 *
 * Este kernel e NOSSO. OpenCL fornece device, memoria e dispatch; a equacao,
 * a ordem das operacoes e o tratamento do refratario sao do Drosobot, e sao os
 * mesmos de sim/fast_lif.py, que e validado contra o Brian2.
 *
 *     u      = v - V_REST
 *     u_novo = u * dec_v + g * acopla
 *     g      = g * dec_g + chega
 *     livre  = passo >= ref_ate
 *     v      = livre ? V_REST + u_novo : V_RESET
 *     spike  = livre && v > V_TH
 *     se spike: v = V_RESET, ref_ate = passo + ref_passos
 *
 * A ordem NAO pode ser reorganizada, nem por equivalencia algebrica: o teste de
 * equivalencia compara contra a referencia em fp64 e qualquer reassociacao muda
 * o ultimo bit, o que em evento discreto vira spike a mais ou a menos.
 *
 * Os coeficientes chegam prontos, calculados em fp64 no host (model.py).
 * Recalcula-los aqui em fp32 daria numeros diferentes dos da referencia.
 *
 * ## Spike forcado: a populacao de entrada e POISSON, nao LIF
 *
 * No modelo (e em sim/fast_lif.py) os sensores nao integram: disparam com
 * probabilidade `taxa_hz * dt_s` e entregam o peso sinaptico cheio. Quem sorteia
 * e o host -- converter taxa em probabilidade e decisao de MODELO e fica em
 * Python; o kernel so recebe a mascara pronta.
 *
 * Isso importa: injetar CORRENTE continua no lugar do spike discreto e outro
 * modelo, e mais fraco. Foi o que quebrou o primeiro laco fechado -- os
 * LC4/LPLC2 nunca cruzavam o limiar e o Giant Fiber nunca disparava.
 */
__kernel void lif_step(
    __global float *v,
    __global float *g,
    __global int   *ref_ate,
    __global uchar *spike,
    __global int   *anel,           // D * N, ponto fixo (ver delay.cl)
    __global const float *externo,  // corrente externa em mV, por neuronio
    __global const uchar *forcado,  // 1 = dispara neste passo (Poisson do host)
    const int    n,
    const int    passo,
    const int    cursor,
    const int    ref_passos,
    const float  v_rest,
    const float  v_reset,
    const float  v_th,
    const float  dec_v,
    const float  dec_g,
    const float  acopla,
    const float  inv_escala,
    // Ultimo argumento de proposito: acrescentar no fim nao mexe no indice de
    // nenhum outro, e os indices estao fixados no host (ver `_fixa_args`).
    __global int *n_frontier)
{
    int i = get_global_id(0);

    // Zera o contador da frontier aqui, de graca. Um `enqueue_fill_buffer` so
    // pra isso custaria mais um lancamento por passo (~19 us medidos), e fazer
    // no inicio do `compacta_spikes` seria corrida com os incrementos. Aqui e
    // seguro: o LIF termina antes de o compacta comecar, porque a fila e em
    // ordem. Fica ANTES do recorte por `n` -- com o tamanho global arredondado
    // pra cima, o item 0 existe sempre.
    if (i == 0) n_frontier[0] = 0;

    if (i >= n) return;

    // o que chega agora vem do anel de atraso, e o slot e liberado pra reuso
    int slot = cursor * n + i;
    float chega = (float)anel[slot] * inv_escala + externo[i];
    anel[slot] = 0;

    float vi = v[i];
    float gi = g[i];
    int refi = ref_ate[i];

    float u = vi - v_rest;
    float u_novo = u * dec_v + gi * acopla;
    gi = gi * dec_g + chega;

    bool livre = (passo >= refi);
    vi = livre ? (v_rest + u_novo) : v_reset;

    // select em vez de ramo: o wave nao diverge entre quem disparou e quem nao
    // O spike forcado ignora o refratario de proposito: a fonte Poisson nao tem
    // um, e dar refratario a ela mudaria a taxa efetiva do estimulo.
    bool disparou = (livre && (vi > v_th)) || (forcado[i] != (uchar)0);
    vi = disparou ? v_reset : vi;
    refi = disparou ? (passo + ref_passos) : refi;

    v[i] = vi;
    g[i] = gi;
    ref_ate[i] = refi;
    spike[i] = disparou ? (uchar)1 : (uchar)0;
}


// ---------------------------------------------------------------- forcados
//
// Escreve a mascara de spike forcado SO nos indices que podem disparar.
//
// A mascara tem um byte por neuronio -- 164.451 no Male CNS inteiro -- mas so
// os 311 sensores de looming mudam de valor. Copiar a mascara inteira a cada
// passo custava 164 KB de transferencia e, pior, o `enqueue_copy` do pyopencl
// e BLOQUEANTE por padrao: o host parava 844 us por passo esperando uma
// transferencia que a GPU fazia em 75 us. Era 64% do relogio neural.
//
// Com este kernel sobem 311 bytes e o device escreve nos lugares certos. O
// conteudo final da mascara e IDENTICO -- os outros 164.140 bytes ja eram zero
// e continuam zero, porque so estes indices sao escritos, sempre.
__kernel void escreve_forcados_esparso(__global uchar *forcado,
                                       __global const int *idx,
                                       __global const uchar *val,
                                       const int k)
{
    int i = get_global_id(0);
    if (i >= k) return;
    forcado[idx[i]] = val[i];
}
