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
 */
__kernel void lif_step(
    __global float *v,
    __global float *g,
    __global int   *ref_ate,
    __global uchar *spike,
    __global int   *anel,           // D * N, ponto fixo (ver delay.cl)
    __global const float *externo,  // entrada sensorial em mV, por neuronio
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
    const float  inv_escala)
{
    int i = get_global_id(0);
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
    bool disparou = livre && (vi > v_th);
    vi = disparou ? v_reset : vi;
    refi = disparou ? (passo + ref_passos) : refi;

    v[i] = vi;
    g[i] = gi;
    ref_ate[i] = refi;
    spike[i] = disparou ? (uchar)1 : (uchar)0;
}
