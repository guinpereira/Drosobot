/*
 * Atraso sinaptico.
 *
 * O modelo usa atraso FIXO (T_DELAY = 1,8 ms). Com dt = 0,5 ms isso vira 4
 * passos inteiros -- o mesmo arredondamento que sim/fast_lif.py (Conexao) ja
 * faz, e por isso os dois concordam sobre QUANDO o sinal chega.
 *
 * Atraso fixo permite uma implementacao muito mais barata que atraso arbitrario:
 * um anel de D buffers de conductancia, D * N floats, em vez de um campo por
 * aresta. No Male CNS inteiro a diferenca e 2,5 MiB contra 24,4 MiB -- e, mais
 * importante, nao acrescenta uma leitura por aresta no caminho quente.
 *
 * O anel NAO tem kernel proprio de avanco: o cursor e aritmetica de indice no
 * host (engine.py), e a leitura/limpeza do slot acontece dentro de lif.cl. Este
 * arquivo existe pra documentar a estrutura e abrigar as operacoes de anel que
 * precisem de kernel -- hoje so a limpeza total, usada no reset.
 */
__kernel void limpa_anel(
    __global int *anel,
    const int total)
{
    int i = get_global_id(0);
    if (i < total) anel[i] = 0;
}
