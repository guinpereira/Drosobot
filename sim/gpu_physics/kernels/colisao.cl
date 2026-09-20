// Colisao plano x malha, pelos pares explicitos do modelo.
//
// OBRA DERIVADA. Traducao de C para OpenCL de:
//     MuJoCo 3.9.0, src/engine/engine_collision_convex.c
//         mjc_PlaneConvex, addplanemesh, mjc_meshSupport
//
// A VERSAO IMPORTA. O clone em research/upstream/mujoco estava em 3.13.1,
// e o `mjc_PlaneConvex` de la usa faces poligonais e poda por area
// (`hull4f`) -- algoritmo que NAO existe no 3.9.0, que e o binario que roda
// os experimentos. O primeiro porte saiu do 3.13.1 e batia o contato mais
// profundo bit a bit enquanto errava os contatos extras, que e o sintoma
// exato de portar a fisica certa da versao errada. Ver `compilador.valida`,
// que agora recusa um runtime cuja versao nao seja a portada.
// Copyright 2021 DeepMind Technologies Limited. Apache License, Version 2.0
// <https://www.apache.org/licenses/LICENSE-2.0>. Ver THIRD_PARTY_NOTICES.md.
//
// MUDANCAS em relacao ao original:
//   * a funcao de suporte e por BUSCA EXAUSTIVA, com uma reducao de argmax num
//     work-group, e nao por hill climbing sobre o grafo da malha. Para uma
//     malha convexa o argmax e o mesmo vertice; o que muda e o caminho ate
//     ele, e a reducao paralela tem forma de GPU enquanto o hill climbing e
//     uma cadeia de leituras dependentes. O indice LOCAL no grafo, que o
//     original obtem de graca (`obj.meshindex`), vem aqui de uma tabela
//     global->local construida no setup
//   * os contatos sao escritos em ranhuras FIXAS por par (3 por par) e depois
//     compactados por uma varredura em ordem de par. O original os emite num
//     laco serial; escrever com atomico daria uma ordem nao determinista, e a
//     ordem dos contatos e a ordem das linhas de restricao
//   * `mjtNum` vira `real`
//
// Concatenado depois de `cinematica.cl`.
//
// ## Por que so plano x malha
//
// Porque e o que o modelo usa. Medido em `mjmodel_inventario.json`: nas arenas
// de looming e optomotor os 55 pares sao TODOS plano x malha. A arena de
// obstaculos acrescenta 55 pares cilindro x malha, que passam por
// `mjc_Convex` (GJK/EPA) no original e NAO estao implementados aqui -- o
// compilador recusa o modelo, em vez de devolver zero contatos em silencio e
// deixar a mosca atravessar o pilar.

#if defined(NPAIR_MAX) && defined(MAX_CON_POR_PAR)

#define GEOM_PLANE 0
#define GEOM_MESH 7

// ------------------------------------------------------------- suporte
//
// Um work-group por par: argmax de `dot(vertice_local, dir_local)` sobre os
// vertices da malha. `dir` e a normal do plano invertida.
//
// O empate e resolvido pelo MENOR indice, como no laco do original
// (`if (vdot > max)`, estritamente maior). A reducao preserva isso comparando
// o indice quando os valores empatam.
__kernel void suporte_plano_malha(
    const int npair,
    __global const int* pair_geom1, __global const int* pair_geom2,
    __global const int* geom_type, __global const int* geom_dataid,
    __global const real* geom_xpos, __global const real* geom_xmat,
    __global const real* geom_rbound, __global const real* pair_margin,
    __global const int* mesh_vertadr, __global const int* mesh_vertnum,
    __global const float* mesh_vert,
    __global int* sup_vert, __global real* sup_dist)
{
    __local real red_v[256];
    __local int red_i[256];

    int p = get_group_id(0);
    if (p >= npair) return;
    int t = get_local_id(0), W = get_local_size(0);

    int g1 = pair_geom1[p], g2 = pair_geom2[p];
    // o plano e sempre o primeiro no par, porque o compilador do MuJoCo ordena
    // por tipo; a checagem existe para falhar alto se isso mudar
    if (geom_type[g1] != GEOM_PLANE || geom_type[g2] != GEOM_MESH) {
        if (t == 0) { sup_vert[p] = -1; sup_dist[p] = (real)1e30; }
        return;
    }

    real normal[3] = {geom_xmat[9*g1+2], geom_xmat[9*g1+5], geom_xmat[9*g1+8]};
    real margin = pair_margin[p];

    // rejeicao por esfera envolvente: exata, e evita a varredura na maioria dos
    // pares. O ponto mais fundo possivel esta a `centro - rbound` do plano.
    real cen[3] = {geom_xpos[3*g2] - geom_xpos[3*g1],
                   geom_xpos[3*g2+1] - geom_xpos[3*g1+1],
                   geom_xpos[3*g2+2] - geom_xpos[3*g1+2]};
    real cdist = normal[0]*cen[0] + normal[1]*cen[1] + normal[2]*cen[2];
    if (cdist - geom_rbound[g2] > margin) {
        if (t == 0) { sup_vert[p] = -1; sup_dist[p] = (real)1e30; }
        return;
    }

    // dir em coordenadas locais da malha: mat2^T * (-normal)
    __global const real* mat2 = geom_xmat + 9*g2;
    real ld[3] = {
        -(mat2[0]*normal[0] + mat2[3]*normal[1] + mat2[6]*normal[2]),
        -(mat2[1]*normal[0] + mat2[4]*normal[1] + mat2[7]*normal[2]),
        -(mat2[2]*normal[0] + mat2[5]*normal[1] + mat2[8]*normal[2])};

    int did = geom_dataid[g2];
    int vadr = mesh_vertadr[did], nvert = mesh_vertnum[did];
    real best = -(real)1e30;
    int besti = 0;
    for (int i = t; i < nvert; i += W) {
        __global const float* v = mesh_vert + 3*(vadr + i);
        real dv = ld[0]*(real)v[0] + ld[1]*(real)v[1] + ld[2]*(real)v[2];
        if (dv > best || (dv == best && i < besti)) { best = dv; besti = i; }
    }
    red_v[t] = best; red_i[t] = besti;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = W >> 1; s > 0; s >>= 1) {
        if (t < s) {
            if (red_v[t+s] > red_v[t]
                || (red_v[t+s] == red_v[t] && red_i[t+s] < red_i[t])) {
                red_v[t] = red_v[t+s];
                red_i[t] = red_i[t+s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t != 0) return;

    int imax = red_i[0];
    __global const float* v = mesh_vert + 3*(vadr + imax);
    real lv[3] = {(real)v[0], (real)v[1], (real)v[2]};
    // localToGlobal
    real gv[3] = {
        mat2[0]*lv[0] + mat2[1]*lv[1] + mat2[2]*lv[2] + geom_xpos[3*g2],
        mat2[3]*lv[0] + mat2[4]*lv[1] + mat2[5]*lv[2] + geom_xpos[3*g2+1],
        mat2[6]*lv[0] + mat2[7]*lv[1] + mat2[8]*lv[2] + geom_xpos[3*g2+2]};
    real dif[3] = {gv[0] - geom_xpos[3*g1],
                   gv[1] - geom_xpos[3*g1+1],
                   gv[2] - geom_xpos[3*g1+2]};
    sup_dist[p] = normal[0]*dif[0] + normal[1]*dif[1] + normal[2]*dif[2];
    sup_vert[p] = imax;
}

// --------------------------------------------------------------- contatos
//
// O 3.9.0 NAO usa faces: depois do ponto de suporte, ele varre os VIZINHOS
// desse vertice no grafo de hull da malha e aceita os que estao abaixo do
// limiar, ate `maxplanemesh` contatos no total.
//
//     threshold = dot(normal, pos2 - pos1) - margin
//     aceita se dot(locdir, v_local) > threshold
//
// que e equivalente a "distancia do vertice ao plano menor que a margem" --
// mas escrito assim, na mesma ordem de operacoes, porque a comparacao com a
// referencia e no ultimo bit.
//
// `addplanemesh` ainda descarta o que estiver perto demais do primeiro
// contato: `dist3(pnt, first) < 0.3 * rbound`.

#define MAXPLANEMESH 3
#define TOLPLANEMESH ((real)0.3)

// Uma thread por par. O trabalho e a varredura dos vizinhos do vertice de
// suporte -- algumas dezenas de leituras, sem paralelismo util dentro.
__kernel void contatos_plano_malha(
    const int npair,
    __global const int* pair_geom1, __global const int* pair_geom2,
    __global const int* geom_dataid, __global const real* geom_rbound,
    __global const real* geom_xpos, __global const real* geom_xmat,
    __global const real* pair_margin,
    __global const int* mesh_vertadr, __global const float* mesh_vert,
    __global const int* mesh_graphadr, __global const int* mesh_graph,
    __global const int* vert_local,
    __global const int* sup_vert, __global const real* sup_dist,
    __global int* n_por_par, __global real* con_dist, __global real* con_pos,
    __global real* con_normal)
{
    int p = get_global_id(0);
    if (p >= npair) return;
    n_por_par[p] = 0;

    int vi = sup_vert[p];
    if (vi < 0) return;
    real dist = sup_dist[p];
    real margin = pair_margin[p];
    if (dist > margin) return;

    int g1 = pair_geom1[p], g2 = pair_geom2[p];
    real normal[3] = {geom_xmat[9*g1+2], geom_xmat[9*g1+5], geom_xmat[9*g1+8]};
    __global const real* mat2 = geom_xmat + 9*g2;
    __global const real* pos2 = geom_xpos + 3*g2;
    __global const real* pos1 = geom_xpos + 3*g1;
    int did = geom_dataid[g2];
    int vadr = mesh_vertadr[did];

    int base = MAX_CON_POR_PAR * p;
    __global const float* sv = mesh_vert + 3*(vadr + vi);
    real lv[3] = {(real)sv[0], (real)sv[1], (real)sv[2]};
    real first[3] = {mat2[0]*lv[0] + mat2[1]*lv[1] + mat2[2]*lv[2] + pos2[0],
                     mat2[3]*lv[0] + mat2[4]*lv[1] + mat2[5]*lv[2] + pos2[1],
                     mat2[6]*lv[0] + mat2[7]*lv[1] + mat2[8]*lv[2] + pos2[2]};
    con_dist[base] = dist;
    for (int k = 0; k < 3; ++k) {
        con_pos[3*base+k] = first[k] - (real)0.5*dist*normal[k];
        con_normal[3*base+k] = normal[k];
    }
    int count = 1;

    int ga = mesh_graphadr[did];
    int meshindex = vert_local[vadr + vi];
    if (ga >= 0 && meshindex >= 0) {
        // locdir = mat2^T * (-normal)
        real ld[3] = {
            -(mat2[0]*normal[0] + mat2[3]*normal[1] + mat2[6]*normal[2]),
            -(mat2[1]*normal[0] + mat2[4]*normal[1] + mat2[7]*normal[2]),
            -(mat2[2]*normal[0] + mat2[5]*normal[1] + mat2[8]*normal[2])};
        real dif[3] = {pos2[0]-pos1[0], pos2[1]-pos1[1], pos2[2]-pos1[2]};
        real threshold = normal[0]*dif[0] + normal[1]*dif[1] + normal[2]*dif[2]
                         - margin;
        real rb = geom_rbound[g2];

        int numvert = mesh_graph[ga];
        __global const int* vert_edgeadr = mesh_graph + ga + 2;
        __global const int* vert_globalid = mesh_graph + ga + 2 + numvert;
        __global const int* edge_localid = mesh_graph + ga + 2 + 2*numvert;

        int i = vert_edgeadr[meshindex];
        int locid;
        while ((locid = edge_localid[i]) >= 0 && count < MAXPLANEMESH) {
            int gid = vert_globalid[locid];
            __global const float* vv = mesh_vert + 3*(vadr + gid);
            real vdot = ld[0]*(real)vv[0] + ld[1]*(real)vv[1] + ld[2]*(real)vv[2];
            if (vdot > threshold) {
                real l[3] = {(real)vv[0], (real)vv[1], (real)vv[2]};
                real pnt[3] = {
                    mat2[0]*l[0] + mat2[1]*l[1] + mat2[2]*l[2] + pos2[0],
                    mat2[3]*l[0] + mat2[4]*l[1] + mat2[5]*l[2] + pos2[1],
                    mat2[6]*l[0] + mat2[7]*l[1] + mat2[8]*l[2] + pos2[2]};
                real dd[3] = {pnt[0]-first[0], pnt[1]-first[1], pnt[2]-first[2]};
                real sep = sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
                if (sep >= TOLPLANEMESH*rb) {
                    real df[3] = {pnt[0]-pos1[0], pnt[1]-pos1[1], pnt[2]-pos1[2]};
                    real vd = normal[0]*df[0] + normal[1]*df[1] + normal[2]*df[2];
                    int s = base + count;
                    con_dist[s] = vd;
                    for (int k = 0; k < 3; ++k) {
                        con_pos[3*s+k] = pnt[k] - (real)0.5*vd*normal[k];
                        con_normal[3*s+k] = normal[k];
                    }
                    count++;
                }
            }
            i++;
        }
    }
    n_por_par[p] = count;
}

// Compactacao em ordem de PAR. Serial de proposito: sao 55 pares, e a ordem
// dos contatos e a ordem das linhas de restricao -- um scan com atomico daria
// uma ordem que muda de corrida para corrida.
__kernel void compacta_contatos(
    const int npair,
    __global const int* n_por_par, __global const real* con_dist,
    __global const real* con_pos, __global const real* con_normal,
    __global const int* pair_geom1, __global const int* pair_geom2,
    __global real* out_dist, __global real* out_pos, __global real* out_normal,
    __global int* out_geom, __global int* out_pair, __global int* ncon)
{
    if (get_global_id(0) != 0) return;
    int n = 0;
    for (int p = 0; p < npair; ++p) {
        int k = n_por_par[p];
        for (int i = 0; i < k; ++i) {
            int s = MAX_CON_POR_PAR * p + i;
            out_dist[n] = con_dist[s];
            for (int c = 0; c < 3; ++c) {
                out_pos[3*n+c] = con_pos[3*s+c];
                out_normal[3*n+c] = con_normal[3*s+c];
            }
            out_geom[2*n] = pair_geom1[p];
            out_geom[2*n+1] = pair_geom2[p];
            out_pair[n] = p;
            n++;
        }
    }
    ncon[0] = n;
}

#endif  // NPAIR_MAX && MAX_CON_POR_PAR
