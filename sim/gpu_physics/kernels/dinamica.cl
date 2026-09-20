// Dinamica suave do NeuroMechFly: do quadro cinematico ate `qacc_smooth`.
//
// OBRA DERIVADA. Traducao de C para OpenCL de:
//     MuJoCo, src/engine/engine_core_smooth.c   mj_comPos, mj_crb, mj_factorI,
//                                               mj_solveLD, mj_comVel, mj_rne
//     MuJoCo, src/engine/engine_passive.c       mj_springdamper
//     MuJoCo, src/engine/engine_forward.c       mj_fwdActuation, mj_Euler
//     MuJoCo, src/engine/engine_util_spatial.c  mju_inertCom, mju_dofCom,
//                                               mju_crossMotion, mju_crossForce,
//                                               mju_mulInertVec
// Copyright 2021 DeepMind Technologies Limited. Apache License, Version 2.0
// <https://www.apache.org/licenses/LICENSE-2.0>. Ver THIRD_PARTY_NOTICES.md.
//
// MUDANCAS em relacao ao original, como a licenca pede que se declare:
//   * os lacos seriais sobre corpos viram lacos sobre NIVEIS da arvore, com os
//     corpos de um nivel em paralelo
//   * as acumulacoes para tras (subtree_com, crb, cfrc_body) sao feitas com uma
//     thread por PAI varrendo os filhos dele, e nao uma thread por filho
//     somando no pai -- sem isso dois irmaos escreveriam no mesmo pai, e o
//     OpenCL 1.2 nao tem atomico de ponto flutuante
//   * `mjtNum` vira `real`, escolhido em tempo de compilacao
//
// Este arquivo e concatenado DEPOIS de `cinematica.cl`, que define `real`, as
// primitivas de quaternio e as constantes. Nao ha `#include` em OpenCL, e
// duplicar as primitivas criaria duas copias da mesma conta livres para
// divergir.
//
// ## O que NAO esta aqui
//
// Contato, restricao e solver. Este arquivo produz a dinamica SEM restricao --
// `qacc_smooth` -- que e exatamente o que o MuJoCo chama assim e o que o solver
// recebe como ponto de partida.

#if defined(NBODY) && defined(NV) && defined(NQ)

// mjtTrn / mjtBias, e os tamanhos dos blocos de parametro do atuador
#define TRN_JOINT 0
#define TRN_BODY 5
#define BIAS_NONE 0
#define BIAS_AFFINE 1
#define NGAINPRM 10
#define NBIASPRM 10

// --------------------------------------------------------- algebra espacial

// `mju_mulInertVec`: inercia 10-vetor x movimento 6-vetor
static void mul_inert_vec(real res[6], const real i[10], const real v[6]) {
    res[0] = i[0]*v[0] + i[3]*v[1] + i[4]*v[2] - i[8]*v[4] + i[7]*v[5];
    res[1] = i[3]*v[0] + i[1]*v[1] + i[5]*v[2] + i[8]*v[3] - i[6]*v[5];
    res[2] = i[4]*v[0] + i[5]*v[1] + i[2]*v[2] - i[7]*v[3] + i[6]*v[4];
    res[3] = i[8]*v[1] - i[7]*v[2] + i[9]*v[3];
    res[4] = i[6]*v[2] - i[8]*v[0] + i[9]*v[4];
    res[5] = i[7]*v[0] - i[6]*v[1] + i[9]*v[5];
}

// `mju_crossMotion`
static void cross_motion(real res[6], const real vel[6], const real v[6]) {
    res[0] = -vel[2]*v[1] + vel[1]*v[2];
    res[1] =  vel[2]*v[0] - vel[0]*v[2];
    res[2] = -vel[1]*v[0] + vel[0]*v[1];
    res[3] = -vel[2]*v[4] + vel[1]*v[5];
    res[4] =  vel[2]*v[3] - vel[0]*v[5];
    res[5] = -vel[1]*v[3] + vel[0]*v[4];
    res[3] += -vel[5]*v[1] + vel[4]*v[2];
    res[4] +=  vel[5]*v[0] - vel[3]*v[2];
    res[5] += -vel[4]*v[0] + vel[3]*v[1];
}

// `mju_crossForce`
static void cross_force(real res[6], const real vel[6], const real f[6]) {
    res[0] = -vel[2]*f[1] + vel[1]*f[2];
    res[1] =  vel[2]*f[0] - vel[0]*f[2];
    res[2] = -vel[1]*f[0] + vel[0]*f[1];
    res[3] = -vel[2]*f[4] + vel[1]*f[5];
    res[4] =  vel[2]*f[3] - vel[0]*f[5];
    res[5] = -vel[1]*f[3] + vel[0]*f[4];
    res[0] += -vel[5]*f[4] + vel[4]*f[5];
    res[1] +=  vel[5]*f[3] - vel[3]*f[5];
    res[2] += -vel[4]*f[3] + vel[3]*f[4];
}

static real dot6(__global const real* a, const real b[6]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
         + a[3]*b[3] + a[4]*b[4] + a[5]*b[5];
}

// As condicoes de contorno do mundo, por passo: `cvel[0] = 0`,
// `cacc[0] = -gravidade`, `cfrc_body[0] = 0`.
//
// Isto era tres `enqueue_copy` do host. Alem de contradizer "estado residente",
// a copia do pyopencl e BLOQUEANTE por padrao: eram tres sincronizacoes por
// passo dentro do laco quente.
static void zera_raizes_um(int i,
    const real gx,
    const real gy,
    const real gz,
    __global real* cvel,
    __global real* cacc,
    __global real* cfrc_body)
{
    if (i >= 6) return;
    cvel[i] = REAL_ZERO;
    cfrc_body[i] = REAL_ZERO;
    cacc[i] = (i == 3) ? -gx : ((i == 4) ? -gy : ((i == 5) ? -gz : REAL_ZERO));
}

__kernel void zera_raizes(
    const real gx,
    const real gy,
    const real gz,
    __global real* cvel,
    __global real* cacc,
    __global real* cfrc_body)
{
    zera_raizes_um(get_global_id(0), gx, gy, gz, cvel, cacc, cfrc_body);
}


// ----------------------------------------------------------------- comPos
//
// `mj_comPos` tem tres fases e as tres tem forma diferente na GPU:
//   1. momento por corpo        totalmente paralelo
//   2. acumular no pai          serial sobre NIVEIS, para tras
//   3. normalizar               totalmente paralelo

static void com_momento_um(int i,
    __global const real* xipos,
    __global const real* body_mass,
    __global real* subtree_com)
{
    if (i >= NBODY) return;
    real m = body_mass[i];
    subtree_com[3*i+0] = xipos[3*i+0]*m;
    subtree_com[3*i+1] = xipos[3*i+1]*m;
    subtree_com[3*i+2] = xipos[3*i+2]*m;
}

__kernel void com_momento(
    __global const real* xipos,
    __global const real* body_mass,
    __global real* subtree_com)
{
    com_momento_um(get_global_id(0), xipos, body_mass, subtree_com);
}


// Uma thread por PAI, varrendo os filhos dele. O original faz o contrario --
// uma passada para tras sobre os corpos, cada um somando no pai -- e isso seria
// uma corrida entre irmaos. No torax da mosca sao oito filhos.
static void com_acumula_nivel_um(int t,
    __global const int* pais,
    __global const int* pais_adr,
    __global const int* pais_num,
    __global const int* filhos_adr,
    __global const int* filhos_num,
    __global const int* filhos,
    __global const int* nivel_de,
    const int nivel,
    __global real* subtree_com)
{
    if (t >= pais_num[nivel]) return;
    int p = pais[pais_adr[nivel] + t];
    real acc[3] = {REAL_ZERO, REAL_ZERO, REAL_ZERO};
    int a = filhos_adr[p], n = filhos_num[p];
    for (int k = 0; k < n; ++k) {
        int c = filhos[a + k];
        if (nivel_de[c] != nivel) continue;
        acc[0] += subtree_com[3*c+0];
        acc[1] += subtree_com[3*c+1];
        acc[2] += subtree_com[3*c+2];
    }
    subtree_com[3*p+0] += acc[0];
    subtree_com[3*p+1] += acc[1];
    subtree_com[3*p+2] += acc[2];
}

__kernel void com_acumula_nivel(
    __global const int* pais,
    __global const int* pais_adr,
    __global const int* pais_num,
    __global const int* filhos_adr,
    __global const int* filhos_num,
    __global const int* filhos,
    __global const int* nivel_de,
    const int nivel,
    __global real* subtree_com)
{
    com_acumula_nivel_um(get_global_id(0), pais, pais_adr, pais_num, filhos_adr, filhos_num, filhos, nivel_de, nivel, subtree_com);
}


static void com_normaliza_um(int i,
    __global const real* body_subtreemass,
    __global const real* xipos,
    __global real* subtree_com)
{
    if (i >= NBODY) return;
    real ms = body_subtreemass[i];
    if (ms < MINVAL) {
        subtree_com[3*i+0] = xipos[3*i+0];
        subtree_com[3*i+1] = xipos[3*i+1];
        subtree_com[3*i+2] = xipos[3*i+2];
    } else {
        real inv = REAL_ONE/ms;
        subtree_com[3*i+0] *= inv;
        subtree_com[3*i+1] *= inv;
        subtree_com[3*i+2] *= inv;
    }
}

__kernel void com_normaliza(
    __global const real* body_subtreemass,
    __global const real* xipos,
    __global real* subtree_com)
{
    com_normaliza_um(get_global_id(0), body_subtreemass, xipos, subtree_com);
}


// `mju_inertCom` por corpo e `mju_dofCom` por dof. Os dois sao totalmente
// paralelos: dependem so do quadro cinematico e do subtree_com ja pronto.
static void com_inercia_um(int i,
    __global const int* body_rootid,
    __global const real* body_inertia,
    __global const real* body_mass,
    __global const real* ximat,
    __global const real* xipos,
    __global const real* subtree_com,
    __global real* cinert)
{
    if (i >= NBODY) return;
    if (i == 0) {
        for (int k = 0; k < 10; ++k) cinert[k] = REAL_ZERO;
        return;
    }
    int r = body_rootid[i];
    real dif[3] = {xipos[3*i+0] - subtree_com[3*r+0],
                   xipos[3*i+1] - subtree_com[3*r+1],
                   xipos[3*i+2] - subtree_com[3*r+2]};
    real inert[3] = {body_inertia[3*i], body_inertia[3*i+1], body_inertia[3*i+2]};
    real mass = body_mass[i];
    __global const real* mat = ximat + 9*i;
    real tmp[9] = {mat[0]*inert[0], mat[3]*inert[0], mat[6]*inert[0],
                   mat[1]*inert[1], mat[4]*inert[1], mat[7]*inert[1],
                   mat[2]*inert[2], mat[5]*inert[2], mat[8]*inert[2]};
    __global real* res = cinert + 10*i;
    res[0] = mat[0]*tmp[0] + mat[1]*tmp[3] + mat[2]*tmp[6];
    res[1] = mat[3]*tmp[1] + mat[4]*tmp[4] + mat[5]*tmp[7];
    res[2] = mat[6]*tmp[2] + mat[7]*tmp[5] + mat[8]*tmp[8];
    res[3] = mat[0]*tmp[1] + mat[1]*tmp[4] + mat[2]*tmp[7];
    res[4] = mat[0]*tmp[2] + mat[1]*tmp[5] + mat[2]*tmp[8];
    res[5] = mat[3]*tmp[2] + mat[4]*tmp[5] + mat[5]*tmp[8];
    res[0] += mass*(dif[1]*dif[1] + dif[2]*dif[2]);
    res[1] += mass*(dif[0]*dif[0] + dif[2]*dif[2]);
    res[2] += mass*(dif[0]*dif[0] + dif[1]*dif[1]);
    res[3] -= mass*dif[0]*dif[1];
    res[4] -= mass*dif[0]*dif[2];
    res[5] -= mass*dif[1]*dif[2];
    res[6] = mass*dif[0];
    res[7] = mass*dif[1];
    res[8] = mass*dif[2];
    res[9] = mass;
}

__kernel void com_inercia(
    __global const int* body_rootid,
    __global const real* body_inertia,
    __global const real* body_mass,
    __global const real* ximat,
    __global const real* xipos,
    __global const real* subtree_com,
    __global real* cinert)
{
    com_inercia_um(get_global_id(0), body_rootid, body_inertia, body_mass, ximat, xipos, subtree_com, cinert);
}


// `cdof`: uma thread por JUNTA. Free tem 6 dofs, hinge tem 1.
static void com_cdof_um(int j,
    const int njnt,
    __global const int* jnt_type,
    __global const int* jnt_dofadr,
    __global const int* jnt_bodyid,
    __global const int* body_rootid,
    __global const real* xmat,
    __global const real* xanchor,
    __global const real* xaxis,
    __global const real* subtree_com,
    __global real* cdof)
{
    if (j >= njnt) return;
    int i = jnt_bodyid[j];
    int da = 6*jnt_dofadr[j];
    int r = body_rootid[i];
    real offset[3] = {subtree_com[3*r+0] - xanchor[3*j+0],
                      subtree_com[3*r+1] - xanchor[3*j+1],
                      subtree_com[3*r+2] - xanchor[3*j+2]};
    int jt = jnt_type[j];

    if (jt == JNT_FREE) {
        for (int k = 0; k < 18; ++k) cdof[da+k] = REAL_ZERO;
        cdof[da+3+7*0] = REAL_ONE;
        cdof[da+3+7*1] = REAL_ONE;
        cdof[da+3+7*2] = REAL_ONE;
        // as tres rotacoes: eixos das colunas de xmat, como no caso ball
        for (int k = 0; k < 3; ++k) {
            real axis[3] = {xmat[9*i+k+0], xmat[9*i+k+3], xmat[9*i+k+6]};
            __global real* res = cdof + da + 18 + 6*k;
            res[0] = axis[0]; res[1] = axis[1]; res[2] = axis[2];
            res[3] = axis[1]*offset[2] - axis[2]*offset[1];
            res[4] = axis[2]*offset[0] - axis[0]*offset[2];
            res[5] = axis[0]*offset[1] - axis[1]*offset[0];
        }
    } else if (jt == JNT_SLIDE) {
        __global real* res = cdof + da;
        res[0] = REAL_ZERO; res[1] = REAL_ZERO; res[2] = REAL_ZERO;
        res[3] = xaxis[3*j+0]; res[4] = xaxis[3*j+1]; res[5] = xaxis[3*j+2];
    } else {                                                        // HINGE
        real axis[3] = {xaxis[3*j+0], xaxis[3*j+1], xaxis[3*j+2]};
        __global real* res = cdof + da;
        res[0] = axis[0]; res[1] = axis[1]; res[2] = axis[2];
        res[3] = axis[1]*offset[2] - axis[2]*offset[1];
        res[4] = axis[2]*offset[0] - axis[0]*offset[2];
        res[5] = axis[0]*offset[1] - axis[1]*offset[0];
    }
}

__kernel void com_cdof(
    const int njnt,
    __global const int* jnt_type,
    __global const int* jnt_dofadr,
    __global const int* jnt_bodyid,
    __global const int* body_rootid,
    __global const real* xmat,
    __global const real* xanchor,
    __global const real* xaxis,
    __global const real* subtree_com,
    __global real* cdof)
{
    com_cdof_um(get_global_id(0), njnt, jnt_type, jnt_dofadr, jnt_bodyid, body_rootid, xmat, xanchor, xaxis, subtree_com, cdof);
}


// -------------------------------------------------------------------- CRB

static void crb_inicia_um(int i,
    __global const real* cinert,
    __global real* crb)
{
    if (i >= NBODY) return;
    for (int k = 0; k < 10; ++k) crb[10*i+k] = cinert[10*i+k];
}

__kernel void crb_inicia(
    __global const real* cinert,
    __global real* crb)
{
    crb_inicia_um(get_global_id(0), cinert, crb);
}


// Mesma inversao do `com_acumula_nivel`: uma thread por pai.
// O original pula o corpo 0 (`if (body_parentid[i] > 0)`), entao o mundo nao
// acumula -- por isso `p != 0` aqui.
static void crb_acumula_nivel_um(int t,
    __global const int* pais,
    __global const int* pais_adr,
    __global const int* pais_num,
    __global const int* filhos_adr,
    __global const int* filhos_num,
    __global const int* filhos,
    __global const int* nivel_de,
    const int nivel,
    __global real* crb)
{
    if (t >= pais_num[nivel]) return;
    int p = pais[pais_adr[nivel] + t];
    if (p == 0) return;
    real acc[10];
    for (int k = 0; k < 10; ++k) acc[k] = REAL_ZERO;
    int a = filhos_adr[p], n = filhos_num[p];
    for (int k = 0; k < n; ++k) {
        int c = filhos[a + k];
        if (nivel_de[c] != nivel) continue;
        for (int q = 0; q < 10; ++q) acc[q] += crb[10*c+q];
    }
    for (int k = 0; k < 10; ++k) crb[10*p+k] += acc[k];
}

__kernel void crb_acumula_nivel(
    __global const int* pais,
    __global const int* pais_adr,
    __global const int* pais_num,
    __global const int* filhos_adr,
    __global const int* filhos_num,
    __global const int* filhos,
    __global const int* nivel_de,
    const int nivel,
    __global real* crb)
{
    crb_acumula_nivel_um(get_global_id(0), pais, pais_adr, pais_num, filhos_adr, filhos_num, filhos, nivel_de, nivel, crb);
}


// Montagem de M em CSR. Uma thread por dof; cada uma percorre a cadeia dela
// ate a raiz, que e exatamente o padrao de esparsidade da linha. Sem corrida:
// cada thread escreve so na propria linha.
static void crb_monta_M_um(int i,
    __global const int* M_rownnz,
    __global const int* M_rowadr,
    __global const int* dof_parentid,
    __global const int* dof_bodyid,
    __global const real* dof_armature,
    __global const real* crb,
    __global const real* cdof,
    __global real* M)
{
    if (i >= NV) return;
    int adr = M_rowadr[i];
    int madr = adr + M_rownnz[i] - 1;
    real buf[6];
    real inert[10];
    for (int k = 0; k < 10; ++k) inert[k] = crb[10*dof_bodyid[i]+k];
    real cd[6];
    for (int k = 0; k < 6; ++k) cd[k] = cdof[6*i+k];
    mul_inert_vec(buf, inert, cd);
    // O original zera M antes e depois acumula; aqui a linha e ESCRITA, porque
    // cada thread e dona da linha inteira e nao ha passada de zeragem. Com `+=`
    // sobre um buffer residente, o segundo passo somaria sobre o primeiro -- foi
    // exatamente assim que a trajetoria divergiu para 1e71 no passo 2.
    real extra = dof_armature[i];
    for (int j = i; j >= 0; j = dof_parentid[j]) {
        M[madr--] = extra + dot6(cdof + 6*j, buf);
        extra = REAL_ZERO;
    }
}

__kernel void crb_monta_M(
    __global const int* M_rownnz,
    __global const int* M_rowadr,
    __global const int* dof_parentid,
    __global const int* dof_bodyid,
    __global const real* dof_armature,
    __global const real* crb,
    __global const real* cdof,
    __global real* M)
{
    crb_monta_M_um(get_global_id(0), M_rownnz, M_rowadr, dof_parentid, dof_bodyid, dof_armature, crb, cdof, M);
}


// ------------------------------------------------------------- fatoracao
//
// `mj_factorI`: L'*D*L esparsa, laco PARA TRAS sobre as linhas. Cada linha `k`
// atualiza as linhas dos ancestrais dela, e ancestrais de folhas diferentes se
// cruzam na raiz -- o que torna o laco externo genuinamente serial. O paralelo
// disponivel e o de dentro: `mju_addToScl` sobre `rownnz[i]` elementos, com
// rownnz <= 17 neste modelo.
//
// Fica num work-group so, com barreira entre linhas. Nao e rapido; e correto,
// e a alternativa -- inventar uma ordem de eliminacao propria -- mudaria o
// arredondamento e tiraria a comparacao com a referencia do lugar.
static void factor_M_um(int t, int W,
    __global const int* M_rownnz,
    __global const int* M_rowadr,
    __global const int* M_colind,
    __global const real* M,
    __global real* qLD,
    __global real* qLDiagInv)
{
    
    for (int i = t; i < NC; i += W) qLD[i] = M[i];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int k = NV - 1; k >= 0; --k) {
        int start = M_rowadr[k];
        int diag = M_rownnz[k] - 1;
        int end = start + diag;
        real piv = qLD[end];
        if (piv < MINVAL) piv = MINVAL;
        real invD = REAL_ONE/piv;
        if (t == 0) {
            qLD[end] = piv;
            qLDiagInv[k] = invD;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        // linha i < k:  L(i, 0..) -= L(k, 0..) * L(k,i) * invD
        // Os `i` sao distintos entre si dentro desta linha k, entao uma thread
        // por `adr` nao colide.
        for (int adr = end - 1 - t; adr >= start; adr -= W) {
            int i = M_colind[adr];
            real s = -qLD[adr] * invD;
            int ri = M_rowadr[i], ni = M_rownnz[i];
            for (int q = 0; q < ni; ++q) qLD[ri + q] += s * qLD[start + q];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (int q = t; q < diag; q += W) qLD[start + q] *= invD;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void factor_M(
    __global const int* M_rownnz,
    __global const int* M_rowadr,
    __global const int* M_colind,
    __global const real* M,
    __global real* qLD,
    __global real* qLDiagInv)
{
    factor_M_um(get_local_id(0), get_local_size(0), M_rownnz, M_rowadr, M_colind, M, qLD, qLDiagInv);
}


// `mj_solveLD` para um vetor: x = M^-1 y. Serial nas duas direcoes pelo mesmo
// motivo da fatoracao.
// `mj_solveLD` com os dados em __local.
//
// As duas substituicoes triangulares sao SERIAIS por natureza: a linha `i`
// depende das anteriores, e nenhuma reordenacao preserva a ordem de soma do
// original. O que da para tirar e a LATENCIA: a versao em memoria global
// fazia ~2500 acessos dependentes de ~400 ciclos cada, numa thread so, e
// media 340 us. Com `qLD`, `x` e os indices em LDS, os mesmos 2500 acessos
// custam poucos ciclos.
//
// A ordem das operacoes e identica a da versao global -- este kernel nao muda
// arredondamento, so de onde le.
// `mj_factorI` com a matriz em __local.
//
// A estrutura e IDENTICA a da versao global: mesmo laco para tras sobre as
// linhas, mesma ordem de atualizacao, mesmas somas. So muda de onde le. O laco
// externo e serial por natureza -- a linha `k` atualiza as linhas dos
// ancestrais dela, e ancestrais de folhas diferentes se cruzam na raiz -- e
// dentro de cada linha ha no maximo `rownnz <= 17` elementos, ou seja 16
// threads ativas de 256.
//
// Com tao pouco paralelismo, o que domina e a LATENCIA de cada acesso. Em
// memoria global sao ~20 mil acessos quase todos dependentes.
static void factor_M_lds_um(int t, int W,
    __global const int* M_rownnz, __global const int* M_rowadr,
    __global const int* M_colind, __global const real* M,
    __global real* qLD, __global real* qLDiagInv,
    __local real* l, __local int* lcol, __local int* lnnz, __local int* ladr)
{
    for (int i = t; i < NC; i += W) { l[i] = M[i]; lcol[i] = M_colind[i]; }
    for (int i = t; i < NV; i += W) { lnnz[i] = M_rownnz[i]; ladr[i] = M_rowadr[i]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = NV - 1; k >= 0; --k) {
        int start = ladr[k];
        int diag = lnnz[k] - 1;
        int end = start + diag;
        real piv = l[end];
        if (piv < MINVAL) piv = MINVAL;
        real invD = REAL_ONE/piv;
        if (t == 0) { l[end] = piv; qLDiagInv[k] = invD; }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int adr = end - 1 - t; adr >= start; adr -= W) {
            int i = lcol[adr];
            real sc = -l[adr] * invD;
            int ri = ladr[i], ni = lnnz[i];
            for (int q = 0; q < ni; ++q) l[ri + q] += sc * l[start + q];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int q = t; q < diag; q += W) l[start + q] *= invD;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = t; i < NC; i += W) qLD[i] = l[i];
    barrier(CLK_LOCAL_MEM_FENCE);
}

static void solve_M_lds_um(int t, int W,
    __global const int* M_rownnz, __global const int* M_rowadr,
    __global const int* M_colind, __global const real* qLD,
    __global const real* qLDiagInv, __global const real* y, __global real* x,
    __local real* lqld, __local real* lx, __local real* linv,
    __local int* lnnz, __local int* ladr, __local int* lcol)
{
    for (int i = t; i < NC; i += W) { lqld[i] = qLD[i]; lcol[i] = M_colind[i]; }
    for (int i = t; i < NV; i += W) {
        lx[i] = y[i]; linv[i] = qLDiagInv[i];
        lnnz[i] = M_rownnz[i]; ladr[i] = M_rowadr[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t == 0) {
        for (int i = NV - 1; i >= 0; --i) {
            int start = ladr[i], n = lnnz[i] - 1;
            real xi = lx[i];
            for (int q = 0; q < n; ++q) lx[lcol[start + q]] -= lqld[start + q] * xi;
        }
        for (int i = 0; i < NV; ++i) lx[i] *= linv[i];
        for (int i = 0; i < NV; ++i) {
            int start = ladr[i], n = lnnz[i] - 1;
            real acc = REAL_ZERO;
            for (int q = 0; q < n; ++q) acc += lqld[start + q] * lx[lcol[start + q]];
            lx[i] -= acc;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = t; i < NV; i += W) x[i] = lx[i];
}

static void solve_M_um(int t, int W,
    __global const int* M_rownnz,
    __global const int* M_rowadr,
    __global const int* M_colind,
    __global const real* qLD,
    __global const real* qLDiagInv,
    __global const real* y,
    __global real* x)
{
    if (t != 0) return;
    for (int i = 0; i < NV; ++i) x[i] = y[i];
    // x <- L^-T x   (para tras)
    for (int i = NV - 1; i >= 0; --i) {
        int start = M_rowadr[i], n = M_rownnz[i] - 1;
        real xi = x[i];
        for (int q = 0; q < n; ++q) x[M_colind[start + q]] -= qLD[start + q] * xi;
    }
    // x <- D^-1 x
    for (int i = 0; i < NV; ++i) x[i] *= qLDiagInv[i];
    // x <- L^-1 x   (para frente)
    for (int i = 0; i < NV; ++i) {
        int start = M_rowadr[i], n = M_rownnz[i] - 1;
        real s = REAL_ZERO;
        for (int q = 0; q < n; ++q) s += qLD[start + q] * x[M_colind[start + q]];
        x[i] -= s;
    }
}

__kernel void solve_M(
    __global const int* M_rownnz,
    __global const int* M_rowadr,
    __global const int* M_colind,
    __global const real* qLD,
    __global const real* qLDiagInv,
    __global const real* y,
    __global real* x)
{
    solve_M_um(get_local_id(0), get_local_size(0), M_rownnz, M_rowadr, M_colind, qLD, qLDiagInv, y, x);
}


// ----------------------------------------------------------------- comVel
//
// Serial sobre niveis: `cvel` do filho depende do `cvel` do pai. Dentro do
// nivel, os corpos sao independentes.
static void com_vel_nivel_um(int t,
    __global const int* nivel_corpos,
    __global const int* nivel_adr,
    __global const int* nivel_num,
    const int nivel,
    __global const int* body_parentid,
    __global const int* body_dofadr,
    __global const int* body_dofnum,
    __global const int* body_jntadr,
    __global const int* jnt_type,
    __global const int* dof_jntid,
    __global const real* cdof,
    __global const real* qvel,
    __global real* cvel,
    __global real* cdof_dot)
{
    if (t >= nivel_num[nivel]) return;
    int i = nivel_corpos[nivel_adr[nivel] + t];
    real v[6];
    int p = body_parentid[i];
    for (int k = 0; k < 6; ++k) v[k] = cvel[6*p+k];

    int dofnum = body_dofnum[i];
    int bda = body_dofadr[i];
    for (int j = 0; j < dofnum; ++j) {
        int jt = jnt_type[dof_jntid[bda + j]];
        if (jt == JNT_FREE) {
            // translacoes: cdofdot = 0, velocidade acumula direto
            for (int k = 0; k < 18; ++k) cdof_dot[6*bda + k] = REAL_ZERO;
            for (int c = 0; c < 3; ++c) {
                real q = qvel[bda + j + c];
                for (int k = 0; k < 6; ++k) v[k] += cdof[6*(bda+j+c)+k]*q;
            }
            j += 3;
            // rotacoes: como ball, usando a velocidade ja atualizada
            for (int c = 0; c < 3; ++c) {
                real cd[6], dd[6];
                for (int k = 0; k < 6; ++k) cd[k] = cdof[6*(bda+j+c)+k];
                cross_motion(dd, v, cd);
                for (int k = 0; k < 6; ++k) cdof_dot[6*(bda+j+c)+k] = dd[k];
            }
            for (int c = 0; c < 3; ++c) {
                real q = qvel[bda + j + c];
                for (int k = 0; k < 6; ++k) v[k] += cdof[6*(bda+j+c)+k]*q;
            }
            j += 2;
        } else {
            real cd[6], dd[6];
            for (int k = 0; k < 6; ++k) cd[k] = cdof[6*(bda+j)+k];
            cross_motion(dd, v, cd);
            for (int k = 0; k < 6; ++k) cdof_dot[6*(bda+j)+k] = dd[k];
            real q = qvel[bda + j];
            for (int k = 0; k < 6; ++k) v[k] += cd[k]*q;
        }
    }
    for (int k = 0; k < 6; ++k) cvel[6*i+k] = v[k];
}

__kernel void com_vel_nivel(
    __global const int* nivel_corpos,
    __global const int* nivel_adr,
    __global const int* nivel_num,
    const int nivel,
    __global const int* body_parentid,
    __global const int* body_dofadr,
    __global const int* body_dofnum,
    __global const int* body_jntadr,
    __global const int* jnt_type,
    __global const int* dof_jntid,
    __global const real* cdof,
    __global const real* qvel,
    __global real* cvel,
    __global real* cdof_dot)
{
    com_vel_nivel_um(get_global_id(0), nivel_corpos, nivel_adr, nivel_num, nivel, body_parentid, body_dofadr, body_dofnum, body_jntadr, jnt_type, dof_jntid, cdof, qvel, cvel, cdof_dot);
}


// ---------------------------------------------------------------- passivo
//
// Mola e amortecedor LINEARES. O compilador recusa o modelo se algum termo
// polinomial for nao nulo, entao este atalho e verificado, nao assumido.
// Separado em dois despachos porque `barrier()` so sincroniza DENTRO de um
// work-group: entre work-groups nao ha barreira em OpenCL, e o amortecedor
// precisa ter escrito antes de a mola somar por cima.
static void passivo_amortecedor_um(int i,
    __global const real* qvel,
    __global const real* dof_damping,
    __global real* qfrc_passive)
{
    if (i >= NV) return;
    qfrc_passive[i] = -qvel[i] * dof_damping[i];
}

__kernel void passivo_amortecedor(
    __global const real* qvel,
    __global const real* dof_damping,
    __global real* qfrc_passive)
{
    passivo_amortecedor_um(get_global_id(0), qvel, dof_damping, qfrc_passive);
}


static void passivo_mola_um(int j,
    const int njnt,
    __global const int* jnt_type,
    __global const int* jnt_qposadr,
    __global const int* jnt_dofadr,
    __global const real* jnt_stiffness,
    __global const real* qpos,
    __global const real* qpos_spring,
    __global real* qfrc_passive)
{
    if (j >= njnt) return;
    real k = jnt_stiffness[j];
    if (k == REAL_ZERO) return;
    int jt = jnt_type[j];
    // free e ball nao tem rigidez neste modelo; o compilador garante que so
    // existem free e hinge, e free com stiffness zero.
    if (jt != JNT_HINGE && jt != JNT_SLIDE) return;
    int padr = jnt_qposadr[j], dadr = jnt_dofadr[j];
    real x = qpos[padr] - qpos_spring[padr];
    qfrc_passive[dadr] += -x * k;
}

__kernel void passivo_mola(
    const int njnt,
    __global const int* jnt_type,
    __global const int* jnt_qposadr,
    __global const int* jnt_dofadr,
    __global const real* jnt_stiffness,
    __global const real* qpos,
    __global const real* qpos_spring,
    __global real* qfrc_passive)
{
    passivo_mola_um(get_global_id(0), njnt, jnt_type, jnt_qposadr, jnt_dofadr, jnt_stiffness, qpos, qpos_spring, qfrc_passive);
}


// -------------------------------------------------------------------- RNE
//
// Tres etapas: para frente acumulando aceleracao, para tras acumulando forca,
// e a projecao nos dofs. `flg_acc = 0`: e o termo de bias (Coriolis, centrifuga
// e gravidade), com `qacc` fora da conta.

static void rne_frente_nivel_um(int t,
    __global const int* nivel_corpos,
    __global const int* nivel_adr,
    __global const int* nivel_num,
    const int nivel,
    __global const int* body_parentid,
    __global const int* body_dofadr,
    __global const int* body_dofnum,
    __global const real* cdof_dot,
    __global const real* qvel,
    __global const real* cinert,
    __global const real* cvel,
    __global real* cacc,
    __global real* cfrc_body)
{
    if (t >= nivel_num[nivel]) return;
    int i = nivel_corpos[nivel_adr[nivel] + t];
    int bda = body_dofadr[i], dofnum = body_dofnum[i];
    int p = body_parentid[i];

    real a[6];
    for (int k = 0; k < 6; ++k) a[k] = cacc[6*p+k];
    for (int j = 0; j < dofnum; ++j) {
        real q = qvel[bda + j];
        for (int k = 0; k < 6; ++k) a[k] += cdof_dot[6*(bda+j)+k]*q;
    }
    for (int k = 0; k < 6; ++k) cacc[6*i+k] = a[k];

    real inert[10];
    for (int k = 0; k < 10; ++k) inert[k] = cinert[10*i+k];
    real f[6], v[6], tmp[6], tmp1[6];
    mul_inert_vec(f, inert, a);
    for (int k = 0; k < 6; ++k) v[k] = cvel[6*i+k];
    mul_inert_vec(tmp, inert, v);
    cross_force(tmp1, v, tmp);
    for (int k = 0; k < 6; ++k) cfrc_body[6*i+k] = f[k] + tmp1[k];
}

__kernel void rne_frente_nivel(
    __global const int* nivel_corpos,
    __global const int* nivel_adr,
    __global const int* nivel_num,
    const int nivel,
    __global const int* body_parentid,
    __global const int* body_dofadr,
    __global const int* body_dofnum,
    __global const real* cdof_dot,
    __global const real* qvel,
    __global const real* cinert,
    __global const real* cvel,
    __global real* cacc,
    __global real* cfrc_body)
{
    rne_frente_nivel_um(get_global_id(0), nivel_corpos, nivel_adr, nivel_num, nivel, body_parentid, body_dofadr, body_dofnum, cdof_dot, qvel, cinert, cvel, cacc, cfrc_body);
}


// Para tras, uma thread por pai. O original pula a escrita no mundo
// (`if (j)`), e aqui isso vira `p != 0`.
static void rne_tras_nivel_um(int t,
    __global const int* pais,
    __global const int* pais_adr,
    __global const int* pais_num,
    __global const int* filhos_adr,
    __global const int* filhos_num,
    __global const int* filhos,
    __global const int* nivel_de,
    const int nivel,
    __global real* cfrc_body)
{
    if (t >= pais_num[nivel]) return;
    int p = pais[pais_adr[nivel] + t];
    if (p == 0) return;
    real acc[6];
    for (int k = 0; k < 6; ++k) acc[k] = REAL_ZERO;
    int a = filhos_adr[p], n = filhos_num[p];
    for (int k = 0; k < n; ++k) {
        int c = filhos[a + k];
        if (nivel_de[c] != nivel) continue;
        for (int q = 0; q < 6; ++q) acc[q] += cfrc_body[6*c+q];
    }
    for (int k = 0; k < 6; ++k) cfrc_body[6*p+k] += acc[k];
}

__kernel void rne_tras_nivel(
    __global const int* pais,
    __global const int* pais_adr,
    __global const int* pais_num,
    __global const int* filhos_adr,
    __global const int* filhos_num,
    __global const int* filhos,
    __global const int* nivel_de,
    const int nivel,
    __global real* cfrc_body)
{
    rne_tras_nivel_um(get_global_id(0), pais, pais_adr, pais_num, filhos_adr, filhos_num, filhos, nivel_de, nivel, cfrc_body);
}


static void rne_projeta_um(int i,
    __global const int* dof_bodyid,
    __global const real* cdof,
    __global const real* cfrc_body,
    __global real* qfrc_bias)
{
    if (i >= NV) return;
    real f[6];
    for (int k = 0; k < 6; ++k) f[k] = cfrc_body[6*dof_bodyid[i]+k];
    qfrc_bias[i] = dot6(cdof + 6*i, f);
}

__kernel void rne_projeta(
    __global const int* dof_bodyid,
    __global const real* cdof,
    __global const real* cfrc_body,
    __global real* qfrc_bias)
{
    rne_projeta_um(get_global_id(0), dof_bodyid, cdof, cfrc_body, qfrc_bias);
}


// --------------------------------------------------------------- atuacao
//
// Servo de posicao: `gain=fixed`, `bias=affine`. Para transmissao de JUNTA a
// alavanca e `gear[0]` sobre um dof so, entao o momento e trivial e nao ha
// Jacobiana a montar. A adesao (`trntype=body`) NAO entra aqui: ela age nos
// contatos, e contato ainda nao existe neste arquivo.

static void atuacao_um(int a,
    const int nu,
    __global const int* actuator_trntype,
    __global const int* actuator_trnid,
    __global const int* actuator_biastype,
    __global const real* actuator_gainprm,
    __global const real* actuator_biasprm,
    __global const real* actuator_gear,
    __global const real* actuator_ctrlrange,
    __global const int* actuator_ctrllimited,
    __global const real* actuator_forcerange,
    __global const int* actuator_forcelimited,
    __global const int* jnt_qposadr,
    __global const int* jnt_dofadr,
    __global const real* qpos,
    __global const real* qvel,
    __global const real* ctrl,
    __global real* actuator_length,
    __global real* actuator_velocity,
    __global real* actuator_force,
    __global real* qfrc_actuator)
{
    if (a >= nu) return;
    real len, vel;
    if (actuator_trntype[a] == TRN_BODY) {
        // adesao: `mj_transmission` nao consegue definir comprimento e o zera.
        // O momento dela vem dos contatos, em `adesao_momento`.
        len = REAL_ZERO;
        vel = REAL_ZERO;
    } else if (actuator_trntype[a] == TRN_JOINT) {
        int j = actuator_trnid[2*a];
        real gear = actuator_gear[6*a];
        len = gear * qpos[jnt_qposadr[j]];
        vel = gear * qvel[jnt_dofadr[j]];
    } else {
        actuator_force[a] = REAL_ZERO;
        return;
    }
    actuator_length[a] = len;
    actuator_velocity[a] = vel;

    real c = ctrl[a];
    if (actuator_ctrllimited[a]) {
        real lo = actuator_ctrlrange[2*a], hi = actuator_ctrlrange[2*a+1];
        c = fmin(fmax(c, lo), hi);
    }
    real gain = actuator_gainprm[NGAINPRM*a];          // gaintype fixed
    real f = gain * c;
    if (actuator_biastype[a] == BIAS_AFFINE) {
        __global const real* bp = actuator_biasprm + NBIASPRM*a;
        f += bp[0] + bp[1]*len + bp[2]*vel;
    }
    if (actuator_forcelimited[a]) {
        real lo = actuator_forcerange[2*a], hi = actuator_forcerange[2*a+1];
        f = fmin(fmax(f, lo), hi);
    }
    actuator_force[a] = f;
}

__kernel void atuacao(
    const int nu,
    __global const int* actuator_trntype,
    __global const int* actuator_trnid,
    __global const int* actuator_biastype,
    __global const real* actuator_gainprm,
    __global const real* actuator_biasprm,
    __global const real* actuator_gear,
    __global const real* actuator_ctrlrange,
    __global const int* actuator_ctrllimited,
    __global const real* actuator_forcerange,
    __global const int* actuator_forcelimited,
    __global const int* jnt_qposadr,
    __global const int* jnt_dofadr,
    __global const real* qpos,
    __global const real* qvel,
    __global const real* ctrl,
    __global real* actuator_length,
    __global real* actuator_velocity,
    __global real* actuator_force,
    __global real* qfrc_actuator)
{
    atuacao_um(get_global_id(0), nu, actuator_trntype, actuator_trnid, actuator_biastype, actuator_gainprm, actuator_biasprm, actuator_gear, actuator_ctrlrange, actuator_ctrllimited, actuator_forcerange, actuator_forcelimited, jnt_qposadr, jnt_dofadr, qpos, qvel, ctrl, actuator_length, actuator_velocity, actuator_force, qfrc_actuator);
}


// A projecao e separada porque dois atuadores poderiam, em principio, tocar o
// mesmo dof. Uma thread por DOF varrendo os atuadores evita a corrida sem
// precisar de atomico.
static void atuacao_projeta_um(int i,
    const int nu,
    __global const int* actuator_trntype,
    __global const int* actuator_trnid,
    __global const real* actuator_gear,
    __global const int* jnt_dofadr,
    __global const real* actuator_force,
    __global real* qfrc_actuator)
{
    if (i >= NV) return;
    real s = REAL_ZERO;
    for (int a = 0; a < nu; ++a) {
        if (actuator_trntype[a] != TRN_JOINT) continue;
        if (jnt_dofadr[actuator_trnid[2*a]] != i) continue;
        s += actuator_gear[6*a] * actuator_force[a];
    }
    qfrc_actuator[i] = s;
}

__kernel void atuacao_projeta(
    const int nu,
    __global const int* actuator_trntype,
    __global const int* actuator_trnid,
    __global const real* actuator_gear,
    __global const int* jnt_dofadr,
    __global const real* actuator_force,
    __global real* qfrc_actuator)
{
    atuacao_projeta_um(get_global_id(0), nu, actuator_trntype, actuator_trnid, actuator_gear, jnt_dofadr, actuator_force, qfrc_actuator);
}


// ------------------------------------------------------ dinamica sem contato

static void soma_smooth_um(int i,
    __global const real* qfrc_passive,
    __global const real* qfrc_bias,
    __global const real* qfrc_actuator,
    __global const real* qfrc_applied,
    __global real* qfrc_smooth)
{
    if (i >= NV) return;
    qfrc_smooth[i] = qfrc_passive[i] - qfrc_bias[i]
                   + qfrc_actuator[i] + qfrc_applied[i];
}

__kernel void soma_smooth(
    __global const real* qfrc_passive,
    __global const real* qfrc_bias,
    __global const real* qfrc_actuator,
    __global const real* qfrc_applied,
    __global real* qfrc_smooth)
{
    soma_smooth_um(get_global_id(0), qfrc_passive, qfrc_bias, qfrc_actuator, qfrc_applied, qfrc_smooth);
}


// ------------------------------------------------------------ integracao
//
// `mj_Euler` com amortecimento IMPLICITO: quando ha `dof_damping`, o MuJoCo
// fatora `M + h*diag(damping)` e resolve com ele, em vez de somar a forca de
// amortecimento explicitamente. O termo explicito ja entrou em `qfrc_passive`,
// e o original o subtrai de volta antes de resolver.

static void euler_copia_M_um(int i,
    __global const real* M,
    __global real* qH)
{
    if (i < NC) qH[i] = M[i];
}

__kernel void euler_copia_M(
    __global const real* M,
    __global real* qH)
{
    euler_copia_M_um(get_global_id(0), M, qH);
}


static void euler_diag_MhD_um(int i,
    const real dt,
    __global const int* M_rownnz,
    __global const int* M_rowadr,
    __global const real* dof_damping,
    __global real* qH)
{
    if (i >= NV) return;
    qH[M_rowadr[i] + M_rownnz[i] - 1] += dt * dof_damping[i];
}

__kernel void euler_diag_MhD(
    const real dt,
    __global const int* M_rownnz,
    __global const int* M_rowadr,
    __global const real* dof_damping,
    __global real* qH)
{
    euler_diag_MhD_um(get_global_id(0), dt, M_rownnz, M_rowadr, dof_damping, qH);
}


// `qfrc_smooth + qfrc_constraint`, e so. O amortecimento explicito JA esta em
// `qfrc_smooth` (veio de `qfrc_passive`); o lado implicito e a diagonal de
// `qH`. Somar o amortecimento de novo aqui contaria a mesma forca duas vezes.
static void euler_rhs_um(int i,
    __global const real* qfrc_smooth,
    __global const real* qfrc_constraint,
    __global real* rhs)
{
    if (i >= NV) return;
    rhs[i] = qfrc_smooth[i] + qfrc_constraint[i];
}

__kernel void euler_rhs(
    __global const real* qfrc_smooth,
    __global const real* qfrc_constraint,
    __global real* rhs)
{
    euler_rhs_um(get_global_id(0), qfrc_smooth, qfrc_constraint, rhs);
}


static void euler_qvel_um(int i,
    const real dt,
    __global const real* qacc,
    __global real* qvel)
{
    if (i < NV) qvel[i] += dt * qacc[i];
}

__kernel void euler_qvel(
    const real dt,
    __global const real* qacc,
    __global real* qvel)
{
    euler_qvel_um(get_global_id(0), dt, qacc, qvel);
}


// `mj_integratePos`, com a velocidade JA atualizada -- Euler semi-implicito,
// como no `mj_advance`. Hinge e slide somam direto; free integra a translacao
// e compoe o quaternio com a rotacao do passo.
static void euler_qpos_um(int i,
    const real dt,
    const int njnt,
    __global const int* jnt_type,
    __global const int* jnt_qposadr,
    __global const int* jnt_dofadr,
    __global const real* qvel,
    __global real* qpos)
{
    if (i >= njnt) return;
    int jt = jnt_type[i];
    int padr = jnt_qposadr[i], dadr = jnt_dofadr[i];
    if (jt == JNT_FREE) {
        for (int k = 0; k < 3; ++k) qpos[padr+k] += dt * qvel[dadr+k];
        real w[3] = {qvel[dadr+3]*dt, qvel[dadr+4]*dt, qvel[dadr+5]*dt};
        real ang = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
        real q[4] = {qpos[padr+3], qpos[padr+4], qpos[padr+5], qpos[padr+6]};
        real qres[4];
        if (ang < MINVAL) {
            qres[0] = q[0]; qres[1] = q[1]; qres[2] = q[2]; qres[3] = q[3];
        } else {
            real axis[3] = {w[0]/ang, w[1]/ang, w[2]/ang};
            real qrot[4];
            axis_angle2quat(qrot, axis, ang);
            quat_mul(qres, q, qrot);
        }
        normalize4(qres);
        for (int k = 0; k < 4; ++k) qpos[padr+3+k] = qres[k];
    } else {
        qpos[padr] += dt * qvel[dadr];
    }
}

__kernel void euler_qpos(
    const real dt,
    const int njnt,
    __global const int* jnt_type,
    __global const int* jnt_qposadr,
    __global const int* jnt_dofadr,
    __global const real* qvel,
    __global real* qpos)
{
    euler_qpos_um(get_global_id(0), dt, njnt, jnt_type, jnt_qposadr, jnt_dofadr, qvel, qpos);
}


#endif  // NBODY && NV && NQ
