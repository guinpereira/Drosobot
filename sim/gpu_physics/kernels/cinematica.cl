// Cinematica direta do NeuroMechFly, um nivel da arvore por vez.
//
// OBRA DERIVADA. Traducao de C para OpenCL de:
//     MuJoCo, src/engine/engine_core_smooth.c   mj_kinematics1, mj_kinematics2
//     MuJoCo, src/engine/engine_util_spatial.c  mju_mulQuat, mju_rotVecQuat,
//                                               mju_quat2Mat, mju_axisAngle2Quat
//     MuJoCo, src/engine/engine_util_blas.c     mju_normalize4, mju_mulMatVec3
// Copyright 2021 DeepMind Technologies Limited. Licenciado sob a Apache License,
// Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>.
//
// MUDANCAS em relacao ao original, como a licenca pede que se declare:
//   * percurso serial da arvore substituido por percurso em NIVEIS paralelos
//   * quadros dos corpos mantidos em `__local` durante a arvore inteira
//   * laco residente num unico work-group, com barreiras em vez de despachos
//   * `mjtNum` vira `real`, escolhido em tempo de compilacao (fp64 ou fp32)
// Nada disso existe no original, que e serial por construcao. Ver
// THIRD_PARTY_NOTICES.md.
// A ordem das operacoes de ponto flutuante segue o
// original de proposito: `mju_mulQuat`, `mju_rotVecQuat` e `mju_quat2Mat` estao
// reproduzidos termo a termo, com os mesmos atalhos para quaternio identidade e
// vetor nulo. Mudar a ordem daria o mesmo resultado matematico e um resultado
// numerico diferente, e ai a comparacao com a referencia mediria o porte em vez
// da fisica.
//
// O laco do MuJoCo e serial sobre os corpos (`parentid < id` garante a ordem).
// Aqui ele e serial sobre NIVEIS e paralelo dentro de cada nivel: 10 etapas em
// vez de 70. Ver `estrutura.py`.
//
// Duas formas do mesmo codigo, de proposito:
//
//   `fk_nivel` / `fk_quadros` / `fk_geoms`   um dispatch por etapa. O teste de
//       equivalencia para em cada uma e compara com o campo correspondente do
//       `mjData`. Serve para achar ONDE diverge.
//   `fk_completa`                            a arvore inteira num dispatch so,
//       um work-group, barreiras por dentro. E o caminho do laco quente, e o
//       que se mede contra os 94 us/passo do MuJoCo CPU.
//
// Um unico work-group na versao residente. Nao e desperdicio por descuido: o
// modelo tem nv=72, e o que decide este problema e latencia, nao vazao. Medido
// nesta placa: um despacho sincrono custa 84 us, uma barreira de work-group
// custa 0,035 us. Mais work-groups exigiriam sincronizacao global, que custa um
// despacho.

#ifdef USA_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
#define REAL_ZERO 0.0
#define REAL_ONE 1.0
#define REAL_HALF 0.5
#define REAL_TWO 2.0
#else
typedef float real;
#define REAL_ZERO 0.0f
#define REAL_ONE 1.0f
#define REAL_HALF 0.5f
#define REAL_TWO 2.0f
#endif

#define MINVAL ((real)1e-15)

// mjJNT_*
#define JNT_FREE 0
#define JNT_SLIDE 2
#define JNT_HINGE 3

// mjtSameFrame
#define SF_NONE 0
#define SF_BODY 1
#define SF_INERTIA 2
#define SF_BODYROT 3
#define SF_INERTIAROT 4

// --------------------------------------------------------------- primitivas

static void quat_mul(real res[4], const real qa[4], const real qb[4]) {
    real t0 = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3];
    real t1 = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2];
    real t2 = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1];
    real t3 = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0];
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3;
}

static void rot_vec_quat(real res[3], const real vec[3], const real quat[4]) {
    if (vec[0] == REAL_ZERO && vec[1] == REAL_ZERO && vec[2] == REAL_ZERO) {
        res[0] = REAL_ZERO; res[1] = REAL_ZERO; res[2] = REAL_ZERO;
        return;
    }
    if (quat[0] == REAL_ONE && quat[1] == REAL_ZERO
        && quat[2] == REAL_ZERO && quat[3] == REAL_ZERO) {
        res[0] = vec[0]; res[1] = vec[1]; res[2] = vec[2];
        return;
    }
    real t0 = quat[0]*vec[0] + quat[2]*vec[2] - quat[3]*vec[1];
    real t1 = quat[0]*vec[1] + quat[3]*vec[0] - quat[1]*vec[2];
    real t2 = quat[0]*vec[2] + quat[1]*vec[1] - quat[2]*vec[0];
    res[0] = vec[0] + REAL_TWO*(quat[2]*t2 - quat[3]*t1);
    res[1] = vec[1] + REAL_TWO*(quat[3]*t0 - quat[1]*t2);
    res[2] = vec[2] + REAL_TWO*(quat[1]*t1 - quat[2]*t0);
}

static void quat2mat(real res[9], const real quat[4]) {
    if (quat[0] == REAL_ONE && quat[1] == REAL_ZERO
        && quat[2] == REAL_ZERO && quat[3] == REAL_ZERO) {
        res[0] = REAL_ONE; res[1] = REAL_ZERO; res[2] = REAL_ZERO;
        res[3] = REAL_ZERO; res[4] = REAL_ONE; res[5] = REAL_ZERO;
        res[6] = REAL_ZERO; res[7] = REAL_ZERO; res[8] = REAL_ONE;
        return;
    }
    real q00 = quat[0]*quat[0], q01 = quat[0]*quat[1];
    real q02 = quat[0]*quat[2], q03 = quat[0]*quat[3];
    real q11 = quat[1]*quat[1], q12 = quat[1]*quat[2];
    real q13 = quat[1]*quat[3], q22 = quat[2]*quat[2];
    real q23 = quat[2]*quat[3], q33 = quat[3]*quat[3];
    res[0] = q00 + q11 - q22 - q33;
    res[4] = q00 - q11 + q22 - q33;
    res[8] = q00 - q11 - q22 + q33;
    res[1] = REAL_TWO*(q12 - q03);
    res[2] = REAL_TWO*(q13 + q02);
    res[3] = REAL_TWO*(q12 + q03);
    res[5] = REAL_TWO*(q23 - q01);
    res[6] = REAL_TWO*(q13 - q02);
    res[7] = REAL_TWO*(q23 + q01);
}

static void axis_angle2quat(real res[4], const real axis[3], real angle) {
    if (angle == REAL_ZERO) {
        res[0] = REAL_ONE; res[1] = REAL_ZERO;
        res[2] = REAL_ZERO; res[3] = REAL_ZERO;
        return;
    }
    real s = sin(angle*REAL_HALF);
    res[0] = cos(angle*REAL_HALF);
    res[1] = axis[0]*s; res[2] = axis[1]*s; res[3] = axis[2]*s;
}

static void normalize4(real v[4]) {
    real norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]);
    if (norm < MINVAL) {
        v[0] = REAL_ONE; v[1] = REAL_ZERO; v[2] = REAL_ZERO; v[3] = REAL_ZERO;
    } else if (fabs(norm - REAL_ONE) > MINVAL) {
        real inv = REAL_ONE/norm;
        v[0] *= inv; v[1] *= inv; v[2] *= inv; v[3] *= inv;
    }
}

static void mat_vec3(real res[3], __global const real* mat, const real vec[3]) {
    real t0 = mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2];
    real t1 = mat[3]*vec[0] + mat[4]*vec[1] + mat[5]*vec[2];
    real t2 = mat[6]*vec[0] + mat[7]*vec[1] + mat[8]*vec[2];
    res[0] = t0; res[1] = t1; res[2] = t2;
}

// ---------------------------------------------------------------- um corpo

// `mj_kinematics1` para um corpo. Todo argumento e o array inteiro: a versao
// por nivel e a residente passam exatamente os mesmos ponteiros, e assim nao
// existem duas copias da conta que possam divergir.
static void fk_um_corpo(
    int i,
    __global const int* body_parentid, __global const int* body_jntadr,
    __global const int* body_jntnum, __global const int* body_mocapid,
    __global const real* body_pos, __global const real* body_quat,
    __global const int* jnt_type, __global const int* jnt_qposadr,
    __global const real* jnt_axis, __global const real* jnt_pos,
    __global const real* qpos0,
    __global const real* qpos, __global const real* mocap_pos,
    __global const real* mocap_quat,
    __global real* xpos, __global real* xquat, __global real* xmat,
    __global real* xanchor, __global real* xaxis)
{
    real px[3], pq[4];
    int jntadr = body_jntadr[i];
    int jntnum = body_jntnum[i];

    if (jntnum == 1 && jnt_type[jntadr] == JNT_FREE) {
        int qadr = jnt_qposadr[jntadr];
        px[0] = qpos[qadr]; px[1] = qpos[qadr+1]; px[2] = qpos[qadr+2];
        pq[0] = qpos[qadr+3]; pq[1] = qpos[qadr+4];
        pq[2] = qpos[qadr+5]; pq[3] = qpos[qadr+6];
        normalize4(pq);
        xanchor[3*jntadr+0] = px[0];
        xanchor[3*jntadr+1] = px[1];
        xanchor[3*jntadr+2] = px[2];
        xaxis[3*jntadr+0] = jnt_axis[3*jntadr+0];
        xaxis[3*jntadr+1] = jnt_axis[3*jntadr+1];
        xaxis[3*jntadr+2] = jnt_axis[3*jntadr+2];
    } else {
        int pid = body_parentid[i];
        real bpos[3], bquat[4];
        int mid = body_mocapid[i];
        if (mid >= 0) {
            bpos[0] = mocap_pos[3*mid]; bpos[1] = mocap_pos[3*mid+1];
            bpos[2] = mocap_pos[3*mid+2];
            bquat[0] = mocap_quat[4*mid]; bquat[1] = mocap_quat[4*mid+1];
            bquat[2] = mocap_quat[4*mid+2]; bquat[3] = mocap_quat[4*mid+3];
            normalize4(bquat);
        } else {
            bpos[0] = body_pos[3*i]; bpos[1] = body_pos[3*i+1];
            bpos[2] = body_pos[3*i+2];
            bquat[0] = body_quat[4*i]; bquat[1] = body_quat[4*i+1];
            bquat[2] = body_quat[4*i+2]; bquat[3] = body_quat[4*i+3];
        }
        if (pid) {
            mat_vec3(px, xmat + 9*pid, bpos);
            px[0] += xpos[3*pid]; px[1] += xpos[3*pid+1]; px[2] += xpos[3*pid+2];
            real pqp[4] = {xquat[4*pid], xquat[4*pid+1],
                           xquat[4*pid+2], xquat[4*pid+3]};
            quat_mul(pq, pqp, bquat);
        } else {
            px[0] = bpos[0]; px[1] = bpos[1]; px[2] = bpos[2];
            pq[0] = bquat[0]; pq[1] = bquat[1];
            pq[2] = bquat[2]; pq[3] = bquat[3];
        }

        for (int j = 0; j < jntnum; ++j) {
            int jid = jntadr + j;
            int qadr = jnt_qposadr[jid];
            int jtype = jnt_type[jid];
            real ax[3], anc[3];
            real jax[3] = {jnt_axis[3*jid], jnt_axis[3*jid+1], jnt_axis[3*jid+2]};
            real jps[3] = {jnt_pos[3*jid], jnt_pos[3*jid+1], jnt_pos[3*jid+2]};
            rot_vec_quat(ax, jax, pq);
            rot_vec_quat(anc, jps, pq);
            anc[0] += px[0]; anc[1] += px[1]; anc[2] += px[2];

            if (jtype == JNT_SLIDE) {
                real d = qpos[qadr] - qpos0[qadr];
                px[0] += ax[0]*d; px[1] += ax[1]*d; px[2] += ax[2]*d;
            } else {                          // HINGE; BALL esta fora do subset
                real qloc[4];
                axis_angle2quat(qloc, jax, qpos[qadr] - qpos0[qadr]);
                quat_mul(pq, pq, qloc);
                real vec[3];
                rot_vec_quat(vec, jps, pq);
                px[0] = anc[0] - vec[0];
                px[1] = anc[1] - vec[1];
                px[2] = anc[2] - vec[2];
            }
            xanchor[3*jid+0] = anc[0];
            xanchor[3*jid+1] = anc[1];
            xanchor[3*jid+2] = anc[2];
            xaxis[3*jid+0] = ax[0];
            xaxis[3*jid+1] = ax[1];
            xaxis[3*jid+2] = ax[2];
        }
    }

    normalize4(pq);
    xquat[4*i] = pq[0]; xquat[4*i+1] = pq[1];
    xquat[4*i+2] = pq[2]; xquat[4*i+3] = pq[3];
    xpos[3*i] = px[0]; xpos[3*i+1] = px[1]; xpos[3*i+2] = px[2];
    real mm[9];
    quat2mat(mm, pq);
    for (int k = 0; k < 9; ++k) xmat[9*i+k] = mm[k];
}

// `mj_local2Global` para o quadro inercial de um corpo.
static void fk_um_quadro(
    int i,
    __global const real* body_ipos, __global const real* body_iquat,
    __global const int* body_sameframe,
    __global const real* xpos, __global const real* xquat,
    __global const real* xmat,
    __global real* xipos, __global real* ximat)
{
    int sf = body_sameframe[i];
    if (sf == SF_BODY) {
        xipos[3*i] = xpos[3*i];
        xipos[3*i+1] = xpos[3*i+1];
        xipos[3*i+2] = xpos[3*i+2];
    } else {
        real p[3] = {body_ipos[3*i], body_ipos[3*i+1], body_ipos[3*i+2]};
        real r[3];
        mat_vec3(r, xmat + 9*i, p);
        xipos[3*i] = r[0] + xpos[3*i];
        xipos[3*i+1] = r[1] + xpos[3*i+1];
        xipos[3*i+2] = r[2] + xpos[3*i+2];
    }
    if (sf == SF_BODY || sf == SF_BODYROT) {
        for (int k = 0; k < 9; ++k) ximat[9*i+k] = xmat[9*i+k];
    } else {
        real q[4] = {body_iquat[4*i], body_iquat[4*i+1],
                     body_iquat[4*i+2], body_iquat[4*i+3]};
        real bq[4] = {xquat[4*i], xquat[4*i+1], xquat[4*i+2], xquat[4*i+3]};
        real tmp[4], mm[9];
        quat_mul(tmp, bq, q);
        quat2mat(mm, tmp);
        for (int k = 0; k < 9; ++k) ximat[9*i+k] = mm[k];
    }
}

// `mj_local2Global` para um geom.
static void fk_um_geom(
    int g,
    __global const int* geom_bodyid, __global const real* geom_pos,
    __global const real* geom_quat, __global const int* geom_sameframe,
    __global const real* xpos, __global const real* xquat,
    __global const real* xmat, __global const real* xipos,
    __global const real* ximat,
    __global real* geom_xpos, __global real* geom_xmat)
{
    int b = geom_bodyid[g];
    int sf = geom_sameframe[g];

    if (sf == SF_BODY) {
        geom_xpos[3*g] = xpos[3*b];
        geom_xpos[3*g+1] = xpos[3*b+1];
        geom_xpos[3*g+2] = xpos[3*b+2];
    } else if (sf == SF_INERTIA) {
        geom_xpos[3*g] = xipos[3*b];
        geom_xpos[3*g+1] = xipos[3*b+1];
        geom_xpos[3*g+2] = xipos[3*b+2];
    } else {
        real p[3] = {geom_pos[3*g], geom_pos[3*g+1], geom_pos[3*g+2]};
        real r[3];
        mat_vec3(r, xmat + 9*b, p);
        geom_xpos[3*g] = r[0] + xpos[3*b];
        geom_xpos[3*g+1] = r[1] + xpos[3*b+1];
        geom_xpos[3*g+2] = r[2] + xpos[3*b+2];
    }

    if (sf == SF_BODY || sf == SF_BODYROT) {
        for (int k = 0; k < 9; ++k) geom_xmat[9*g+k] = xmat[9*b+k];
    } else if (sf == SF_INERTIA || sf == SF_INERTIAROT) {
        for (int k = 0; k < 9; ++k) geom_xmat[9*g+k] = ximat[9*b+k];
    } else {
        real q[4] = {geom_quat[4*g], geom_quat[4*g+1],
                     geom_quat[4*g+2], geom_quat[4*g+3]};
        real bq[4] = {xquat[4*b], xquat[4*b+1], xquat[4*b+2], xquat[4*b+3]};
        real tmp[4], mm[9];
        quat_mul(tmp, bq, q);
        quat2mat(mm, tmp);
        for (int k = 0; k < 9; ++k) geom_xmat[9*g+k] = mm[k];
    }
}

// ------------------------------------------------------ um dispatch por etapa

__kernel void fk_nivel(
    __global const int* nivel_corpos, const int nivel_adr, const int nivel_num,
    __global const int* body_parentid, __global const int* body_jntadr,
    __global const int* body_jntnum, __global const int* body_mocapid,
    __global const real* body_pos, __global const real* body_quat,
    __global const int* jnt_type, __global const int* jnt_qposadr,
    __global const real* jnt_axis, __global const real* jnt_pos,
    __global const real* qpos0,
    __global const real* qpos, __global const real* mocap_pos,
    __global const real* mocap_quat,
    __global real* xpos, __global real* xquat, __global real* xmat,
    __global real* xanchor, __global real* xaxis)
{
    int t = get_global_id(0);
    if (t >= nivel_num) return;
    fk_um_corpo(nivel_corpos[nivel_adr + t], body_parentid, body_jntadr,
                body_jntnum, body_mocapid, body_pos, body_quat,
                jnt_type, jnt_qposadr, jnt_axis, jnt_pos, qpos0,
                qpos, mocap_pos, mocap_quat,
                xpos, xquat, xmat, xanchor, xaxis);
}

__kernel void fk_quadros(
    const int nbody,
    __global const real* body_ipos, __global const real* body_iquat,
    __global const int* body_sameframe,
    __global const real* xpos, __global const real* xquat,
    __global const real* xmat,
    __global real* xipos, __global real* ximat)
{
    int i = get_global_id(0);
    if (i < 1 || i >= nbody) return;
    fk_um_quadro(i, body_ipos, body_iquat, body_sameframe,
                 xpos, xquat, xmat, xipos, ximat);
}

__kernel void fk_geoms(
    const int ngeom,
    __global const int* geom_bodyid, __global const real* geom_pos,
    __global const real* geom_quat, __global const int* geom_sameframe,
    __global const real* xpos, __global const real* xquat,
    __global const real* xmat, __global const real* xipos,
    __global const real* ximat,
    __global real* geom_xpos, __global real* geom_xmat)
{
    int g = get_global_id(0);
    if (g >= ngeom) return;
    fk_um_geom(g, geom_bodyid, geom_pos, geom_quat, geom_sameframe,
               xpos, xquat, xmat, xipos, ximat, geom_xpos, geom_xmat);
}

// ------------------------------------------------- residente, um dispatch so

// `repeticoes` existe para medir: uma corrida com K repeticoes e outra com 1
// separam o custo do dispatch do custo da conta. Nao tem uso no laco real, que
// passa 1.
__kernel void fk_completa(
    const int nbody, const int ngeom, const int profundidade,
    __global const int* nivel_adr, __global const int* nivel_num,
    __global const int* nivel_corpos,
    __global const int* body_parentid, __global const int* body_jntadr,
    __global const int* body_jntnum, __global const int* body_mocapid,
    __global const real* body_pos, __global const real* body_quat,
    __global const real* body_ipos, __global const real* body_iquat,
    __global const int* body_sameframe,
    __global const int* jnt_type, __global const int* jnt_qposadr,
    __global const real* jnt_axis, __global const real* jnt_pos,
    __global const real* qpos0,
    __global const int* geom_bodyid, __global const real* geom_pos,
    __global const real* geom_quat, __global const int* geom_sameframe,
    __global const real* qpos, __global const real* mocap_pos,
    __global const real* mocap_quat,
    __global real* xpos, __global real* xquat, __global real* xmat,
    __global real* xanchor, __global real* xaxis,
    __global real* xipos, __global real* ximat,
    __global real* geom_xpos, __global real* geom_xmat,
    const int repeticoes)
{
    int t = get_local_id(0);
    int W = get_local_size(0);

    for (int rep = 0; rep < repeticoes; ++rep) {
        for (int d = 1; d < profundidade; ++d) {
            int adr = nivel_adr[d], num = nivel_num[d];
            for (int k = t; k < num; k += W) {
                fk_um_corpo(nivel_corpos[adr + k], body_parentid, body_jntadr,
                            body_jntnum, body_mocapid, body_pos, body_quat,
                            jnt_type, jnt_qposadr, jnt_axis, jnt_pos, qpos0,
                            qpos, mocap_pos, mocap_quat,
                            xpos, xquat, xmat, xanchor, xaxis);
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        for (int i = 1 + t; i < nbody; i += W) {
            fk_um_quadro(i, body_ipos, body_iquat, body_sameframe,
                         xpos, xquat, xmat, xipos, ximat);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (int g = t; g < ngeom; g += W) {
            fk_um_geom(g, geom_bodyid, geom_pos, geom_quat, geom_sameframe,
                       xpos, xquat, xmat, xipos, ximat, geom_xpos, geom_xmat);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

// ------------------------------------------- residente com estado em __local
//
// A versao `fk_completa` acima mede 32 us (fp64) contra 6,87 us do
// `mj_kinematics` na CPU. O diagnostico nao e a conta -- sao 70 corpos, umas
// poucas centenas de flops cada. E a MEMORIA: cada nivel escreve
// `xpos/xquat/xmat` em memoria global e o nivel seguinte le de volta, com
// `barrier(CLK_GLOBAL_MEM_FENCE)` no meio. Dez viagens ate a VRAM num problema
// que cabe inteiro em 16 KiB.
//
// Esta versao mantem os quadros dos corpos em `__local` durante a arvore toda e
// so escreve em global no fim. NBODY/NJNT/NGEOM entram por `-D` no build,
// porque `__local` de tamanho variavel exige passar o tamanho por argumento e
// perde o endereco estatico.
//
//     (3 + 4 + 9 + 3 + 9) * NBODY * sizeof(real)
//
// Para a mosca (NBODY=70, fp64): 15,7 KiB dos 64 KiB do work-group.

#if defined(NBODY) && defined(NJNT) && defined(NGEOM)

#define L_XPOS(i)  (l_xpos + 3*(i))
#define L_XQUAT(i) (l_xquat + 4*(i))
#define L_XMAT(i)  (l_xmat + 9*(i))

// `fk_um_corpo` com os quadros em __local. E a mesma conta, termo a termo; o
// que muda e de onde o pai e lido e para onde o resultado vai.
static void fk_corpo_lds(
    int i,
    __global const int* body_parentid, __global const int* body_jntadr,
    __global const int* body_jntnum, __global const int* body_mocapid,
    __global const real* body_pos, __global const real* body_quat,
    __global const int* jnt_type, __global const int* jnt_qposadr,
    __global const real* jnt_axis, __global const real* jnt_pos,
    __global const real* qpos0,
    __global const real* qpos, __global const real* mocap_pos,
    __global const real* mocap_quat,
    __local real* l_xpos, __local real* l_xquat, __local real* l_xmat,
    __global real* xanchor, __global real* xaxis)
{
    real px[3], pq[4];
    int jntadr = body_jntadr[i];
    int jntnum = body_jntnum[i];

    if (jntnum == 1 && jnt_type[jntadr] == JNT_FREE) {
        int qadr = jnt_qposadr[jntadr];
        px[0] = qpos[qadr]; px[1] = qpos[qadr+1]; px[2] = qpos[qadr+2];
        pq[0] = qpos[qadr+3]; pq[1] = qpos[qadr+4];
        pq[2] = qpos[qadr+5]; pq[3] = qpos[qadr+6];
        normalize4(pq);
        xanchor[3*jntadr+0] = px[0];
        xanchor[3*jntadr+1] = px[1];
        xanchor[3*jntadr+2] = px[2];
        xaxis[3*jntadr+0] = jnt_axis[3*jntadr+0];
        xaxis[3*jntadr+1] = jnt_axis[3*jntadr+1];
        xaxis[3*jntadr+2] = jnt_axis[3*jntadr+2];
    } else {
        int pid = body_parentid[i];
        real bpos[3], bquat[4];
        int mid = body_mocapid[i];
        if (mid >= 0) {
            bpos[0] = mocap_pos[3*mid]; bpos[1] = mocap_pos[3*mid+1];
            bpos[2] = mocap_pos[3*mid+2];
            bquat[0] = mocap_quat[4*mid]; bquat[1] = mocap_quat[4*mid+1];
            bquat[2] = mocap_quat[4*mid+2]; bquat[3] = mocap_quat[4*mid+3];
            normalize4(bquat);
        } else {
            bpos[0] = body_pos[3*i]; bpos[1] = body_pos[3*i+1];
            bpos[2] = body_pos[3*i+2];
            bquat[0] = body_quat[4*i]; bquat[1] = body_quat[4*i+1];
            bquat[2] = body_quat[4*i+2]; bquat[3] = body_quat[4*i+3];
        }
        if (pid) {
            __local const real* pm = L_XMAT(pid);
            real t0 = pm[0]*bpos[0] + pm[1]*bpos[1] + pm[2]*bpos[2];
            real t1 = pm[3]*bpos[0] + pm[4]*bpos[1] + pm[5]*bpos[2];
            real t2 = pm[6]*bpos[0] + pm[7]*bpos[1] + pm[8]*bpos[2];
            px[0] = t0 + L_XPOS(pid)[0];
            px[1] = t1 + L_XPOS(pid)[1];
            px[2] = t2 + L_XPOS(pid)[2];
            real pqp[4] = {L_XQUAT(pid)[0], L_XQUAT(pid)[1],
                           L_XQUAT(pid)[2], L_XQUAT(pid)[3]};
            quat_mul(pq, pqp, bquat);
        } else {
            px[0] = bpos[0]; px[1] = bpos[1]; px[2] = bpos[2];
            pq[0] = bquat[0]; pq[1] = bquat[1];
            pq[2] = bquat[2]; pq[3] = bquat[3];
        }

        for (int j = 0; j < jntnum; ++j) {
            int jid = jntadr + j;
            int qadr = jnt_qposadr[jid];
            int jtype = jnt_type[jid];
            real ax[3], anc[3];
            real jax[3] = {jnt_axis[3*jid], jnt_axis[3*jid+1], jnt_axis[3*jid+2]};
            real jps[3] = {jnt_pos[3*jid], jnt_pos[3*jid+1], jnt_pos[3*jid+2]};
            rot_vec_quat(ax, jax, pq);
            rot_vec_quat(anc, jps, pq);
            anc[0] += px[0]; anc[1] += px[1]; anc[2] += px[2];

            if (jtype == JNT_SLIDE) {
                real dd = qpos[qadr] - qpos0[qadr];
                px[0] += ax[0]*dd; px[1] += ax[1]*dd; px[2] += ax[2]*dd;
            } else {
                real qloc[4];
                axis_angle2quat(qloc, jax, qpos[qadr] - qpos0[qadr]);
                quat_mul(pq, pq, qloc);
                real vec[3];
                rot_vec_quat(vec, jps, pq);
                px[0] = anc[0] - vec[0];
                px[1] = anc[1] - vec[1];
                px[2] = anc[2] - vec[2];
            }
            xanchor[3*jid+0] = anc[0];
            xanchor[3*jid+1] = anc[1];
            xanchor[3*jid+2] = anc[2];
            xaxis[3*jid+0] = ax[0];
            xaxis[3*jid+1] = ax[1];
            xaxis[3*jid+2] = ax[2];
        }
    }

    normalize4(pq);
    real mm[9];
    quat2mat(mm, pq);
    L_XQUAT(i)[0] = pq[0]; L_XQUAT(i)[1] = pq[1];
    L_XQUAT(i)[2] = pq[2]; L_XQUAT(i)[3] = pq[3];
    L_XPOS(i)[0] = px[0]; L_XPOS(i)[1] = px[1]; L_XPOS(i)[2] = px[2];
    for (int k = 0; k < 9; ++k) L_XMAT(i)[k] = mm[k];
}

__kernel void fk_lds(
    const int profundidade,
    __global const int* nivel_adr, __global const int* nivel_num,
    __global const int* nivel_corpos,
    __global const int* body_parentid, __global const int* body_jntadr,
    __global const int* body_jntnum, __global const int* body_mocapid,
    __global const real* body_pos, __global const real* body_quat,
    __global const real* body_ipos, __global const real* body_iquat,
    __global const int* body_sameframe,
    __global const int* jnt_type, __global const int* jnt_qposadr,
    __global const real* jnt_axis, __global const real* jnt_pos,
    __global const real* qpos0,
    __global const int* geom_bodyid, __global const real* geom_pos,
    __global const real* geom_quat, __global const int* geom_sameframe,
    __global const real* qpos, __global const real* mocap_pos,
    __global const real* mocap_quat,
    __global real* xpos, __global real* xquat, __global real* xmat,
    __global real* xanchor, __global real* xaxis,
    __global real* xipos, __global real* ximat,
    __global real* geom_xpos, __global real* geom_xmat,
    const int repeticoes)
{
    __local real l_xpos[3*NBODY];
    __local real l_xquat[4*NBODY];
    __local real l_xmat[9*NBODY];
    __local real l_xipos[3*NBODY];
    __local real l_ximat[9*NBODY];

    int t = get_local_id(0);
    int W = get_local_size(0);

    // o mundo: identidade, uma vez por dispatch
    if (t == 0) {
        l_xpos[0] = l_xpos[1] = l_xpos[2] = REAL_ZERO;
        l_xipos[0] = l_xipos[1] = l_xipos[2] = REAL_ZERO;
        l_xquat[0] = REAL_ONE;
        l_xquat[1] = l_xquat[2] = l_xquat[3] = REAL_ZERO;
        for (int k = 0; k < 9; ++k) {
            l_xmat[k] = (k == 0 || k == 4 || k == 8) ? REAL_ONE : REAL_ZERO;
            l_ximat[k] = l_xmat[k];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int rep = 0; rep < repeticoes; ++rep) {
        for (int dd = 1; dd < profundidade; ++dd) {
            int adr = nivel_adr[dd], num = nivel_num[dd];
            for (int k = t; k < num; k += W) {
                fk_corpo_lds(nivel_corpos[adr + k], body_parentid, body_jntadr,
                             body_jntnum, body_mocapid, body_pos, body_quat,
                             jnt_type, jnt_qposadr, jnt_axis, jnt_pos, qpos0,
                             qpos, mocap_pos, mocap_quat,
                             l_xpos, l_xquat, l_xmat, xanchor, xaxis);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // quadros inerciais, direto em __local
        for (int i = 1 + t; i < NBODY; i += W) {
            int sf = body_sameframe[i];
            if (sf == SF_BODY) {
                l_xipos[3*i] = l_xpos[3*i];
                l_xipos[3*i+1] = l_xpos[3*i+1];
                l_xipos[3*i+2] = l_xpos[3*i+2];
            } else {
                real p[3] = {body_ipos[3*i], body_ipos[3*i+1], body_ipos[3*i+2]};
                __local const real* mm = L_XMAT(i);
                l_xipos[3*i]   = mm[0]*p[0] + mm[1]*p[1] + mm[2]*p[2] + l_xpos[3*i];
                l_xipos[3*i+1] = mm[3]*p[0] + mm[4]*p[1] + mm[5]*p[2] + l_xpos[3*i+1];
                l_xipos[3*i+2] = mm[6]*p[0] + mm[7]*p[1] + mm[8]*p[2] + l_xpos[3*i+2];
            }
            if (sf == SF_BODY || sf == SF_BODYROT) {
                for (int k = 0; k < 9; ++k) l_ximat[9*i+k] = l_xmat[9*i+k];
            } else {
                real q[4] = {body_iquat[4*i], body_iquat[4*i+1],
                             body_iquat[4*i+2], body_iquat[4*i+3]};
                real bq[4] = {L_XQUAT(i)[0], L_XQUAT(i)[1],
                              L_XQUAT(i)[2], L_XQUAT(i)[3]};
                real tmp[4], mm[9];
                quat_mul(tmp, bq, q);
                quat2mat(mm, tmp);
                for (int k = 0; k < 9; ++k) l_ximat[9*i+k] = mm[k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // geoms: leem de __local, escrevem em global (o consumidor e a colisao)
        for (int g = t; g < NGEOM; g += W) {
            int b = geom_bodyid[g];
            int sf = geom_sameframe[g];
            if (sf == SF_BODY) {
                geom_xpos[3*g] = l_xpos[3*b];
                geom_xpos[3*g+1] = l_xpos[3*b+1];
                geom_xpos[3*g+2] = l_xpos[3*b+2];
            } else if (sf == SF_INERTIA) {
                geom_xpos[3*g] = l_xipos[3*b];
                geom_xpos[3*g+1] = l_xipos[3*b+1];
                geom_xpos[3*g+2] = l_xipos[3*b+2];
            } else {
                real p[3] = {geom_pos[3*g], geom_pos[3*g+1], geom_pos[3*g+2]};
                __local const real* mm = L_XMAT(b);
                geom_xpos[3*g]   = mm[0]*p[0] + mm[1]*p[1] + mm[2]*p[2] + l_xpos[3*b];
                geom_xpos[3*g+1] = mm[3]*p[0] + mm[4]*p[1] + mm[5]*p[2] + l_xpos[3*b+1];
                geom_xpos[3*g+2] = mm[6]*p[0] + mm[7]*p[1] + mm[8]*p[2] + l_xpos[3*b+2];
            }
            if (sf == SF_BODY || sf == SF_BODYROT) {
                for (int k = 0; k < 9; ++k) geom_xmat[9*g+k] = l_xmat[9*b+k];
            } else if (sf == SF_INERTIA || sf == SF_INERTIAROT) {
                for (int k = 0; k < 9; ++k) geom_xmat[9*g+k] = l_ximat[9*b+k];
            } else {
                real q[4] = {geom_quat[4*g], geom_quat[4*g+1],
                             geom_quat[4*g+2], geom_quat[4*g+3]};
                real bq[4] = {L_XQUAT(b)[0], L_XQUAT(b)[1],
                              L_XQUAT(b)[2], L_XQUAT(b)[3]};
                real tmp[4], mm[9];
                quat_mul(tmp, bq, q);
                quat2mat(mm, tmp);
                for (int k = 0; k < 9; ++k) geom_xmat[9*g+k] = mm[k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // quadros dos corpos vao pra global no fim: quem le sao a telemetria e o
    // teste, nao o proximo estagio da fisica
    for (int i = t; i < NBODY; i += W) {
        for (int k = 0; k < 3; ++k) {
            xpos[3*i+k] = l_xpos[3*i+k];
            xipos[3*i+k] = l_xipos[3*i+k];
        }
        for (int k = 0; k < 4; ++k) xquat[4*i+k] = l_xquat[4*i+k];
        for (int k = 0; k < 9; ++k) {
            xmat[9*i+k] = l_xmat[9*i+k];
            ximat[9*i+k] = l_ximat[9*i+k];
        }
    }
}

#endif  // NBODY && NJNT && NGEOM
