// Construcao das restricoes de contato: efc_J, efc_pos, efc_R, efc_D, efc_aref.
//
// OBRA DERIVADA. Traducao de C para OpenCL de:
//     MuJoCo 3.9.0, src/engine/engine_core_constraint.c
//         mj_instantiateContact, mj_diagApprox, mj_makeImpedance,
//         getimpedance, mj_referenceConstraint, mj_constraintUpdate_impl
//     MuJoCo 3.9.0, src/engine/engine_core_util.c   mj_jac
//     MuJoCo 3.9.0, src/engine/engine_util_spatial.c  mju_makeFrame
// Copyright 2021 DeepMind Technologies Limited. Apache License, Version 2.0
// <https://www.apache.org/licenses/LICENSE-2.0>. Ver THIRD_PARTY_NOTICES.md.
//
// 3.9.0 e a referencia semantica, nao "MuJoCo upstream". O `mj_makeImpedance`
// do 3.13 tem termo implicito e `refsafe` que NAO existem aqui; usar as
// formulas de la daria uma fisica diferente da que o runtime executa.
//
// MUDANCAS em relacao ao original:
//   * a Jacobiana e DENSA (nefc x nv). O original usa o caminho esparso quando
//     `nv >= 60`, com uma cadeia de dofs por linha. Os valores sao os mesmos --
//     o que muda e o armazenamento -- e denso cabe folgado em nv=72 enquanto
//     evita montar e percorrer a cadeia dentro do kernel
//   * `mjtNum` vira `real`
//
// ## O subconjunto
//
// Contato piramidal com `condim = 3`: quatro linhas por contato, na ordem
//
//     k=1:  jN + mu1*jT1,  jN - mu1*jT1
//     k=2:  jN + mu2*jT2,  jN - mu2*jT2
//
// Sem restricao de igualdade, sem limite de junta, sem atrito de dof, sem
// tendao -- o compilador recusa o modelo se algum deles aparecer, entao
// `ne = nf = 0` e toda linha de `efc` e de contato.

#if defined(NV) && defined(NEFC_MAX) && defined(NCON_MAX)

#define CNSTR_CONTACT_PYRAMIDAL 3
#define NREF 2
#define NIMP 5

// ------------------------------------------------------------ quadro e pos

// `mju_makeFrame` com o eixo x ja escrito: completa y e z ortonormais.
// Determinista, e para o chao plano da arena da exatamente
// x=(0,0,1), y=(0,1,0), z=(-1,0,0).
static void faz_quadro(real f[9]) {
    real n = sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    real inv = REAL_ONE/n;
    f[0] *= inv; f[1] *= inv; f[2] *= inv;
    if (f[3]*f[3] + f[4]*f[4] + f[5]*f[5] < (real)0.25) {
        f[3] = REAL_ZERO; f[4] = REAL_ZERO; f[5] = REAL_ZERO;
        if (f[1] < (real)0.5 && f[1] > -(real)0.5) f[4] = REAL_ONE;
        else f[5] = REAL_ONE;
    }
    real d = f[0]*f[3] + f[1]*f[4] + f[2]*f[5];
    f[3] -= f[0]*d; f[4] -= f[1]*d; f[5] -= f[2]*d;
    real ny = sqrt(f[3]*f[3] + f[4]*f[4] + f[5]*f[5]);
    real iy = REAL_ONE/ny;
    f[3] *= iy; f[4] *= iy; f[5] *= iy;
    f[6] = f[1]*f[5] - f[2]*f[4];
    f[7] = f[2]*f[3] - f[0]*f[5];
    f[8] = f[0]*f[4] - f[1]*f[3];
}

// Uma thread por contato: quadro completo e `exclude` pela margem util.
//
// `mj_setContact` marca `exclude = (dist >= includemargin)`. Um contato
// excluido CONTA em `ncon` -- ele existe, e a interface o desenha -- mas NAO
// gera linha de restricao. Faltou isso no primeiro porte, e a trajetoria batia
// nos dois primeiros passos e divergia no terceiro, que e quando o primeiro
// contato entrou nessa faixa.
__kernel void contato_quadro(
    __global const int* ncon, __global const real* con_normal,
    __global const int* con_pair, __global const real* pair_margin,
    __global const real* pair_gap, __global const real* con_dist,
    __global real* con_frame, __global real* con_includemargin,
    __global int* con_exclude)
{
    int i = get_global_id(0);
    if (i >= ncon[0]) return;
    real f[9];
    f[0] = con_normal[3*i]; f[1] = con_normal[3*i+1]; f[2] = con_normal[3*i+2];
    for (int k = 3; k < 9; ++k) f[k] = REAL_ZERO;
    faz_quadro(f);
    for (int k = 0; k < 9; ++k) con_frame[9*i+k] = f[k];
    int p = con_pair[i];
    real im = pair_margin[p] - pair_gap[p];
    con_includemargin[i] = im;
    con_exclude[i] = (con_dist[i] >= im) ? 1 : 0;
}

// Endereco da primeira linha de cada contato incluido, e `nefc`.
// Serial de proposito: sao poucas dezenas de contatos, e a ORDEM das linhas e
// a ordem dos contatos -- um scan com atomico a tornaria nao determinista.
__kernel void contato_enderecos(
    __global const int* ncon, __global const int* con_exclude,
    __global int* con_efcadr, __global int* nefc)
{
    if (get_global_id(0) != 0) return;
    int n = 0;
    for (int i = 0; i < ncon[0]; ++i) {
        if (con_exclude[i]) { con_efcadr[i] = -1; continue; }
        con_efcadr[i] = n;
        n += 4;
    }
    nefc[0] = n;
}

// ------------------------------------------------------------- Jacobiana
//
// `mj_jac` para o corpo do geom 2 no ponto de contato, menos o do geom 1.
// O geom 1 e o plano, que pertence ao mundo e nao tem dof, entao a diferenca
// e a Jacobiana do corpo 2 -- mas o kernel nao assume isso: ele percorre as
// duas cadeias e subtrai, porque cilindro x malha tera dois corpos moveis.
//
// Uma thread por (contato, dof). Cada uma decide se o dof esta na cadeia do
// corpo e, se estiver, escreve a coluna. Sem corrida: cada thread escreve numa
// posicao propria.
__kernel void contato_jacobiana(
    __global const int* ncon,
    __global const int* con_geom, __global const real* con_pos,
    __global const real* con_frame, __global const real* con_dist,
    __global const real* con_includemargin, __global const int* con_pair,
    __global const real* pair_friction, __global const int* con_efcadr,
    __global const int* geom_bodyid, __global const int* body_rootid,
    __global const int* body_weldid, __global const int* body_dofadr,
    __global const int* body_dofnum, __global const int* dof_parentid,
    __global const real* subtree_com, __global const real* cdof,
    __global real* efc_J, __global real* efc_pos, __global real* efc_margin,
    __global int* efc_id)
{
    int gid = get_global_id(0);
    int c = gid / NV;
    int i = gid - c * NV;
    if (c >= ncon[0]) return;
    int base = con_efcadr[c];
    if (base < 0) return;                       // contato excluido pela margem

    real pnt[3] = {con_pos[3*c], con_pos[3*c+1], con_pos[3*c+2]};

    // coluna i da Jacobiana translacional, por lado; a diferenca e lado1 - lado0
    real jd[3] = {REAL_ZERO, REAL_ZERO, REAL_ZERO};
    for (int lado = 0; lado < 2; ++lado) {
        int b = geom_bodyid[con_geom[2*c + lado]];
        real off[3] = {pnt[0] - subtree_com[3*body_rootid[b]],
                       pnt[1] - subtree_com[3*body_rootid[b]+1],
                       pnt[2] - subtree_com[3*body_rootid[b]+2]};
        int w = body_weldid[b];
        if (!w) continue;
        // o dof `i` pertence a cadeia deste corpo?
        int pertence = 0;
        for (int k = body_dofadr[w] + body_dofnum[w] - 1; k >= 0;
             k = dof_parentid[k]) {
            if (k == i) { pertence = 1; break; }
        }
        if (!pertence) continue;
        __global const real* cd = cdof + 6*i;
        real cr[3] = {cd[1]*off[2] - cd[2]*off[1],
                      cd[2]*off[0] - cd[0]*off[2],
                      cd[0]*off[1] - cd[1]*off[0]};
        real s = lado ? REAL_ONE : -REAL_ONE;
        jd[0] += s*(cd[3] + cr[0]);
        jd[1] += s*(cd[4] + cr[1]);
        jd[2] += s*(cd[5] + cr[2]);
    }

    // rotaciona para o quadro do contato: tres linhas (normal, t1, t2)
    __global const real* F = con_frame + 9*c;
    real j3[3];
    for (int r = 0; r < 3; ++r) {
        j3[r] = F[3*r]*jd[0] + F[3*r+1]*jd[1] + F[3*r+2]*jd[2];
    }

    // quatro linhas piramidais
    int p = con_pair[c];
    real mu1 = pair_friction[5*p], mu2 = pair_friction[5*p+1];
    efc_J[(base+0)*NV + i] = j3[0] + mu1*j3[1];
    efc_J[(base+1)*NV + i] = j3[0] - mu1*j3[1];
    efc_J[(base+2)*NV + i] = j3[0] + mu2*j3[2];
    efc_J[(base+3)*NV + i] = j3[0] - mu2*j3[2];

    if (i == 0) {
        for (int k = 0; k < 4; ++k) {
            efc_pos[base+k] = con_dist[c];
            efc_margin[base+k] = con_includemargin[c];
            efc_id[base+k] = c;
        }
    }
}

// ------------------------------------------------------------- impedancia

// `getimpedance`: curva de impedancia de `solimp`, mais a derivada.
static void impedancia(const real solimp[NIMP], real pos, real margin,
                       real* imp, real* impP)
{
    if (solimp[0] == solimp[1] || solimp[2] <= MINVAL) {
        *imp = (real)0.5*(solimp[0] + solimp[1]);
        *impP = REAL_ZERO;
        return;
    }
    real x = (pos - margin) / solimp[2];
    real sgn = REAL_ONE;
    if (x < REAL_ZERO) { x = -x; sgn = -REAL_ONE; }
    if (x >= REAL_ONE || x <= REAL_ZERO) {
        *imp = (x >= REAL_ONE ? solimp[1] : solimp[0]);
        *impP = REAL_ZERO;
        return;
    }
    real y, yP;
    real p = solimp[4];
    if (p == REAL_ONE) {
        y = x; yP = REAL_ONE;
    } else if (x <= solimp[3]) {
        real a = REAL_ONE/pow(solimp[3], p - REAL_ONE);
        y = a*pow(x, p);
        yP = p * a*pow(x, p - REAL_ONE);
    } else {
        real b = REAL_ONE/pow(REAL_ONE - solimp[3], p - REAL_ONE);
        y = REAL_ONE - b*pow(REAL_ONE - x, p);
        yP = p * b*pow(REAL_ONE - x, p - REAL_ONE);
    }
    *imp = solimp[0] + y*(solimp[1] - solimp[0]);
    *impP = yP * sgn * (solimp[1] - solimp[0]) / solimp[2];
}

// `mj_diagApprox` + `mj_makeImpedance` para um contato inteiro (4 linhas).
// Uma thread por CONTATO, porque as quatro linhas compartilham `imp`, o
// ajuste piramidal de R e o `mu` do cone regularizado.
__kernel void restricao_impedancia(
    __global const int* ncon, const real impratio,
    __global const int* con_efcadr,
    __global const int* con_geom, __global const int* con_pair,
    __global const real* con_dist, __global const real* con_includemargin,
    __global const real* pair_friction, __global const real* pair_solref,
    __global const real* pair_solimp,
    __global const int* geom_bodyid, __global const real* body_invweight0,
    __global real* efc_diagApprox, __global real* efc_R, __global real* efc_D,
    __global real* efc_KBIP, __global real* con_mu)
{
    int c = get_global_id(0);
    if (c >= ncon[0]) return;
    int base = con_efcadr[c];
    if (base < 0) return;
    int p = con_pair[c];

    // --- diagApprox: inercia inversa aproximada, somada dos dois lados
    real tran = REAL_ZERO;
    for (int lado = 0; lado < 2; ++lado) {
        int b = geom_bodyid[con_geom[2*c + lado]];
        tran += body_invweight0[2*b];
    }
    real fri[2] = {pair_friction[5*p], pair_friction[5*p+1]};
    real dA[4];
    for (int j = 0; j < 2; ++j) {
        real f = fri[j];
        dA[2*j] = dA[2*j+1] = tran + f*f*tran;
    }

    // --- impedancia
    real solimp[NIMP];
    for (int k = 0; k < NIMP; ++k) solimp[k] = pair_solimp[NIMP*p + k];
    real solref[NREF] = {pair_solref[NREF*p], pair_solref[NREF*p+1]};
    real imp, impP;
    impedancia(solimp, con_dist[c], con_includemargin[c], &imp, &impP);

    real R[4];
    for (int j = 0; j < 4; ++j) {
        R[j] = fmax(MINVAL, (REAL_ONE - imp)*dA[j]/imp);
        real K, B;
        if (solref[0] > REAL_ZERO) {
            K = REAL_ONE / fmax(MINVAL, solimp[1]*solimp[1]
                                        * solref[0]*solref[0]
                                        * solref[1]*solref[1]);
        } else {
            K = -solref[0] / fmax(MINVAL, solimp[1]*solimp[1]);
        }
        if (solref[1] > REAL_ZERO) {
            B = (real)2 / fmax(MINVAL, solimp[1]*solref[0]);
        } else {
            B = -solref[1] / fmax(MINVAL, solimp[1]);
        }
        efc_KBIP[4*(base+j)+0] = K;
        efc_KBIP[4*(base+j)+1] = B;
        efc_KBIP[4*(base+j)+2] = imp;
        efc_KBIP[4*(base+j)+3] = impP;
    }

    // --- ajuste piramidal: R comum, casando a impedancia de atrito do modelo
    // eliptico. `R[1] = R[0]/impratio` e so um passo intermediario para `mu`.
    real R1 = R[0]/fmax(MINVAL, impratio);
    real mu = fri[0] * sqrt(R1/R[0]);
    real Rpy = (real)2*mu*mu*R[0];
    con_mu[c] = mu;
    for (int j = 0; j < 4; ++j) {
        efc_R[base+j] = Rpy;
        efc_D[base+j] = REAL_ONE/Rpy;
        efc_diagApprox[base+j] = Rpy * imp / (REAL_ONE - imp);
    }
}

// `mj_referenceConstraint`: aref = -B*vel - K*I*(pos - margin).
// `vel = J*qvel` sai antes, no kernel de produto.
__kernel void restricao_aref(
    __global const int* nefc, __global const real* efc_J,
    __global const real* qvel, __global const real* efc_KBIP,
    __global const real* efc_pos, __global const real* efc_margin,
    __global real* efc_vel, __global real* efc_aref)
{
    int i = get_global_id(0);
    if (i >= nefc[0]) return;
    real v = REAL_ZERO;
    __global const real* row = efc_J + (size_t)i*NV;
    for (int k = 0; k < NV; ++k) v += row[k]*qvel[k];
    efc_vel[i] = v;
    efc_aref[i] = -efc_KBIP[4*i+1]*v
                  - efc_KBIP[4*i]*efc_KBIP[4*i+2]*(efc_pos[i] - efc_margin[i]);
}

#endif  // NV && NEFC_MAX && NCON_MAX
