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
static void contato_quadro_um(int i,
    __global const int* ncon,
    __global const real* con_normal,
    __global const int* con_pair,
    __global const real* pair_margin,
    __global const real* pair_gap,
    __global const real* con_dist,
    __global real* con_frame,
    __global real* con_includemargin,
    __global int* con_exclude)
{
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

__kernel void contato_quadro(
    __global const int* ncon,
    __global const real* con_normal,
    __global const int* con_pair,
    __global const real* pair_margin,
    __global const real* pair_gap,
    __global const real* con_dist,
    __global real* con_frame,
    __global real* con_includemargin,
    __global int* con_exclude)
{
    contato_quadro_um(get_global_id(0), ncon, con_normal, con_pair, pair_margin, pair_gap, con_dist, con_frame, con_includemargin, con_exclude);
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
static void contato_jacobiana_um(int gid,
    __global const int* ncon,
    __global const int* con_geom,
    __global const real* con_pos,
    __global const real* con_frame,
    __global const real* con_dist,
    __global const real* con_includemargin,
    __global const int* con_pair,
    __global const real* pair_friction,
    __global const int* con_efcadr,
    __global const int* geom_bodyid,
    __global const int* body_rootid,
    __global const int* body_weldid,
    __global const int* body_dofadr,
    __global const int* body_dofnum,
    __global const int* dof_parentid,
    __global const real* subtree_com,
    __global const real* cdof,
    __global real* efc_J,
    __global real* efc_pos,
    __global real* efc_margin,
    __global int* efc_id)
{
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

__kernel void contato_jacobiana(
    __global const int* ncon,
    __global const int* con_geom,
    __global const real* con_pos,
    __global const real* con_frame,
    __global const real* con_dist,
    __global const real* con_includemargin,
    __global const int* con_pair,
    __global const real* pair_friction,
    __global const int* con_efcadr,
    __global const int* geom_bodyid,
    __global const int* body_rootid,
    __global const int* body_weldid,
    __global const int* body_dofadr,
    __global const int* body_dofnum,
    __global const int* dof_parentid,
    __global const real* subtree_com,
    __global const real* cdof,
    __global real* efc_J,
    __global real* efc_pos,
    __global real* efc_margin,
    __global int* efc_id)
{
    contato_jacobiana_um(get_global_id(0), ncon, con_geom, con_pos, con_frame, con_dist, con_includemargin, con_pair, pair_friction, con_efcadr, geom_bodyid, body_rootid, body_weldid, body_dofadr, body_dofnum, dof_parentid, subtree_com, cdof, efc_J, efc_pos, efc_margin, efc_id);
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
static void restricao_impedancia_um(int c,
    __global const int* ncon,
    const real impratio,
    __global const int* con_efcadr,
    __global const int* con_geom,
    __global const int* con_pair,
    __global const real* con_dist,
    __global const real* con_includemargin,
    __global const real* pair_friction,
    __global const real* pair_solref,
    __global const real* pair_solimp,
    __global const int* geom_bodyid,
    __global const real* body_invweight0,
    __global real* efc_diagApprox,
    __global real* efc_R,
    __global real* efc_D,
    __global real* efc_KBIP,
    __global real* con_mu)
{
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

__kernel void restricao_impedancia(
    __global const int* ncon,
    const real impratio,
    __global const int* con_efcadr,
    __global const int* con_geom,
    __global const int* con_pair,
    __global const real* con_dist,
    __global const real* con_includemargin,
    __global const real* pair_friction,
    __global const real* pair_solref,
    __global const real* pair_solimp,
    __global const int* geom_bodyid,
    __global const real* body_invweight0,
    __global real* efc_diagApprox,
    __global real* efc_R,
    __global real* efc_D,
    __global real* efc_KBIP,
    __global real* con_mu)
{
    restricao_impedancia_um(get_global_id(0), ncon, impratio, con_efcadr, con_geom, con_pair, con_dist, con_includemargin, pair_friction, pair_solref, pair_solimp, geom_bodyid, body_invweight0, efc_diagApprox, efc_R, efc_D, efc_KBIP, con_mu);
}


// `mj_referenceConstraint`: aref = -B*vel - K*I*(pos - margin).
// `vel = J*qvel` sai antes, no kernel de produto.
static void restricao_aref_um(int i,
    __global const int* nefc,
    __global const real* efc_J,
    __global const real* qvel,
    __global const real* efc_KBIP,
    __global const real* efc_pos,
    __global const real* efc_margin,
    __global real* efc_vel,
    __global real* efc_aref)
{
    if (i >= nefc[0]) return;
    real v = REAL_ZERO;
    __global const real* row = efc_J + (size_t)i*NV;
    for (int k = 0; k < NV; ++k) v += row[k]*qvel[k];
    efc_vel[i] = v;
    efc_aref[i] = -efc_KBIP[4*i+1]*v
                  - efc_KBIP[4*i]*efc_KBIP[4*i+2]*(efc_pos[i] - efc_margin[i]);
}

__kernel void restricao_aref(
    __global const int* nefc,
    __global const real* efc_J,
    __global const real* qvel,
    __global const real* efc_KBIP,
    __global const real* efc_pos,
    __global const real* efc_margin,
    __global real* efc_vel,
    __global real* efc_aref)
{
    restricao_aref_um(get_global_id(0), nefc, efc_J, qvel, efc_KBIP, efc_pos, efc_margin, efc_vel, efc_aref);
}


// ---------------------------------------------------------------- adesao
//
// `mj_transmission`, ramo `mjTRN_BODY`. O atuador de adesao nao tem
// comprimento: o "momento" dele e a MEDIA das Jacobianas normais dos contatos
// que tocam o corpo, com o sinal trocado -- puxar, nao empurrar.
//
// Para cone piramidal a normal nao e uma linha: ela e a media das `2*(dim-1)`
// direcoes da piramide, cada uma com peso `0.5/(dim-1)`. Com `condim = 3` sao
// quatro linhas com peso 0,25.
//
// Contatos na faixa de `gap` (`exclude == 1`) entram no original por um caminho
// proprio, com a Jacobiana montada na hora. O compilador recusa o modelo se
// algum `pair_gap` for nao nulo, entao esse caminho nao pode ocorrer aqui e
// nao e implementado -- em vez de ser implementado errado e nunca exercitado.
static void adesao_momento_um(int gid,
    const int nu,
    __global const int* ncon,
    __global const int* actuator_trntype,
    __global const int* actuator_trnid,
    __global const int* con_geom,
    __global const int* con_efcadr,
    __global const int* geom_bodyid,
    __global const real* efc_J,
    __global real* ades_momento,
    __global int* ades_conta)
{
    int a = gid / NV;
    int i = gid - a * NV;
    if (a >= nu) return;
    if (actuator_trntype[a] != TRN_BODY) {
        ades_momento[(size_t)a*NV + i] = REAL_ZERO;
        if (i == 0) ades_conta[a] = 0;
        return;
    }
    int corpo = actuator_trnid[2*a];
    int conta = 0;
    real mom = REAL_ZERO;
    for (int c = 0; c < ncon[0]; ++c) {
        int b1 = geom_bodyid[con_geom[2*c]], b2 = geom_bodyid[con_geom[2*c+1]];
        if (b1 != corpo && b2 != corpo) continue;
        int base = con_efcadr[c];
        if (base < 0) continue;              // contato excluido: fora do subset
        conta++;
        // condim = 3 -> quatro linhas, peso 0,5/(dim-1) = 0,25
        for (int k = 0; k < 4; ++k) {
            mom += (real)0.25 * efc_J[(size_t)(base+k)*NV + i];
        }
    }
    ades_momento[(size_t)a*NV + i] = conta ? (-mom/(real)conta) : REAL_ZERO;
    if (i == 0) ades_conta[a] = conta;
}

__kernel void adesao_momento(
    const int nu,
    __global const int* ncon,
    __global const int* actuator_trntype,
    __global const int* actuator_trnid,
    __global const int* con_geom,
    __global const int* con_efcadr,
    __global const int* geom_bodyid,
    __global const real* efc_J,
    __global real* ades_momento,
    __global int* ades_conta)
{
    adesao_momento_um(get_global_id(0), nu, ncon, actuator_trntype, actuator_trnid, con_geom, con_efcadr, geom_bodyid, efc_J, ades_momento, ades_conta);
}


// Soma a contribuicao da adesao em `qfrc_actuator`. Uma thread por DOF varrendo
// os atuadores: sem corrida e sem atomico.
static void adesao_projeta_um(int i,
    const int nu,
    __global const int* actuator_trntype,
    __global const real* ades_momento,
    __global const real* actuator_force,
    __global real* qfrc_actuator)
{
    if (i >= NV) return;
    real s = REAL_ZERO;
    for (int a = 0; a < nu; ++a) {
        if (actuator_trntype[a] != TRN_BODY) continue;
        s += ades_momento[(size_t)a*NV + i] * actuator_force[a];
    }
    qfrc_actuator[i] += s;
}

__kernel void adesao_projeta(
    const int nu,
    __global const int* actuator_trntype,
    __global const real* ades_momento,
    __global const real* actuator_force,
    __global real* qfrc_actuator)
{
    adesao_projeta_um(get_global_id(0), nu, actuator_trntype, ades_momento, actuator_force, qfrc_actuator);
}


// ------------------------------------------------- forca por segmento
//
// `mj_contactForce` + `mju_decodePyramid` + a reducao que o controlador de
// marcha faz em `get_bodysegment_contact_forces`.
//
// Existe na GPU para que o controlador NAO precise ler de volta o estado de
// restricao inteiro por passo. Ele quer 30 segmentos x 3 componentes: 90
// numeros, contra `efc_force` + `efc_J` + a tabela de contatos.
//
// Uma thread por SEGMENTO, varrendo os contatos. Ao contrario de uma thread
// por contato, nao ha corrida: cada segmento acumula so no proprio lugar. Sao
// poucas dezenas de contatos, entao a varredura e barata.
__kernel void forcas_segmentos(
    const int nseg, __global const int* ncon,
    __global const int* con_geom, __global const int* con_efcadr,
    __global const int* con_pair, __global const real* pair_friction,
    __global const real* con_frame, __global const real* efc_force,
    __global const int* geom_saida, __global const int* geom_chao,
    const int so_chao, __global real* forcas)
{
    int s = get_global_id(0);
    if (s >= nseg) return;
    real acc[3] = {REAL_ZERO, REAL_ZERO, REAL_ZERO};
    for (int c = 0; c < ncon[0]; ++c) {
        int base = con_efcadr[c];
        if (base < 0) continue;                  // contato excluido
        int g1 = con_geom[2*c], g2 = con_geom[2*c+1];
        int s1 = geom_saida[g1], s2 = geom_saida[g2];
        if (s1 != s && s2 != s) continue;
        if (so_chao) {
            int ok = (s1 >= 0 && geom_chao[g2]) || (s2 >= 0 && geom_chao[g1]);
            if (!ok) continue;
        }
        // `mju_decodePyramid` com dim = 3: normal e a soma das quatro, e cada
        // tangente e a diferenca do par vezes o atrito daquela direcao
        int p = con_pair[c];
        real f0 = efc_force[base], f1 = efc_force[base+1];
        real f2 = efc_force[base+2], f3 = efc_force[base+3];
        real w[3];
        w[0] = f0 + f1 + f2 + f3;
        w[1] = (f0 - f1) * pair_friction[5*p];
        w[2] = (f2 - f3) * pair_friction[5*p+1];
        // quadro^T * w: do quadro do contato para o mundo
        __global const real* F = con_frame + 9*c;
        real fw[3];
        for (int k = 0; k < 3; ++k)
            fw[k] = F[k]*w[0] + F[3+k]*w[1] + F[6+k]*w[2];
        real sinal = (s2 == s) ? REAL_ONE : -REAL_ONE;
        for (int k = 0; k < 3; ++k) acc[k] += sinal * fw[k];
    }
    for (int k = 0; k < 3; ++k) forcas[3*s+k] = acc[k];
}

// ------------------------------------------ pacote de observacao do controlador
//
// Tudo que o controlador de marcha le por passo, num buffer contiguo:
//
//     [0            .. 3*nfly)          xpos dos corpos da mosca
//     [3*nfly       .. 3*nfly+9)        xmat do torax
//     [3*nfly+9     .. 3*nfly+9+3*nseg) forca de contato por segmento
//
// Eram tres leituras de volta por passo, e cada `enqueue_copy` do pyopencl e
// BLOQUEANTE: tres esperas pela placa, ~80 us de ida e volta cada. Uma leitura
// so paga uma.
//
// Isto nao reduz o acoplamento com o Python -- o controlador continua na CPU.
// Reduz o numero de vezes que a CPU para para esperar.
__kernel void empacota_observacao(
    const int nfly, const int nseg, const int id_torax,
    __global const int* bodyids_fly,
    __global const real* xpos, __global const real* xmat,
    __global const real* forcas_seg, __global real* pacote)
{
    int i = get_global_id(0);
    if (i < nfly) {
        int b = bodyids_fly[i];
        pacote[3*i+0] = xpos[3*b+0];
        pacote[3*i+1] = xpos[3*b+1];
        pacote[3*i+2] = xpos[3*b+2];
    }
    if (i < 9) pacote[3*nfly + i] = xmat[9*id_torax + i];
    if (i < 3*nseg) pacote[3*nfly + 9 + i] = forcas_seg[i];
}

#endif  // NV && NEFC_MAX && NCON_MAX
