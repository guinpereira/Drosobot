// O passo em poucos despachos: o laco por nivel migra para dentro do kernel.
//
// Nada de fisica novo mora aqui. Cada kernel deste arquivo chama as MESMAS
// funcoes `..._um` que os kernels por estagio chamam -- a conta existe uma vez
// so, e o caminho de depuracao nao pode divergir do caminho rapido. O que muda
// e quem percorre o indice: la o NDRange, aqui um laco com `barrier()`.
//
// ## Por que isto existe
//
// Medido no baseline: 80 despachos por passo, 6174 us no total, dos quais
// 3960 us (64%) sao o HOST enfileirando. O custo por despacho e 53 us, e 21 us
// disso e `set_args` -- religar argumentos que nunca mudam entre passos.
//
// Quarenta e seis dos 80 despachos eram um laco `for nivel: dispatch(...)`.
// Eles existiam para depurar: com um despacho por nivel da para parar em
// qualquer um e comparar com o `mjData`. Os campos ja estao validados, entao o
// laco pode entrar no kernel.
//
// ## Um work-group, e por que isso basta
//
// Todos os kernels daqui rodam num work-group so, porque `barrier()` nao
// sincroniza entre work-groups. O modelo cabe: `nbody = 72`, `njnt = 67`,
// `nv = 72`, `nC = 813` -- um grupo de 256 threads percorre qualquer um deles
// com passo `W`. O que nao cabe num grupo (a varredura de vertices da colisao,
// 55 pares x ~1000 vertices) continua com um work-group por par, fora daqui.

#if defined(NBODY) && defined(NV) && defined(NQ) && defined(NEFC_MAX)

// ------------------------------------------------------------------ comPos
//
// `com_momento` -> niveis para tras -> `com_normaliza` -> `com_inercia` e
// `com_cdof`. A barreira entre niveis e o que o despacho fazia antes.
__kernel void com_pos_fundido(
    const int profundidade, const int njnt,
    __global const int* pais, __global const int* pais_adr,
    __global const int* pais_num, __global const int* filhos_adr,
    __global const int* filhos_num, __global const int* filhos,
    __global const int* nivel_de,
    __global const real* xipos, __global const real* body_mass,
    __global const real* body_subtreemass, __global const int* body_rootid,
    __global const real* body_inertia, __global const real* ximat,
    __global const int* jnt_type, __global const int* jnt_dofadr,
    __global const int* jnt_bodyid, __global const real* xmat,
    __global const real* xanchor, __global const real* xaxis,
    __global real* subtree_com, __global real* cinert, __global real* cdof)
{
    int t = get_local_id(0), W = get_local_size(0);

    for (int i = t; i < NBODY; i += W)
        com_momento_um(i, xipos, body_mass, subtree_com);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int d = profundidade - 1; d >= 1; --d) {
        for (int k = t; k < pais_num[d]; k += W)
            com_acumula_nivel_um(k, pais, pais_adr, pais_num, filhos_adr,
                                 filhos_num, filhos, nivel_de, d, subtree_com);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    for (int i = t; i < NBODY; i += W)
        com_normaliza_um(i, body_subtreemass, xipos, subtree_com);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < NBODY; i += W)
        com_inercia_um(i, body_rootid, body_inertia, body_mass, ximat, xipos,
                       subtree_com, cinert);
    for (int j = t; j < njnt; j += W)
        com_cdof_um(j, njnt, jnt_type, jnt_dofadr, jnt_bodyid, body_rootid,
                    xmat, xanchor, xaxis, subtree_com, cdof);
}

// -------------------------------------------------------------------- massa
//
// `crb_inicia` -> niveis para tras -> `crb_monta_M` -> `factor_M`.
// A fatoracao ja era cooperativa de work-group; aqui ela so deixa de ser um
// despacho proprio.
__kernel void massa_fundida(
    const int profundidade,
    __global const int* pais, __global const int* pais_adr,
    __global const int* pais_num, __global const int* filhos_adr,
    __global const int* filhos_num, __global const int* filhos,
    __global const int* nivel_de,
    __global const int* M_rownnz, __global const int* M_rowadr,
    __global const int* M_colind, __global const int* dof_parentid,
    __global const int* dof_bodyid, __global const real* dof_armature,
    __global const real* cinert, __global const real* cdof,
    __global real* crb, __global real* M, __global real* qLD,
    __global real* qLDiagInv)
{
    __local real l[NC];
    __local int lcol[NC], lnnz[NV], ladr[NV];
    int t = get_local_id(0), W = get_local_size(0);

    for (int i = t; i < NBODY; i += W) crb_inicia_um(i, cinert, crb);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int d = profundidade - 1; d >= 1; --d) {
        for (int k = t; k < pais_num[d]; k += W)
            crb_acumula_nivel_um(k, pais, pais_adr, pais_num, filhos_adr,
                                 filhos_num, filhos, nivel_de, d, crb);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    for (int i = t; i < NV; i += W)
        crb_monta_M_um(i, M_rownnz, M_rowadr, dof_parentid, dof_bodyid,
                       dof_armature, crb, cdof, M);
    barrier(CLK_GLOBAL_MEM_FENCE);

    factor_M_lds_um(t, W, M_rownnz, M_rowadr, M_colind, M, qLD, qLDiagInv,
                    l, lcol, lnnz, ladr);
}

// ------------------------------------------------------------------ comVel
__kernel void com_vel_fundido(
    const int profundidade,
    __global const int* nivel_corpos, __global const int* nivel_adr,
    __global const int* nivel_num, __global const int* body_parentid,
    __global const int* body_dofadr, __global const int* body_dofnum,
    __global const int* body_jntadr, __global const int* jnt_type,
    __global const int* dof_jntid, __global const real* cdof,
    __global const real* qvel, __global real* cvel, __global real* cdof_dot)
{
    int t = get_local_id(0), W = get_local_size(0);
    for (int d = 1; d < profundidade; ++d) {
        for (int k = t; k < nivel_num[d]; k += W)
            com_vel_nivel_um(k, nivel_corpos, nivel_adr, nivel_num, d,
                             body_parentid, body_dofadr, body_dofnum,
                             body_jntadr, jnt_type, dof_jntid, cdof, qvel,
                             cvel, cdof_dot);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

// --------------------------------------------------------------------- RNE
//
// Para frente sobre os niveis, para tras sobre os pais, e a projecao nos dofs.
// Dezenove despachos viram um.
__kernel void bias_fundido(
    const int profundidade,
    __global const int* nivel_corpos, __global const int* nivel_adr,
    __global const int* nivel_num,
    __global const int* pais, __global const int* pais_adr,
    __global const int* pais_num, __global const int* filhos_adr,
    __global const int* filhos_num, __global const int* filhos,
    __global const int* nivel_de,
    __global const int* body_parentid, __global const int* body_dofadr,
    __global const int* body_dofnum, __global const int* dof_bodyid,
    __global const real* cdof_dot, __global const real* qvel,
    __global const real* cinert, __global const real* cvel,
    __global const real* cdof,
    __global real* cacc, __global real* cfrc_body, __global real* qfrc_bias)
{
    int t = get_local_id(0), W = get_local_size(0);
    for (int d = 1; d < profundidade; ++d) {
        for (int k = t; k < nivel_num[d]; k += W)
            rne_frente_nivel_um(k, nivel_corpos, nivel_adr, nivel_num, d,
                                body_parentid, body_dofadr, body_dofnum,
                                cdof_dot, qvel, cinert, cvel, cacc, cfrc_body);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    for (int d = profundidade - 1; d >= 1; --d) {
        for (int k = t; k < pais_num[d]; k += W)
            rne_tras_nivel_um(k, pais, pais_adr, pais_num, filhos_adr,
                              filhos_num, filhos, nivel_de, d, cfrc_body);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    for (int i = t; i < NV; i += W)
        rne_projeta_um(i, dof_bodyid, cdof, cfrc_body, qfrc_bias);
}

// ------------------------------------------------------- passivo + atuacao
//
// Cinco despachos pequenos viram um. A adesao fica no fim porque o momento
// dela sai de `efc_J`, que ja existe quando este kernel roda.
__kernel void forcas_fundidas(
    const int njnt, const int nu,
    __global const real* qvel, __global const real* dof_damping,
    __global const int* jnt_type, __global const int* jnt_qposadr,
    __global const int* jnt_dofadr, __global const real* jnt_stiffness,
    __global const real* qpos, __global const real* qpos_spring,
    __global const int* actuator_trntype, __global const int* actuator_trnid,
    __global const int* actuator_biastype,
    __global const real* actuator_gainprm, __global const real* actuator_biasprm,
    __global const real* actuator_gear, __global const real* actuator_ctrlrange,
    __global const int* actuator_ctrllimited,
    __global const real* actuator_forcerange,
    __global const int* actuator_forcelimited, __global const real* ctrl,
    __global const int* ncon, __global const int* con_geom,
    __global const int* con_efcadr, __global const int* geom_bodyid,
    __global const real* efc_J,
    __global const real* qfrc_applied, __global const real* qfrc_bias,
    __global real* qfrc_passive, __global real* actuator_length,
    __global real* actuator_velocity, __global real* actuator_force,
    __global real* qfrc_actuator, __global real* ades_momento,
    __global int* ades_conta, __global real* qfrc_smooth)
{
    int t = get_local_id(0), W = get_local_size(0);

    for (int i = t; i < NV; i += W)
        passivo_amortecedor_um(i, qvel, dof_damping, qfrc_passive);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int j = t; j < njnt; j += W)
        passivo_mola_um(j, njnt, jnt_type, jnt_qposadr, jnt_dofadr,
                        jnt_stiffness, qpos, qpos_spring, qfrc_passive);

    for (int a = t; a < nu; a += W)
        atuacao_um(a, nu, actuator_trntype, actuator_trnid, actuator_biastype,
                   actuator_gainprm, actuator_biasprm, actuator_gear,
                   actuator_ctrlrange, actuator_ctrllimited,
                   actuator_forcerange, actuator_forcelimited, jnt_qposadr,
                   jnt_dofadr, qpos, qvel, ctrl, actuator_length,
                   actuator_velocity, actuator_force, qfrc_actuator);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < NV; i += W)
        atuacao_projeta_um(i, nu, actuator_trntype, actuator_trnid,
                           actuator_gear, jnt_dofadr, actuator_force,
                           qfrc_actuator);
    for (int g = t; g < nu*NV; g += W)
        adesao_momento_um(g, nu, ncon, actuator_trntype, actuator_trnid,
                          con_geom, con_efcadr, geom_bodyid, efc_J,
                          ades_momento, ades_conta);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < NV; i += W)
        adesao_projeta_um(i, nu, actuator_trntype, ades_momento,
                          actuator_force, qfrc_actuator);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < NV; i += W)
        soma_smooth_um(i, qfrc_passive, qfrc_bias, qfrc_actuator,
                       qfrc_applied, qfrc_smooth);
}

// ------------------------------------------------------------- restricoes
//
// Quadro do contato, enderecos das linhas, Jacobiana, impedancia e referencia.
// `contato_enderecos` e serial (a ordem das linhas e a ordem dos contatos),
// entao ele fica na thread 0 com barreira depois.
__kernel void restricoes_fundidas(
    const int npair_max, const real impratio,
    __global const int* ncon, __global const real* con_normal,
    __global const int* con_pair, __global const real* pair_margin,
    __global const real* pair_gap, __global const real* con_dist,
    __global const int* con_geom, __global const real* con_pos,
    __global const real* pair_friction, __global const real* pair_solref,
    __global const real* pair_solimp,
    __global const int* geom_bodyid, __global const int* body_rootid,
    __global const int* body_weldid, __global const int* body_dofadr,
    __global const int* body_dofnum, __global const int* dof_parentid,
    __global const real* body_invweight0,
    __global const real* subtree_com, __global const real* cdof,
    __global const real* qvel,
    __global real* con_frame, __global real* con_includemargin,
    __global int* con_exclude, __global int* con_efcadr, __global int* nefc,
    __global real* efc_J, __global real* efc_pos, __global real* efc_margin,
    __global int* efc_id, __global real* efc_diagApprox, __global real* efc_R,
    __global real* efc_D, __global real* efc_KBIP, __global real* con_mu,
    __global real* efc_vel, __global real* efc_aref)
{
    int t = get_local_id(0), W = get_local_size(0);

    for (int i = t; i < npair_max; i += W)
        contato_quadro_um(i, ncon, con_normal, con_pair, pair_margin, pair_gap,
                          con_dist, con_frame, con_includemargin, con_exclude);
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (t == 0) {
        int n = 0;
        for (int i = 0; i < ncon[0]; ++i) {
            if (con_exclude[i]) { con_efcadr[i] = -1; continue; }
            con_efcadr[i] = n;
            n += 4;
        }
        nefc[0] = n;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int g = t; g < NCON_MAX*NV; g += W)
        contato_jacobiana_um(g, ncon, con_geom, con_pos, con_frame, con_dist,
                             con_includemargin, con_pair, pair_friction,
                             con_efcadr, geom_bodyid, body_rootid, body_weldid,
                             body_dofadr, body_dofnum, dof_parentid,
                             subtree_com, cdof, efc_J, efc_pos, efc_margin,
                             efc_id);
    for (int c = t; c < NCON_MAX; c += W)
        restricao_impedancia_um(c, ncon, impratio, con_efcadr, con_geom,
                                con_pair, con_dist, con_includemargin,
                                pair_friction, pair_solref, pair_solimp,
                                geom_bodyid, body_invweight0, efc_diagApprox,
                                efc_R, efc_D, efc_KBIP, con_mu);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < NEFC_MAX; i += W)
        restricao_aref_um(i, nefc, efc_J, qvel, efc_KBIP, efc_pos, efc_margin,
                          efc_vel, efc_aref);
}

// ------------------------------------------------------- aceleracao suave
__kernel void smooth_fundido(
    __global const int* M_rownnz, __global const int* M_rowadr,
    __global const int* M_colind, __global const real* qLD,
    __global const real* qLDiagInv, __global const real* qfrc_smooth,
    __global real* qacc_smooth)
{
    __local real lqld[NC], lx[NV], linv[NV];
    __local int lnnz[NV], ladr[NV], lcol[NC];
    solve_M_lds_um(get_local_id(0), get_local_size(0), M_rownnz, M_rowadr,
                   M_colind, qLD, qLDiagInv, qfrc_smooth, qacc_smooth,
                   lqld, lx, linv, lnnz, ladr, lcol);
}

// `M` esparsa -> densa SIMETRICA, num kernel so.
//
// Eram dois: um enchia o triangulo inferior e outro espelhava. O espelho
// custava 159 us -- 2600 copias com 72 threads e leitura em passo de NV, que
// e o pior padrao possivel. Aqui cada thread e dona da propria linha do CSR e
// escreve os DOIS triangulos: a entrada (i,j) so pertence a linha i, entao nao
// ha corrida, e a zeragem separada por barreira resolve o resto.
__kernel void M_densa_simetrica(
    __global const int* M_rownnz, __global const int* M_rowadr,
    __global const int* M_colind, __global const real* M, __global real* Md)
{
    int t = get_local_id(0), W = get_local_size(0);
    for (int k = t; k < NV*NV; k += W) Md[k] = REAL_ZERO;
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NV; i += W) {
        int adr = M_rowadr[i], n = M_rownnz[i];
        for (int k = 0; k < n; ++k) {
            int j = M_colind[adr + k];
            real v = M[adr + k];
            Md[i*NV + j] = v;
            Md[j*NV + i] = v;
        }
    }
}

// ------------------------------------------------------------- integracao
//
// Copia de M, diagonal com `h*damping`, fatoracao, lado direito, solve e as
// duas integracoes. Sete despachos viram um.
__kernel void euler_fundido(
    const real dt, const int njnt,
    __global const int* M_rownnz, __global const int* M_rowadr,
    __global const int* M_colind, __global const real* M,
    __global const real* dof_damping, __global const real* qfrc_smooth,
    __global const real* qfrc_constraint,
    __global const int* jnt_type, __global const int* jnt_qposadr,
    __global const int* jnt_dofadr,
    __global real* qH_in, __global real* qH, __global real* qHDiagInv,
    __global real* rhs, __global real* qacc_euler, __global real* qvel,
    __global real* qpos)
{
    __local real lqld[NC], lx[NV], linv[NV];
    __local int lnnz[NV], ladr[NV], lcol[NC];
    int t = get_local_id(0), W = get_local_size(0);

    for (int i = t; i < NC; i += W) euler_copia_M_um(i, M, qH_in);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NV; i += W)
        euler_diag_MhD_um(i, dt, M_rownnz, M_rowadr, dof_damping, qH_in);
    barrier(CLK_GLOBAL_MEM_FENCE);

    factor_M_lds_um(t, W, M_rownnz, M_rowadr, M_colind, qH_in, qH, qHDiagInv,
                    lqld, lcol, lnnz, ladr);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < NV; i += W)
        euler_rhs_um(i, qfrc_smooth, qfrc_constraint, rhs);
    barrier(CLK_GLOBAL_MEM_FENCE);

    solve_M_lds_um(t, W, M_rownnz, M_rowadr, M_colind, qH, qHDiagInv, rhs,
                   qacc_euler, lqld, lx, linv, lnnz, ladr, lcol);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < NV; i += W) euler_qvel_um(i, dt, qacc_euler, qvel);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < njnt; i += W)
        euler_qpos_um(i, dt, njnt, jnt_type, jnt_qposadr, jnt_dofadr, qvel,
                      qpos);
}


// ------------------------------------------------------ grupos de etapas
//
// Segunda rodada de fusao, e desta vez o alvo nao e o host.
//
// Medido depois da primeira fusao: 15 despachos, 1450 us por passo com a fila
// pipelinada, dos quais os eventos de GPU somam ~690 us. Os outros ~760 us sao
// LACUNA entre kernels -- ~51 us por despacho. Um kernel trivial dependente
// custa 4 us de lacuna; um kernel de um work-group com secoes seriais longas
// custa dez vezes mais, porque a placa drena o pipeline inteiro entre dois
// despachos que nao podem se sobrepor.
//
// Entao: menos despachos, de novo. 15 -> 6.
//
// O agrupamento respeita duas fronteiras que NAO dao para cruzar:
//
//   * `suporte_plano_malha` quer um work-group POR PAR (55 grupos varrendo
//     ~1000 vertices cada). Trazer isso para um grupo so trocaria 51 us de
//     lacuna por uma varredura serializada;
//   * o solver e o Euler alocam LDS grande (`H` de 41 KiB, `qLD` de 6,5 KiB).
//     Junta-los a cinematica, que tambem quer 15 KiB, passaria de 64 KiB.
//
// Como sempre neste arquivo: nada de conta nova. Os grupos chamam os mesmos
// kernels fundidos, que chamam as mesmas funcoes `..._um`.

// Grupo B: quadros inerciais -> matriz de massa -> velocidades de corpo.
// Sao tres kernels que so passam buffers um para o outro; nenhum deles usa
// LDS, entao juntar os tres nao custa ocupacao.
__kernel void grupo_arvore(
    const int profundidade, const int njnt,
    __global const int* pais, __global const int* pais_adr,
    __global const int* pais_num, __global const int* filhos_adr,
    __global const int* filhos_num, __global const int* filhos,
    __global const int* nivel_de, __global const int* nivel_corpos,
    __global const int* nivel_adr, __global const int* nivel_num,
    __global const real* xipos, __global const real* body_mass,
    __global const real* body_subtreemass, __global const int* body_rootid,
    __global const real* body_inertia, __global const real* ximat,
    __global const int* jnt_type, __global const int* jnt_dofadr,
    __global const int* jnt_bodyid, __global const real* xmat,
    __global const real* xanchor, __global const real* xaxis,
    __global const int* M_rownnz, __global const int* M_rowadr,
    __global const int* M_colind, __global const int* dof_parentid,
    __global const int* dof_bodyid, __global const real* dof_armature,
    __global const int* body_parentid, __global const int* body_dofadr,
    __global const int* body_dofnum, __global const int* body_jntadr,
    __global const int* dof_jntid, __global const real* qvel,
    __global real* subtree_com, __global real* cinert, __global real* cdof,
    __global real* crb, __global real* M, __global real* qLD,
    __global real* qLDiagInv, __global real* cvel, __global real* cdof_dot)
{
    int t = get_local_id(0), W = get_local_size(0);

    // --- comPos
    for (int i = t; i < NBODY; i += W)
        com_momento_um(i, xipos, body_mass, subtree_com);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int d = profundidade - 1; d >= 1; --d) {
        for (int k = t; k < pais_num[d]; k += W)
            com_acumula_nivel_um(k, pais, pais_adr, pais_num, filhos_adr,
                                 filhos_num, filhos, nivel_de, d, subtree_com);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    for (int i = t; i < NBODY; i += W)
        com_normaliza_um(i, body_subtreemass, xipos, subtree_com);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NBODY; i += W)
        com_inercia_um(i, body_rootid, body_inertia, body_mass, ximat, xipos,
                       subtree_com, cinert);
    for (int j = t; j < njnt; j += W)
        com_cdof_um(j, njnt, jnt_type, jnt_dofadr, jnt_bodyid, body_rootid,
                    xmat, xanchor, xaxis, subtree_com, cdof);
    barrier(CLK_GLOBAL_MEM_FENCE);

    // --- massa
    for (int i = t; i < NBODY; i += W) crb_inicia_um(i, cinert, crb);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int d = profundidade - 1; d >= 1; --d) {
        for (int k = t; k < pais_num[d]; k += W)
            crb_acumula_nivel_um(k, pais, pais_adr, pais_num, filhos_adr,
                                 filhos_num, filhos, nivel_de, d, crb);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    for (int i = t; i < NV; i += W)
        crb_monta_M_um(i, M_rownnz, M_rowadr, dof_parentid, dof_bodyid,
                       dof_armature, crb, cdof, M);
    barrier(CLK_GLOBAL_MEM_FENCE);
    factor_M_um(t, W, M_rownnz, M_rowadr, M_colind, M, qLD, qLDiagInv);
    barrier(CLK_GLOBAL_MEM_FENCE);

    // --- comVel
    for (int d = 1; d < profundidade; ++d) {
        for (int k = t; k < nivel_num[d]; k += W)
            com_vel_nivel_um(k, nivel_corpos, nivel_adr, nivel_num, d,
                             body_parentid, body_dofadr, body_dofnum,
                             body_jntadr, jnt_type, dof_jntid, cdof, qvel,
                             cvel, cdof_dot);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

// Grupo D: contatos -> restricoes -> RNE -> forcas -> aceleracao suave.
// Seis kernels que formam uma cadeia estrita de dependencia.
__kernel void grupo_restricoes(
    const int npair, const int njnt, const int nu, const real impratio,
    const int profundidade,
    __global const int* pair_geom1, __global const int* pair_geom2,
    __global const int* geom_dataid, __global const real* geom_rbound,
    __global const real* geom_xpos, __global const real* geom_xmat,
    __global const real* pair_margin, __global const real* pair_gap,
    __global const int* mesh_vertadr, __global const float* mesh_vert,
    __global const int* mesh_graphadr, __global const int* mesh_graph,
    __global const int* vert_local,
    __global const int* sup_vert, __global const real* sup_dist,
    __global const real* pair_friction, __global const real* pair_solref,
    __global const real* pair_solimp,
    __global const int* geom_bodyid, __global const int* body_rootid,
    __global const int* body_weldid, __global const int* body_dofadr,
    __global const int* body_dofnum, __global const int* dof_parentid,
    __global const real* body_invweight0, __global const real* subtree_com,
    __global const real* cdof, __global const real* qvel,
    __global const int* nivel_corpos, __global const int* nivel_adr,
    __global const int* nivel_num, __global const int* pais,
    __global const int* pais_adr, __global const int* pais_num,
    __global const int* filhos_adr, __global const int* filhos_num,
    __global const int* filhos, __global const int* nivel_de,
    __global const int* body_parentid, __global const int* dof_bodyid,
    __global const real* cdof_dot, __global const real* cinert,
    __global const real* cvel,
    __global const real* dof_damping, __global const int* jnt_type,
    __global const int* jnt_qposadr, __global const int* jnt_dofadr,
    __global const real* jnt_stiffness, __global const real* qpos,
    __global const real* qpos_spring,
    __global const int* actuator_trntype, __global const int* actuator_trnid,
    __global const int* actuator_biastype,
    __global const real* actuator_gainprm, __global const real* actuator_biasprm,
    __global const real* actuator_gear, __global const real* actuator_ctrlrange,
    __global const int* actuator_ctrllimited,
    __global const real* actuator_forcerange,
    __global const int* actuator_forcelimited, __global const real* ctrl,
    __global const real* qfrc_applied,
    __global const int* M_rownnz, __global const int* M_rowadr,
    __global const int* M_colind, __global const real* qLD,
    __global const real* qLDiagInv,
    __global int* n_por_par, __global real* con_dist_bruto,
    __global real* con_pos_bruto, __global real* con_normal_bruto,
    __global real* con_dist, __global real* con_pos, __global real* con_normal,
    __global int* con_geom, __global int* con_pair, __global int* ncon,
    __global real* con_frame, __global real* con_includemargin,
    __global int* con_exclude, __global int* con_efcadr, __global int* nefc,
    __global real* efc_J, __global real* efc_pos, __global real* efc_margin,
    __global int* efc_id, __global real* efc_diagApprox, __global real* efc_R,
    __global real* efc_D, __global real* efc_KBIP, __global real* con_mu,
    __global real* efc_vel, __global real* efc_aref,
    __global real* cacc, __global real* cfrc_body, __global real* qfrc_bias,
    __global real* qfrc_passive, __global real* actuator_length,
    __global real* actuator_velocity, __global real* actuator_force,
    __global real* qfrc_actuator, __global real* ades_momento,
    __global int* ades_conta, __global real* qfrc_smooth,
    __global real* qacc_smooth)
{
    int t = get_local_id(0), W = get_local_size(0);

    // --- contatos a partir do ponto de suporte (que veio de fora)
    for (int p = t; p < npair; p += W)
        contatos_plano_malha_um(p, npair, pair_geom1, pair_geom2, geom_dataid,
                                geom_rbound, geom_xpos, geom_xmat, pair_margin,
                                mesh_vertadr, mesh_vert, mesh_graphadr,
                                mesh_graph, vert_local, sup_vert, sup_dist,
                                n_por_par, con_dist_bruto, con_pos_bruto,
                                con_normal_bruto);
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (t == 0)
        compacta_contatos_um(0, npair, n_por_par, con_dist_bruto,
                             con_pos_bruto, con_normal_bruto, pair_geom1,
                             pair_geom2, con_dist, con_pos, con_normal,
                             con_geom, con_pair, ncon);
    barrier(CLK_GLOBAL_MEM_FENCE);

    // --- restricoes
    for (int i = t; i < NCON_MAX; i += W)
        contato_quadro_um(i, ncon, con_normal, con_pair, pair_margin, pair_gap,
                          con_dist, con_frame, con_includemargin, con_exclude);
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (t == 0) {
        int n = 0;
        for (int i = 0; i < ncon[0]; ++i) {
            if (con_exclude[i]) { con_efcadr[i] = -1; continue; }
            con_efcadr[i] = n;
            n += 4;
        }
        nefc[0] = n;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int gg = t; gg < NCON_MAX*NV; gg += W)
        contato_jacobiana_um(gg, ncon, con_geom, con_pos, con_frame, con_dist,
                             con_includemargin, con_pair, pair_friction,
                             con_efcadr, geom_bodyid, body_rootid, body_weldid,
                             body_dofadr, body_dofnum, dof_parentid,
                             subtree_com, cdof, efc_J, efc_pos, efc_margin,
                             efc_id);
    for (int cc = t; cc < NCON_MAX; cc += W)
        restricao_impedancia_um(cc, ncon, impratio, con_efcadr, con_geom,
                                con_pair, con_dist, con_includemargin,
                                pair_friction, pair_solref, pair_solimp,
                                geom_bodyid, body_invweight0, efc_diagApprox,
                                efc_R, efc_D, efc_KBIP, con_mu);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NEFC_MAX; i += W)
        restricao_aref_um(i, nefc, efc_J, qvel, efc_KBIP, efc_pos, efc_margin,
                          efc_vel, efc_aref);
    barrier(CLK_GLOBAL_MEM_FENCE);

    // --- RNE
    for (int d = 1; d < profundidade; ++d) {
        for (int k = t; k < nivel_num[d]; k += W)
            rne_frente_nivel_um(k, nivel_corpos, nivel_adr, nivel_num, d,
                                body_parentid, body_dofadr, body_dofnum,
                                cdof_dot, qvel, cinert, cvel, cacc, cfrc_body);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    for (int d = profundidade - 1; d >= 1; --d) {
        for (int k = t; k < pais_num[d]; k += W)
            rne_tras_nivel_um(k, pais, pais_adr, pais_num, filhos_adr,
                              filhos_num, filhos, nivel_de, d, cfrc_body);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    for (int i = t; i < NV; i += W)
        rne_projeta_um(i, dof_bodyid, cdof, cfrc_body, qfrc_bias);
    barrier(CLK_GLOBAL_MEM_FENCE);

    // --- passivo, atuacao, adesao, soma
    for (int i = t; i < NV; i += W)
        passivo_amortecedor_um(i, qvel, dof_damping, qfrc_passive);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int j = t; j < njnt; j += W)
        passivo_mola_um(j, njnt, jnt_type, jnt_qposadr, jnt_dofadr,
                        jnt_stiffness, qpos, qpos_spring, qfrc_passive);
    for (int a = t; a < nu; a += W)
        atuacao_um(a, nu, actuator_trntype, actuator_trnid, actuator_biastype,
                   actuator_gainprm, actuator_biasprm, actuator_gear,
                   actuator_ctrlrange, actuator_ctrllimited,
                   actuator_forcerange, actuator_forcelimited, jnt_qposadr,
                   jnt_dofadr, qpos, qvel, ctrl, actuator_length,
                   actuator_velocity, actuator_force, qfrc_actuator);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NV; i += W)
        atuacao_projeta_um(i, nu, actuator_trntype, actuator_trnid,
                           actuator_gear, jnt_dofadr, actuator_force,
                           qfrc_actuator);
    for (int gg = t; gg < nu*NV; gg += W)
        adesao_momento_um(gg, nu, ncon, actuator_trntype, actuator_trnid,
                          con_geom, con_efcadr, geom_bodyid, efc_J,
                          ades_momento, ades_conta);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NV; i += W)
        adesao_projeta_um(i, nu, actuator_trntype, ades_momento,
                          actuator_force, qfrc_actuator);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NV; i += W)
        soma_smooth_um(i, qfrc_passive, qfrc_bias, qfrc_actuator,
                       qfrc_applied, qfrc_smooth);
    barrier(CLK_GLOBAL_MEM_FENCE);

    // --- aceleracao sem restricao
    solve_M_um(t, W, M_rownnz, M_rowadr, M_colind, qLD, qLDiagInv,
               qfrc_smooth, qacc_smooth);
}

#endif  // NBODY && NV && NQ && NEFC_MAX
