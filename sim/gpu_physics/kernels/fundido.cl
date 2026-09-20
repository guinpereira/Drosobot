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

    factor_M_um(t, W, M_rownnz, M_rowadr, M_colind, M, qLD, qLDiagInv);
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
    solve_M_um(get_local_id(0), get_local_size(0), M_rownnz, M_rowadr,
               M_colind, qLD, qLDiagInv, qfrc_smooth, qacc_smooth);
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
    int t = get_local_id(0), W = get_local_size(0);

    for (int i = t; i < NC; i += W) euler_copia_M_um(i, M, qH_in);
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NV; i += W)
        euler_diag_MhD_um(i, dt, M_rownnz, M_rowadr, dof_damping, qH_in);
    barrier(CLK_GLOBAL_MEM_FENCE);

    factor_M_um(t, W, M_rownnz, M_rowadr, M_colind, qH_in, qH, qHDiagInv);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < NV; i += W)
        euler_rhs_um(i, qfrc_smooth, qfrc_constraint, rhs);
    barrier(CLK_GLOBAL_MEM_FENCE);

    solve_M_um(t, W, M_rownnz, M_rowadr, M_colind, qH, qHDiagInv, rhs,
               qacc_euler);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < NV; i += W) euler_qvel_um(i, dt, qacc_euler, qvel);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = t; i < njnt; i += W)
        euler_qpos_um(i, dt, njnt, jnt_type, jnt_qposadr, jnt_dofadr, qvel,
                      qpos);
}

#endif  // NBODY && NV && NQ && NEFC_MAX
