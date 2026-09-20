"""
Caminho rapido do laco de fisica: a MESMA conta, sem o trabalho repetido.

## O que isto NAO faz

Nao muda fisica, nao muda controlador, nao muda timestep, nao muda modelo. Nao
ha aproximacao, nao ha atalho numerico, nao ha `float32` no lugar de `float64`.
Cada numero que sai daqui e **bit a bit** o mesmo que sairia do caminho
original -- e isso e verificado, nao afirmado: ver
`tests/test_fastpath_equivalencia.py`, que roda os dois lado a lado por
milhares de passos e reprova na primeira diferenca.

## Por que existe

Perfil do passo de looming com FlyGym 2.x, 3.000 passos, medido:

    controlador      63,5%   918 us/passo
    observacao       16,8%   243 us/passo
    mj_step          11,0%   159 us/passo
    retina            6,1%
    resto             2,6%

**So 11% do tempo esta dentro do `sim.step()`.** O resto e trabalho de Python
em volta da fisica -- e quase todo ele e a mesma coisa sendo refeita a cada
passo, 10.000 vezes por segundo simulado:

  * `dof_spec_to_jointdof` constroi 4 dataclasses congeladas por DOF, 42 vezes
    por passo, so pra usar o resultado como chave de dicionario. O mapa
    (perna, dof) -> posicao na saida nao muda nunca.
  * `_step_phase_gain` remonta dois arrays de 5 elementos a cada chamada, 6
    vezes por passo. Eles dependem so do `swing_period` da perna, que e
    constante.
  * `get_joint_angles` revalida o nome da perna e refaz `np.asarray` por
    chamada, 6 vezes por passo.
  * `get_bodysegment_contact_forces` reconstroi a lista de segmentos, o
    dicionario geom->saida e o array de geoms pedidos a cada chamada, pra 30
    segmentos. Nenhum deles muda durante a corrida.

Nada disso e conta: e bookkeeping. O caminho rapido pre-calcula tudo que e
constante e escreve em buffers ja alocados.

## Por que nao editar o upstream

`research/upstream/flygym` e um clone de leitura. Editar la faria o projeto
depender de um fork silencioso: quem clonasse o repositorio teria um
comportamento diferente do nosso sem nenhum sinal disso. Aqui a subclasse e
nossa, visivel no diff, e o teste de equivalencia e a prova de que ela nao
mudou nada.
"""
from __future__ import annotations

import numpy as np

# constante do `_step_phase_gain` do upstream; nao depende de nada
_INCREMENTOS = np.array([0.0, 0.8, 0.0, -0.1, 0.0])


def _pontos_de_fase(swing_period, swing_extension: float) -> np.ndarray:
    """
    Os `step_points` do `_step_phase_gain`, calculados uma vez por perna.

    Mesma expressao do upstream, na mesma ordem -- `np.mean` de dois elementos
    e `+` em float64 dao o mesmo bit toda vez.
    """
    inicio, fim = float(swing_period[0]), float(swing_period[1])
    return np.array([
        inicio,
        np.mean([inicio, fim]),
        fim + swing_extension,
        np.mean([fim, 2 * np.pi]),
        2 * np.pi,
    ])


class ControladorRapido:
    """
    Envolve o `HybridTurningController` e refaz o `step()` sem o bookkeeping.

    Nao herda: compoe. Herdar exigiria confiar em qual metodo privado o
    upstream chama de qual, e uma versao nova quebraria em silencio. Compondo,
    tudo que e chamado esta escrito aqui embaixo, na cara.

    O estado continua sendo o do controlador original -- `cpg_network`,
    `retraction_correction`, `stumbling_correction`, contadores. Este objeto
    nao guarda estado de simulacao nenhum, so tabelas constantes.
    """

    def __init__(self, ctl):
        from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
        from flygym_demo.complex_terrain.hybrid_controller import (
            _CORRECTION_VECTORS, _RIGHT_LEG_CORRECTION_SIGN,
        )

        self.ctl = ctl
        ps = ctl.preprogrammed_steps
        self.pernas = tuple(ctl.legs)
        self.n_dofs_por_perna = len(ps.dofs_per_leg)

        ordem = ctl.output_dof_order
        if ordem is None:
            from flygym_demo.complex_terrain.common import (
                get_default_locomotion_dof_order,
            )
            ordem = get_default_locomotion_dof_order()
        self.ordem_saida = list(ordem)
        posicao = {dof: i for i, dof in enumerate(self.ordem_saida)}

        # (perna, dof) -> posicao no vetor de saida. E o `dof_spec_to_jointdof`
        # de 42 chamadas por passo, resolvido uma vez.
        self.indices = np.empty((len(self.pernas), self.n_dofs_por_perna),
                                dtype=np.int64)
        for i, perna in enumerate(self.pernas):
            for j, spec in enumerate(ps.dofs_per_leg):
                jd = dof_spec_to_jointdof(perna, spec)
                if jd not in posicao:
                    raise KeyError(
                        f"{perna}/{spec} nao esta na ordem de saida do "
                        "controlador; o caminho rapido nao cobre esta "
                        "configuracao")
                self.indices[i, j] = posicao[jd]

        # constantes por perna
        self.psi = [ps._psi_funcs[p] for p in self.pernas]
        self.psi_junto = self._empilha_splines()
        self.neutro = [np.asarray(ps.neutral_pos[p]) for p in self.pernas]
        self.pontos_fase = [
            _pontos_de_fase(ps.swing_period[p], ctl.swing_extension)
            for p in self.pernas
        ]
        # O controlador ESTENDE o fim do swing por `swing_extension` antes de
        # decidir a adesao -- o `get_adhesion_onoff` do passo pre-programado
        # nao faz isso. Usar o do passo pre-programado aqui trocava a adesao de
        # todas as pernas; o teste bit a bit pegou no primeiro passo.
        self.swing = [(float(ps.swing_period[p][0]),
                       float(ps.swing_period[p][1]) + ctl.swing_extension)
                      for p in self.pernas]
        self.vetor_corr = []
        for p in self.pernas:
            v = _CORRECTION_VECTORS[p[1]]
            if p.startswith("r"):
                v = v * _RIGHT_LEG_CORRECTION_SIGN
            self.vetor_corr.append(np.asarray(v))

        # buffers reaproveitados entre passos
        self._saida = np.zeros(len(self.ordem_saida), dtype=float)
        self._adesao = np.zeros(len(self.pernas), dtype=bool)
        self._fase = np.zeros(1, dtype=float)
        self._correcoes = np.zeros(len(self.pernas), dtype=float)

    def _empilha_splines(self):
        """
        Um PPoly so, com as seis pernas, quando isso for possivel.

        As seis `CubicSpline` do passo pre-programado tem os MESMOS 45 nos --
        so os coeficientes mudam. Empilhando os coeficientes no eixo de saida
        da pra avaliar as seis numa chamada em vez de seis, e a chamada do
        scipy custa ~26 us de overhead de Python cada uma, contra microssegundos
        de conta.

        Avalia mais do que precisa (as seis pernas em cada uma das seis fases,
        e so a diagonal e usada), e ainda assim sai na frente: o custo aqui e
        overhead por CHAMADA, nao por ponto.

        Devolve None se as pernas nao compartilharem os nos -- ai o caminho por
        perna continua valendo, sem nenhuma perda de correcao.
        """
        from scipy.interpolate import PPoly

        base = self.psi[0]
        x = np.asarray(base.x)
        for f in self.psi[1:]:
            if not np.array_equal(np.asarray(f.x), x):
                return None
            if f.extrapolate != base.extrapolate:
                return None
        coef = np.concatenate([np.asarray(f.c) for f in self.psi], axis=2)
        self._saidas_por_perna = np.asarray(base.c).shape[2]
        # `construct_fast` nao revalida nem rola eixo: o resultado sai como
        # (n_pontos, n_saidas), que e a forma que o laco quer.
        return PPoly.construct_fast(coef, x, extrapolate=base.extrapolate)

    def step(self, sinal_descendente, obs):
        """Mesma assinatura e mesma saida de `HybridTurningController.step`."""
        from flygym_demo.complex_terrain.hybrid_controller import LocomotionAction

        ctl = self.ctl
        sinal = np.asarray(sinal_descendente, dtype=float)
        if sinal.shape != (2,):
            raise ValueError("descending_signal must have shape (2,).")

        # --- o que o HybridTurningController faz antes de delegar ---
        ctl.cpg_network.intrinsic_amps = np.repeat(
            np.abs(sinal[:, np.newaxis]), 3, axis=1).ravel()
        freqs = ctl._base_intrinsic_freqs.copy()
        freqs[:3] *= 1 if sinal[0] >= 0 else -1
        freqs[3:] *= 1 if sinal[1] >= 0 else -1
        ctl.cpg_network.intrinsic_freqs = freqs

        # --- o corpo do HybridController.step ---
        perna_retracao = ctl._select_retraction_leg(obs)
        if perna_retracao is not None:
            if (ctl.retraction_correction[perna_retracao]
                    > ctl.retraction_persistence_initiation_threshold):
                ctl.retraction_persistence_counter[perna_retracao] = 1
        ctl._update_persistence_counter()

        mascara = ctl._get_stumbling_mask(obs)
        ctl.cpg_network.step()

        fases = ctl.cpg_network.curr_phases
        magnitudes = ctl.cpg_network.curr_magnitudes
        saida, adesao = self._saida, self._adesao
        dois_pi = 2 * np.pi
        # uma chamada ao scipy pelas seis pernas, quando elas compartilham nos
        psi_todas = (self.psi_junto(fases) if self.psi_junto is not None
                     else None)

        for i, perna in enumerate(self.pernas):
            ctl._update_retraction_correction(i, perna_retracao)
            ctl._update_stumbling_correction(i, mascara[i])

            if ctl.retraction_correction[i] > 0:
                correcao = ctl.retraction_correction[i]
                ctl.stumbling_correction[i] = 0
            else:
                correcao = ctl.stumbling_correction[i]

            fase = fases[i]
            magnitude = magnitudes[i]

            # get_joint_angles, sem a revalidacao do nome da perna
            neutro = self.neutro[i]
            if psi_todas is None:
                self._fase[0] = fase
                psi_i = self.psi[i](self._fase)
            else:
                k = i * self._saidas_por_perna
                psi_i = psi_todas[i, k:k + self._saidas_por_perna, np.newaxis]
            angulos = (neutro + magnitude * (psi_i - neutro))[:, 0]

            # `np.clip` num escalar constroi um array de zero dimensoes e
            # volta; `min`/`max` dao o MESMO valor sem alocar. Sao 6 chamadas
            # por passo, 60.000 por segundo simulado.
            teto = ctl.max_correction
            correcao = 0.0 if correcao < 0 else (teto if correcao > teto
                                                 else correcao)
            ganho = float(np.interp(fase % dois_pi, self.pontos_fase[i],
                                    _INCREMENTOS))
            angulos = angulos + correcao * ganho * self.vetor_corr[i]
            self._correcoes[i] = correcao * ganho

            saida[self.indices[i]] = angulos
            # `_get_adhesion_onoff` faz `swing_period[leg.lower()]` por
            # chamada; o par ja esta resolvido aqui
            if ctl.enable_adhesion:
                ini, fim = self.swing[i]
                adesao[i] = not (ini < fase % dois_pi < fim)
            else:
                adesao[i] = False

        # O upstream publica `last_info` a cada passo. Nada no Drosobot le, mas
        # publicar custa quatro copias de seis elementos -- barato demais pra
        # justificar divergir da API.
        ctl.last_info = {
            "net_corrections": self._correcoes.copy(),
            "retraction_correction": ctl.retraction_correction.copy(),
            "stumbling_correction": ctl.stumbling_correction.copy(),
            "stumbling_mask": mascara.copy(),
            "leg_to_correct_retraction": perna_retracao,
        }
        # copia na saida: quem recebe pode guardar a acao, e o buffer e reusado
        return LocomotionAction(joint_angles=saida.copy(),
                                adhesion_onoff=adesao.copy())


class ForcasContato:
    """
    `get_bodysegment_contact_forces` com as tabelas resolvidas uma vez.

    O upstream reconstroi, a cada chamada: a lista de `BodySegment`, o
    dicionario geom->saida e o array de geoms pedidos. Para 30 segmentos isso e
    30 dataclasses congeladas, 30 hashes e dois `np.isin` de construcao por
    passo. Nenhuma dessas coisas muda durante a corrida.
    """

    def __init__(self, sim, fly_name: str, segmentos, ground_only: bool = True):
        from flygym.anatomy import BodySegment

        self.sim = sim
        self.ground_only = ground_only
        segs = [s if isinstance(s, BodySegment) else BodySegment(s)
                for s in segmentos]
        por_seg = sim._internal_geomid_by_bodyseg_by_fly[fly_name]
        self.geom_para_saida = {por_seg[s]: i for i, s in enumerate(segs)}
        self.geoms = np.array(list(self.geom_para_saida.keys()), dtype=np.int32)
        self.chao = sim._internal_ground_geom_ids

        # `np.isin` e o jeito certo quando o conjunto e desconhecido; aqui ele
        # e fixo e pequeno, e o id do geom ja e o indice. Uma tabela booleana
        # por geom troca quatro varreduras por quatro indexacoes -- mesmo
        # resultado, sem construir conjunto por chamada.
        ngeom = int(sim.mj_model.ngeom)
        self.eh_pedido = np.zeros(ngeom, dtype=bool)
        self.eh_pedido[self.geoms] = True
        self.eh_chao = np.zeros(ngeom, dtype=bool)
        self.eh_chao[np.asarray(self.chao, dtype=np.int32)] = True
        self.forcas = np.zeros((len(segs), 3), dtype=float)
        self._wrench = np.zeros(6, dtype=float)

    def le(self) -> np.ndarray:
        import mujoco as mj

        forcas = self.forcas
        forcas[:] = 0.0
        d = self.sim.mj_data
        ncon = d.ncon
        if ncon == 0:
            return forcas

        contatos = d.contact
        geom1 = contatos.geom1[:ncon]
        geom2 = contatos.geom2[:ncon]
        excluido = contatos.exclude[:ncon].astype(bool)

        g1_pedido = self.eh_pedido[geom1]
        g2_pedido = self.eh_pedido[geom2]
        ativo = (g1_pedido | g2_pedido) & ~excluido
        if self.ground_only:
            g1_chao = self.eh_chao[geom1]
            g2_chao = self.eh_chao[geom2]
            ativo &= (g1_pedido & g2_chao) | (g2_pedido & g1_chao)

        wrench = self._wrench
        mapa = self.geom_para_saida
        for cid in np.where(ativo)[0]:
            mj.mj_contactForce(self.sim.mj_model, d, int(cid), wrench)
            quadro = contatos.frame[cid].reshape(3, 3)
            forca_mundo = quadro.T @ wrench[:3]
            a, b = int(geom1[cid]), int(geom2[cid])
            if a in mapa:
                forcas[mapa[a]] -= forca_mundo
            if b in mapa:
                forcas[mapa[b]] += forca_mundo
        return forcas
