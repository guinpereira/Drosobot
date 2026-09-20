// Solver de restricao: Newton projetado sobre o objetivo do MuJoCo 3.9.0.
//
// OBRA DERIVADA. As SEMANTICAS vem de:
//     MuJoCo 3.9.0, src/engine/engine_core_constraint.c
//         mj_constraintUpdate_impl (custo e forca por linha)
//     MuJoCo 3.9.0, src/engine/engine_solver.c
//         mj_solveNewton (objetivo, Hessiano, criterio de parada)
// Copyright 2021 DeepMind Technologies Limited. Apache License, Version 2.0
// <https://www.apache.org/licenses/LICENSE-2.0>. Ver THIRD_PARTY_NOTICES.md.
//
// ## O que e igual e o que nao e
//
// IGUAL: o problema. Para o subconjunto do NeuroMechFly -- contato piramidal,
// `condim = 3`, sem igualdade, sem atrito de dof, sem tendao -- o objetivo e
//
//     custo(a) = 1/2 (a - a_s)' M (a - a_s) + sum_i 1/2 D_i min(jar_i, 0)^2
//     jar      = J a - aref
//
// que e estritamente convexo e C1. O minimo e UNICO, e `efc_force_i` e
// `-D_i jar_i` nas linhas ativas e zero nas satisfeitas -- termo a termo o que
// `mj_constraintUpdate_impl` faz no ramo de contato nao eliptico.
//
// DIFERENTE: o caminho ate ele. O Newton do MuJoCo fatora o Hessiano uma vez
// por passo e depois faz atualizacoes Cholesky de posto 1 conforme o conjunto
// ativo muda, com uma busca de linha exata por partes. Aqui o Hessiano e
// refatorado a cada iteracao, denso, e a busca de linha e um recuo de Armijo.
//
// Isso e deliberado e e a ordem certa de construir: como o minimo e unico, os
// dois convergem para a MESMA solucao dentro da tolerancia, e comparar a
// solucao -- nao a trajetoria de iterados -- e o que valida a fisica. Uma
// formulacao dual, mais rapida, vira depois e sera comparada contra ESTE
// solver e contra o MuJoCo, o que reduz a superficie de depuracao.
//
// Um work-group. `H` (nv x nv) fica em `__local`: 72*72*8 = 41 KiB dos 64 KiB.

#if defined(NV) && defined(NEFC_MAX)

#define SOLVER_MAXITER 100
#define SOLVER_LS_MAXITER 30

// Expande a matriz de massa esparsa (CSR triangular inferior) para densa
// simetrica. O Hessiano `M + J' D J` tem preenchimento total, entao nao ha o
// que preservar da esparsidade de M aqui.
__kernel void M_densa(
    __global const int* M_rownnz, __global const int* M_rowadr,
    __global const int* M_colind, __global const real* M, __global real* Md)
{
    int i = get_global_id(0);
    if (i >= NV) return;
    for (int j = 0; j < NV; ++j) Md[i*NV + j] = REAL_ZERO;
    int adr = M_rowadr[i], n = M_rownnz[i];
    for (int k = 0; k < n; ++k) {
        int j = M_colind[adr + k];
        Md[i*NV + j] = M[adr + k];
    }
}

__kernel void M_simetriza(__global real* Md) {
    int i = get_global_id(0);
    if (i >= NV) return;
    for (int j = i + 1; j < NV; ++j) Md[i*NV + j] = Md[j*NV + i];
}

// ------------------------------------------------------------------ Newton

__kernel void solver_newton(
    __global const int* nefc,
    __global const real* efc_J, __global const real* efc_D,
    __global const real* efc_aref, __global const real* Md,
    __global const real* qacc_smooth, const real tol, const real meaninertia,
    __global real* qacc_warmstart, const int usa_warmstart,
    __global real* qacc, __global real* efc_force,
    __global real* qfrc_constraint, __global real* jar_out,
    __global int* iters)
{
    __local real H[NV*NV];
    __local real g[NV], sdir[NV], a[NV], atmp[NV], Mda[NV];
    __local real red[256];
    __local int ativo_n;

    int t = get_local_id(0), W = get_local_size(0);
    int n = nefc[0];

    // Warm start, como no `mj_fwdConstraint`: o ponto de partida e o `qacc`
    // do passo anterior. Nao muda o MINIMO -- o problema e estritamente
    // convexo -- mas muda o caminho, e com ele quantas iteracoes custam. E o
    // que o MuJoCo faz, entao partir de `qacc_smooth` compararia dois
    // algoritmos em regimes diferentes.
    for (int i = t; i < NV; i += W)
        a[i] = usa_warmstart ? qacc_warmstart[i] : qacc_smooth[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // sem restricao: o passo suave ja e a resposta
    if (n == 0) {
        for (int i = t; i < NV; i += W) {
            qacc[i] = a[i];
            qacc_warmstart[i] = a[i];
            qfrc_constraint[i] = REAL_ZERO;
        }
        if (t == 0) iters[0] = 0;
        return;
    }

    // `scale` do criterio de parada: mesma normalizacao do mj_solveNewton,
    // para que `tol` signifique aqui o que significa la.
    real scale = REAL_ONE / fmax(MINVAL, meaninertia * (real)(NV > 1 ? NV : 1));
    int iter = 0;

    for (iter = 0; iter < SOLVER_MAXITER; ++iter) {
        // --- jar = J a - aref, forca e conjunto ativo ---------------------
        for (int i = t; i < n; i += W) {
            __global const real* row = efc_J + (size_t)i*NV;
            real v = REAL_ZERO;
            for (int k = 0; k < NV; ++k) v += row[k]*a[k];
            real jar = v - efc_aref[i];
            jar_out[i] = jar;
            efc_force[i] = (jar < REAL_ZERO) ? -efc_D[i]*jar : REAL_ZERO;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // --- Mda = M (a - a_s);  grad = Mda - J' force --------------------
        for (int i = t; i < NV; i += W) {
            real s = REAL_ZERO;
            for (int j = 0; j < NV; ++j) s += Md[i*NV + j]*(a[j] - qacc_smooth[j]);
            Mda[i] = s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = t; i < NV; i += W) {
            real s = REAL_ZERO;
            for (int k = 0; k < n; ++k) s += efc_J[(size_t)k*NV + i]*efc_force[k];
            g[i] = Mda[i] - s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- norma do gradiente: criterio de parada -----------------------
        real acc = REAL_ZERO;
        for (int i = t; i < NV; i += W) acc += g[i]*g[i];
        red[t] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = W >> 1; s > 0; s >>= 1) {
            if (t < s) red[t] += red[t+s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (scale*sqrt(red[0]) < tol) break;

        // --- H = M + J' diag(D_ativo) J -----------------------------------
        for (int idx = t; idx < NV*NV; idx += W) H[idx] = Md[idx];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = t; i < NV; i += W) {
            for (int k = 0; k < n; ++k) {
                if (jar_out[k] >= REAL_ZERO) continue;
                real jik = efc_J[(size_t)k*NV + i];
                if (jik == REAL_ZERO) continue;
                real dk = efc_D[k];
                __global const real* row = efc_J + (size_t)k*NV;
                for (int j = 0; j <= i; ++j) H[i*NV + j] += dk*jik*row[j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Cholesky densa no triangulo inferior -------------------------
        for (int c = 0; c < NV; ++c) {
            if (t == 0) {
                real dd = H[c*NV + c];
                H[c*NV + c] = sqrt(fmax(dd, MINVAL));
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            real inv = REAL_ONE/H[c*NV + c];
            for (int i = c + 1 + t; i < NV; i += W) H[i*NV + c] *= inv;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = c + 1 + t; i < NV; i += W) {
                real lic = H[i*NV + c];
                for (int j = c + 1; j <= i; ++j) H[i*NV + j] -= lic*H[j*NV + c];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // --- sdir = -H^-1 g  (duas substituicoes, seriais por natureza) ---
        if (t == 0) {
            for (int i = 0; i < NV; ++i) {
                real s = -g[i];
                for (int j = 0; j < i; ++j) s -= H[i*NV + j]*sdir[j];
                sdir[i] = s / H[i*NV + i];
            }
            for (int i = NV - 1; i >= 0; --i) {
                real s = sdir[i];
                for (int j = i + 1; j < NV; ++j) s -= H[j*NV + i]*sdir[j];
                sdir[i] = s / H[i*NV + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- busca de linha: recuo de Armijo sobre o custo ----------------
        // `gTs` e a inclinacao; com H definida positiva ela e negativa, e a
        // direcao e de descida.
        acc = REAL_ZERO;
        for (int i = t; i < NV; i += W) acc += g[i]*sdir[i];
        red[t] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = W >> 1; s > 0; s >>= 1) {
            if (t < s) red[t] += red[t+s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        real gTs = red[0];
        if (gTs >= REAL_ZERO) break;           // direcao nao desce: parar

        real custo0 = REAL_ZERO;
        {
            real s1 = REAL_ZERO;
            for (int i = t; i < NV; i += W) s1 += (a[i]-qacc_smooth[i])*Mda[i];
            red[t] = s1;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int s = W >> 1; s > 0; s >>= 1) {
                if (t < s) red[t] += red[t+s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            real q = (real)0.5*red[0];
            real s2 = REAL_ZERO;
            for (int i = t; i < n; i += W) {
                real jr = jar_out[i];
                if (jr < REAL_ZERO) s2 += (real)0.5*efc_D[i]*jr*jr;
            }
            red[t] = s2;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int s = W >> 1; s > 0; s >>= 1) {
                if (t < s) red[t] += red[t+s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            custo0 = q + red[0];
        }

        real alpha = REAL_ONE;
        int ok = 0;
        for (int ls = 0; ls < SOLVER_LS_MAXITER; ++ls) {
            for (int i = t; i < NV; i += W) atmp[i] = a[i] + alpha*sdir[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            // custo(atmp)
            real s1 = REAL_ZERO;
            for (int i = t; i < NV; i += W) {
                real dloc = REAL_ZERO;
                for (int j = 0; j < NV; ++j)
                    dloc += Md[i*NV + j]*(atmp[j] - qacc_smooth[j]);
                s1 += (atmp[i]-qacc_smooth[i])*dloc;
            }
            red[t] = s1;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int s = W >> 1; s > 0; s >>= 1) {
                if (t < s) red[t] += red[t+s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            real q = (real)0.5*red[0];
            real s2 = REAL_ZERO;
            for (int i = t; i < n; i += W) {
                __global const real* row = efc_J + (size_t)i*NV;
                real v = REAL_ZERO;
                for (int k = 0; k < NV; ++k) v += row[k]*atmp[k];
                real jr = v - efc_aref[i];
                if (jr < REAL_ZERO) s2 += (real)0.5*efc_D[i]*jr*jr;
            }
            red[t] = s2;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int s = W >> 1; s > 0; s >>= 1) {
                if (t < s) red[t] += red[t+s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            real custo = q + red[0];
            if (custo <= custo0 + (real)1e-4*alpha*gTs) { ok = 1; break; }
            alpha *= (real)0.5;
        }
        if (!ok) break;
        for (int i = t; i < NV; i += W) a[i] += alpha*sdir[i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- forca final e projecao nos dofs ---------------------------------
    for (int i = t; i < n; i += W) {
        __global const real* row = efc_J + (size_t)i*NV;
        real v = REAL_ZERO;
        for (int k = 0; k < NV; ++k) v += row[k]*a[k];
        real jr = v - efc_aref[i];
        jar_out[i] = jr;
        efc_force[i] = (jr < REAL_ZERO) ? -efc_D[i]*jr : REAL_ZERO;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NV; i += W) {
        real s = REAL_ZERO;
        for (int k = 0; k < n; ++k) s += efc_J[(size_t)k*NV + i]*efc_force[k];
        qfrc_constraint[i] = s;
        qacc[i] = a[i];
        qacc_warmstart[i] = a[i];
    }
    if (t == 0) iters[0] = iter;
}


// ------------------------------------------------- Newton, caminho rapido
//
// Mesmo problema, mesmo objetivo, mesma tolerancia. Duas mudancas de
// organizacao, cada uma medida:
//
//   1. **M nunca e densificada.** O caminho de referencia expandia a matriz de
//      massa para `nv x nv` em memoria global para poder multiplicar por ela.
//      Isso custava 170 us por passo -- mais do que o solver inteiro deveria
//      custar. Aqui o produto usa um CSR SIMETRICO de 1554 entradas, montado
//      uma vez no setup a partir do triangular do MuJoCo
//      (`estrutura.massa_simetrica`). O Hessiano continua denso, porque a
//      fatoracao precisa, mas vive em LDS e nunca vai para memoria global.
//
//   2. **A busca de linha custa O(nefc), nao O(nv^2).** Ao longo de uma
//      direcao `s`, o custo e um polinomio:
//
//          q(alpha)   = q0 + alpha*(s'M(a-a_s)) + alpha^2/2*(s'Ms)
//          jar(alpha) = jar0 + alpha*(J s)
//
//      entao `Ms` e `Js` saem UMA vez por iteracao de Newton, e cada tentativa
//      de `alpha` vira uma varredura sobre as linhas de restricao. Antes, cada
//      tentativa refazia um produto matriz-vetor cheio. E o mesmo truque que o
//      `PrimalSearch` do MuJoCo usa com `Mv` e `Jv`.
//
// A ORDEM das somas muda em relacao ao caminho de referencia -- o produto
// esparso soma na ordem do CSR, o denso somava por coluna -- entao os dois
// concordam ate arredondamento, nao bit a bit. O teste compara com tolerancia
// por isso, e o caminho de referencia continua existindo para quando a
// diferenca importar.

__kernel void solver_newton_rapido(
    __global const int* nefc,
    __global const real* efc_J, __global const real* efc_D,
    __global const real* efc_aref,
    __global const int* Ms_rownnz, __global const int* Ms_rowadr,
    __global const int* Ms_colind, __global const int* Ms_mapa,
    __global const real* M,
    __global const real* qacc_smooth, const real tol, const real meaninertia,
    __global real* qacc_warmstart, const int usa_warmstart,
    __global real* qacc, __global real* efc_force,
    __global real* qfrc_constraint, __global real* jar_out,
    __global real* Jv, __global int* iters)
{
    __local real H[NV*NV];
    __local real g[NV], sdir[NV], a[NV], Mda[NV], Msd[NV];
    __local real red[256];

    int t = get_local_id(0), W = get_local_size(0);
    int n = nefc[0];

    for (int i = t; i < NV; i += W)
        a[i] = usa_warmstart ? qacc_warmstart[i] : qacc_smooth[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (n == 0) {
        for (int i = t; i < NV; i += W) {
            qacc[i] = a[i];
            qacc_warmstart[i] = a[i];
            qfrc_constraint[i] = REAL_ZERO;
        }
        if (t == 0) iters[0] = 0;
        return;
    }

    real scale = REAL_ONE / fmax(MINVAL, meaninertia * (real)(NV > 1 ? NV : 1));
    int iter = 0;

    for (iter = 0; iter < SOLVER_MAXITER; ++iter) {
        for (int i = t; i < n; i += W) {
            __global const real* row = efc_J + (size_t)i*NV;
            real v = REAL_ZERO;
            for (int k = 0; k < NV; ++k) v += row[k]*a[k];
            real jar = v - efc_aref[i];
            jar_out[i] = jar;
            efc_force[i] = (jar < REAL_ZERO) ? -efc_D[i]*jar : REAL_ZERO;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int i = t; i < NV; i += W) {
            int adr = Ms_rowadr[i], nn = Ms_rownnz[i];
            real acc = REAL_ZERO;
            for (int k = 0; k < nn; ++k) {
                int j = Ms_colind[adr + k];
                acc += M[Ms_mapa[adr + k]] * (a[j] - qacc_smooth[j]);
            }
            Mda[i] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = t; i < NV; i += W) {
            real acc = REAL_ZERO;
            for (int k = 0; k < n; ++k) acc += efc_J[(size_t)k*NV + i]*efc_force[k];
            g[i] = Mda[i] - acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        real acc = REAL_ZERO;
        for (int i = t; i < NV; i += W) acc += g[i]*g[i];
        red[t] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int sft = W >> 1; sft > 0; sft >>= 1) {
            if (t < sft) red[t] += red[t+sft];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (scale*sqrt(red[0]) < tol) break;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int idx = t; idx < NV*NV; idx += W) H[idx] = REAL_ZERO;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = t; i < NV; i += W) {
            int adr = Ms_rowadr[i], nn = Ms_rownnz[i];
            for (int k = 0; k < nn; ++k)
                H[i*NV + Ms_colind[adr + k]] = M[Ms_mapa[adr + k]];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = t; i < NV; i += W) {
            for (int k = 0; k < n; ++k) {
                if (jar_out[k] >= REAL_ZERO) continue;
                real jik = efc_J[(size_t)k*NV + i];
                if (jik == REAL_ZERO) continue;
                real dk = efc_D[k];
                __global const real* row = efc_J + (size_t)k*NV;
                for (int j = 0; j <= i; ++j) H[i*NV + j] += dk*jik*row[j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int c = 0; c < NV; ++c) {
            if (t == 0) H[c*NV + c] = sqrt(fmax(H[c*NV + c], MINVAL));
            barrier(CLK_LOCAL_MEM_FENCE);
            real inv = REAL_ONE/H[c*NV + c];
            for (int i = c + 1 + t; i < NV; i += W) H[i*NV + c] *= inv;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = c + 1 + t; i < NV; i += W) {
                real lic = H[i*NV + c];
                for (int j = c + 1; j <= i; ++j) H[i*NV + j] -= lic*H[j*NV + c];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0) {
            for (int i = 0; i < NV; ++i) {
                real sv = -g[i];
                for (int j = 0; j < i; ++j) sv -= H[i*NV + j]*sdir[j];
                sdir[i] = sv / H[i*NV + i];
            }
            for (int i = NV - 1; i >= 0; --i) {
                real sv = sdir[i];
                for (int j = i + 1; j < NV; ++j) sv -= H[j*NV + i]*sdir[j];
                sdir[i] = sv / H[i*NV + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // pre-calculo da busca de linha
        for (int i = t; i < NV; i += W) {
            int adr = Ms_rowadr[i], nn = Ms_rownnz[i];
            real s2 = REAL_ZERO;
            for (int k = 0; k < nn; ++k)
                s2 += M[Ms_mapa[adr + k]] * sdir[Ms_colind[adr + k]];
            Msd[i] = s2;
        }
        for (int i = t; i < n; i += W) {
            __global const real* row = efc_J + (size_t)i*NV;
            real v = REAL_ZERO;
            for (int k = 0; k < NV; ++k) v += row[k]*sdir[k];
            Jv[i] = v;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        real r1 = REAL_ZERO, r2 = REAL_ZERO, r3 = REAL_ZERO;
        for (int i = t; i < NV; i += W) {
            r1 += g[i]*sdir[i];
            r2 += sdir[i]*Mda[i];
            r3 += sdir[i]*Msd[i];
        }
        red[t] = r1;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int sft = W >> 1; sft > 0; sft >>= 1) {
            if (t < sft) red[t] += red[t+sft];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        real gTs = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        red[t] = r2;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int sft = W >> 1; sft > 0; sft >>= 1) {
            if (t < sft) red[t] += red[t+sft];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        real sMda = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        red[t] = r3;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int sft = W >> 1; sft > 0; sft >>= 1) {
            if (t < sft) red[t] += red[t+sft];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        real sMs = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (gTs >= REAL_ZERO) break;

        real s0 = REAL_ZERO;
        for (int i = t; i < n; i += W) {
            real jr = jar_out[i];
            if (jr < REAL_ZERO) s0 += (real)0.5*efc_D[i]*jr*jr;
        }
        red[t] = s0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int sft = W >> 1; sft > 0; sft >>= 1) {
            if (t < sft) red[t] += red[t+sft];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        real c0 = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        real alpha = REAL_ONE;
        int ok = 0;
        for (int ls = 0; ls < SOLVER_LS_MAXITER; ++ls) {
            real sc = REAL_ZERO;
            for (int i = t; i < n; i += W) {
                real jr = jar_out[i] + alpha*Jv[i];
                if (jr < REAL_ZERO) sc += (real)0.5*efc_D[i]*jr*jr;
            }
            red[t] = sc;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int sft = W >> 1; sft > 0; sft >>= 1) {
                if (t < sft) red[t] += red[t+sft];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            real dcusto = alpha*sMda + (real)0.5*alpha*alpha*sMs + (red[0] - c0);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (dcusto <= (real)1e-4*alpha*gTs) { ok = 1; break; }
            alpha *= (real)0.5;
        }
        if (!ok) break;
        for (int i = t; i < NV; i += W) a[i] += alpha*sdir[i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = t; i < n; i += W) {
        __global const real* row = efc_J + (size_t)i*NV;
        real v = REAL_ZERO;
        for (int k = 0; k < NV; ++k) v += row[k]*a[k];
        real jr = v - efc_aref[i];
        jar_out[i] = jr;
        efc_force[i] = (jr < REAL_ZERO) ? -efc_D[i]*jr : REAL_ZERO;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = t; i < NV; i += W) {
        real acc = REAL_ZERO;
        for (int k = 0; k < n; ++k) acc += efc_J[(size_t)k*NV + i]*efc_force[k];
        qfrc_constraint[i] = acc;
        qacc[i] = a[i];
        qacc_warmstart[i] = a[i];
    }
    if (t == 0) iters[0] = iter;
}

#endif  // NV && NEFC_MAX
