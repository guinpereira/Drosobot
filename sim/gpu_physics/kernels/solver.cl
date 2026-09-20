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

#endif  // NV && NEFC_MAX
