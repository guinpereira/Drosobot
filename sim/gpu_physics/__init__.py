r"""
Drosobot GPU Physics: um backend de fisica para o subconjunto do NeuroMechFly.

    inventario    o que o modelo real usa do MuJoCo (nao o MuJoCo inteiro)
    compilador    mjModel -> modelo plano, validacao do subset, hash do corpo
    estrutura     a arvore de corpos reorganizada em niveis paralelos
    device        device, buffers, dispatch. Nada de fisica
    cinematica    cinematica direta residente na GPU
    kernels/      os kernels OpenCL

O MuJoCo CPU e a REFERENCIA, nao o concorrente: e dele que sai o modelo, e
contra ele que cada estagio e validado.

## Leia isto antes de usar

Este backend **ainda nao resolve o passo inteiro**. So a cinematica direta roda
de fato na GPU; inercia, colisao, restricoes, solver, atuacao e integracao
continuam no MuJoCo. Por isso `physics.cria("drosobot-gpu")` levanta
`BackendIncompleto` em vez de montar -- gravar o nome no metadata com o MuJoCo
integrando por tras seria procedencia falsa.

E a medicao diz que continuar nao e obvio. Com `nv=72` e um mundo, esta placa
perde em todos os estagios medidos: 84 us por despacho sincrono, 40 us por
Cholesky densa 72x72 num work-group, 12,3 us de cinematica contra 6,3 us da
CPU -- e o tempo da cinematica e plano de 16 a 256 threads, o que mostra que
nao falta paralelismo: o que limita e a cadeia serial da arvore.

Os numeros, o metodo e o que mudaria a resposta estao em `docs/GPU_PHYSICS.md`.
"""
