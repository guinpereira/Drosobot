"""
Exporta a mosca do NeuroMechFly pra Unity. Nada de mosca generica.

    .venv-flygym2\\Scripts\\python tools\\export_fly_mesh.py

Sai em unity/DrosobotLab/Assets/Resources/Fly/:

    *.obj            um por MALHA do modelo
    fly_body.json    hierarquia, transform local de cada geom, escala, eixos

## Por que nao basta copiar os STL

Unity nao importa STL nativamente, entao convertemos pra OBJ. Mas o essencial
nao e o formato -- e o METADADO. Uma malha solta nao monta uma mosca: cada geom
tem posicao e orientacao LOCAIS em relacao ao corpo dele, e e isso que faz o
femur ficar preso na coxa no lugar certo.

Tudo aqui sai do modelo COMPILADO, nao de leitura de arquivo: e o mesmo modelo
que a fisica roda, entao o que a Unity desenha corresponde ao que esta sendo
simulado.

## Convencoes que a Unity precisa saber, e que vao no JSON

    unidades      MuJoCo do NeuroMechFly trabalha em MILIMETROS
    quaternio     MuJoCo e (w, x, y, z); Unity e (x, y, z, w)
    eixos         MuJoCo e Z-up destro; Unity e Y-up canhoto

A conversao de eixo e feita na Unity, na hora de aplicar a pose -- aqui os
numeros saem como o MuJoCo os tem. Converter nos dois lugares seria o jeito mais
facil de aplicar a rotacao duas vezes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "unity" / "DrosobotLab" / "Assets" / "Resources" / "Fly"
sys.path.insert(0, str(RAIZ / "sim"))


def escreve_obj(caminho: Path, vertices: np.ndarray, faces: np.ndarray,
                nome: str) -> None:
    """OBJ minimo: so vertices e faces. Sem normal, sem UV -- nao precisamos.

    Os VERTICES saem ja em eixos da Unity (Y pra cima), trocando Y e Z como
    `MujocoFrame.Pos` faz. Sem isso a malha fica em eixos do MuJoCo pendurada
    num transform ja convertido: cada peca aparece girada 90 graus sobre a
    propria origem. Num torax quase isotropico ninguem ve; num tarso alongado
    parece que a perna desmontou.

    A troca de eixo e uma reflexao, entao inverte a orientacao dos triangulos.
    A ordem dos indices e invertida junto pra que as normais continuem
    apontando pra fora -- senao o backface culling mostra o interior da malha e
    o corpo aparece esburacado.
    """
    linhas = [f"# {nome} -- NeuroMechFly, exportado do modelo compilado",
              f"# {len(vertices)} vertices, {len(faces)} faces",
              "# vertices em eixos da Unity (Y pra cima); winding invertido",
              f"o {nome}"]
    linhas += [f"v {v[0]:.6f} {v[2]:.6f} {v[1]:.6f}" for v in vertices]
    # OBJ indexa a partir de 1
    linhas += [f"f {f[0]+1} {f[2]+1} {f[1]+1}" for f in faces]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main():
    import mujoco as mj
    from physics import cria

    print("montando a mosca no FlyGym 2.x...")
    corpo = cria("flygym2", arena="flat", com_visao=False)
    corpo.reset(seed=0)
    m = corpo.sim.mj_model
    d = corpo.sim.mj_data

    SAIDA.mkdir(parents=True, exist_ok=True)
    for antigo in SAIDA.glob("*.obj"):
        antigo.unlink()

    # ---- malhas ----
    malhas = {}
    for i in range(m.nmesh):
        nome = mj.mj_id2name(m, mj.mjtObj.mjOBJ_MESH, i) or f"mesh{i}"
        nome_arq = nome.replace("/", "_")
        vi, vn = m.mesh_vertadr[i], m.mesh_vertnum[i]
        fi, fn = m.mesh_faceadr[i], m.mesh_facenum[i]
        verts = m.mesh_vert[vi:vi + vn].reshape(-1, 3)
        faces = m.mesh_face[fi:fi + fn].reshape(-1, 3)
        escreve_obj(SAIDA / f"{nome_arq}.obj", verts, faces, nome_arq)
        malhas[nome] = {"arquivo": f"{nome_arq}.obj",
                        "vertices": int(vn), "faces": int(fn)}
    print(f"  {len(malhas)} malhas -> OBJ")

    # ---- geoms: qual malha, em que corpo, com que transform local ----
    geoms = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        nome_malha = mj.mj_id2name(m, mj.mjtObj.mjOBJ_MESH, mid) or f"mesh{mid}"
        bid = int(m.geom_bodyid[g])
        geoms.append({
            "geom": mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or f"geom{g}",
            "body": mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bid) or f"body{bid}",
            "body_index": bid,
            "mesh": nome_malha,
            # transform LOCAL do geom dentro do corpo: sem isto as pecas
            # montam na origem do corpo, empilhadas
            "pos": [float(x) for x in m.geom_pos[g]],
            "quat_wxyz": [float(x) for x in m.geom_quat[g]],
            "rgba": [float(x) for x in m.geom_rgba[g]],
            "group": int(m.geom_group[g]),
        })

    # ---- hierarquia dos corpos ----
    corpos = []
    for b in range(m.nbody):
        corpos.append({
            "index": b,
            "name": mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or f"body{b}",
            "parent": int(m.body_parentid[b]),
            "pos": [float(x) for x in m.body_pos[b]],
            "quat_wxyz": [float(x) for x in m.body_quat[b]],
        })

    # ---- pose de repouso, em MUNDO ----
    #
    # `body_pos`/`body_quat` acima sao a arvore cinematica com as JUNTAS EM
    # ZERO, e isso NAO e a pose de repouso do modelo: as 73 qpos do padrao sao
    # todas nao-nulas (postura de pe). Compor a arvore ignorando as juntas erra
    # ate 1,77 mm numa mosca de 2,7 mm -- as pernas descem coladas na linha
    # media em vez de abrirem, e a mosca aparece desmontada.
    #
    # Entao exportamos o que o MuJoCo de fato tem depois de mj_forward: xpos e
    # xquat, em MUNDO. E a MESMA grandeza que a telemetria manda por quadro, o
    # que faz a pose de repouso e a pose viva percorrerem o mesmo caminho de
    # codigo na Unity.
    mj.mj_forward(m, d)
    repouso = {
        "segments": [c["name"] for c in corpos[1:]],
        "pos": [[float(x) for x in d.xpos[c["index"]]] for c in corpos[1:]],
        "quat_wxyz": [[float(x) for x in d.xquat[c["index"]]] for c in corpos[1:]],
        "nota": "xpos/xquat em MUNDO apos mj_forward na qpos padrao do modelo",
    }

    meta = {
        "fonte": "NeuroMechFly / FlyGym 2.1.0, do modelo compilado",
        "licenca": "Apache-2.0 (NeLy-EPFL) -- ver THIRD_PARTY_NOTICES.md",
        "unidades": "milimetros",
        "convencoes": {
            "quaternion": "MuJoCo (w, x, y, z) -- Unity usa (x, y, z, w)",
            "eixos": "MuJoCo Z-up destro -- Unity Y-up canhoto; "
                     "a conversao e feita na Unity, nao aqui",
        },
        "n_bodies": int(m.nbody),
        "n_mesh_geoms": len(geoms),
        "meshes": malhas,
        "geoms": geoms,
        "bodies": corpos,
        "pose_repouso": repouso,
    }
    (SAIDA / "fly_body.json").write_text(json.dumps(meta, indent=1),
                                         encoding="utf-8")
    total = sum(f.stat().st_size for f in SAIDA.glob("*.obj"))
    print(f"  {len(geoms)} geoms de malha em {m.nbody} corpos")
    print(f"  {total/1048576:.1f} MiB de OBJ + metadados")
    print(f"  salvo em {SAIDA.relative_to(RAIZ)}")
    corpo.fecha()


if __name__ == "__main__":
    main()
