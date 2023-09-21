import networkx as nx
import numpy as np
from tqdm.notebook import tqdm  # for Jupyter notebook or IPython


def angle_between(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))


def relation(v, e1, e2, EExtract, E, positions):
    v_i = E[e1][0] if E[e1][1] == v else E[e1][1]
    v_j = E[e2][0] if E[e2][1] == v else E[e2][1]

    vec_i = np.array(positions[v_i]) - np.array(positions[v])
    vec_j = np.array(positions[v_j]) - np.array(positions[v])

    theta = np.pi / 4  # Adjust this value based on your requirements

    # Check the first condition
    if abs(angle_between(vec_i, vec_j) - np.pi) > theta:
        return False

    # Check the second condition
    for e in EExtract:
        if e != e1 and e != e2:
            v_k = E[e][0] if E[e][1] == v else E[e][1]
            vec_k = np.array(positions[v_k]) - np.array(positions[v])

            if abs(angle_between(vec_k, vec_i) - np.pi) < abs(
                angle_between(vec_i, vec_j) - np.pi
            ) or abs(angle_between(vec_k, vec_j) - np.pi) < abs(
                angle_between(vec_i, vec_j) - np.pi
            ):
                return False

    return True


def hypergraph_from_graph(G, positions):
    V = list(G.nodes())
    E = list(G.edges())
    e = len(E)
    v = len(V)

    H = [0] * e
    Cor = [[0] * 10 for _ in range(e)]

    # STEP 1
    for i in tqdm(V, desc="Processing vertices"):
        EExtract = [edge_idx for edge_idx, edge in enumerate(E) if i in edge]
        for j in range(len(EExtract)):
            for k in range(j + 1, len(EExtract)):
                e1 = EExtract[j]
                e2 = EExtract[k]
                if relation(i, e1, e2, EExtract, E, positions):
                    Cor[e1][Cor[e1].index(0)] = e2
                    Cor[e2][Cor[e2].index(0)] = e1

    # STEP 2
    CurrentMark = 1
    for i in tqdm(range(e), desc="Processing stack"):
        if H[i] == 0:
            stack = [i]
            visited = set()  # To keep track of edges that have been added to the stack
            while stack:
                current = stack.pop()
                H[current] = CurrentMark
                # Only add edges to the stack that haven't been assigned to a hyperedge and aren't already on the stack
                related_edges = [
                    cor
                    for cor in Cor[current]
                    if cor != 0 and H[cor] == 0 and cor not in visited
                ]
                stack.extend(related_edges)
                visited.update(related_edges)
            CurrentMark += 1
    H = {edge: H[i] for i, edge in enumerate(E)}
    return H
