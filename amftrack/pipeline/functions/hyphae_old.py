def clean_obvious_fake_tips(exp):
    exp_clean = exp  # better to modify in place
    for hyph in exp_clean.hyphaes:
        hyph.update_ts()
    to_remove_hyphae = set()
    for hyphae in exp_clean.hyphaes:
        for t in hyphae.ts:
            try:
                hyphae.get_nodes_within(t)
            except nx.exception.NetworkXNoPath:
                to_remove_hyphae.add(hyphae)
                print(
                    f"clean tips begin :error for hyphae {hyphae} on position {hyphae.end.pos(t),hyphae.root.pos(t)}"
                )
    for hyphae in to_remove_hyphae:
        exp_clean.hyphaes.remove(hyphae)
    print(f"There is {len(exp_clean.hyphaes)} hyphae")
    hyphae_with_degree4 = {}
    hyph_anas_tip_tip = []
    hyph_anas_tip_hyph = [
        hyphat
        for hyphat in exp_clean.hyphaes
        if len(hyphat.ts) >= 2
        and hyphat.end.degree(hyphat.ts[-1]) >= 3
        and hyphat.end.degree(hyphat.ts[-2]) >= 3
    ]
    potential = []
    for hyph in exp_clean.hyphaes:
        if (
            len(hyph.ts) >= 2
            and hyph.end.degree(hyph.ts[-1]) == 1
            and hyph.end.ts()[-1] != len(exp_clean.nx_graph) - 1
            and not np.all([hyph.get_length_pixel(t) <= 20 for t in hyph.ts])
        ):
            potential.append(hyph)
    for hyph in potential:
        t0 = hyph.ts[-1]
        for hyph2 in potential:
            if hyph2.ts[-1] == t0 and hyph != hyph2:
                vector = (hyph2.end.pos(t0) - hyph.end.pos(t0)) / np.linalg.norm(
                    hyph2.end.pos(t0) - hyph.end.pos(t0)
                )
                vertical_vector = np.array([-1, 0])
                dot_product = np.dot(vertical_vector, vector)
                if (
                    vertical_vector[1] * vector[0] - vertical_vector[0] * vector[1] >= 0
                ):  # determinant
                    angle = np.arccos(dot_product) / (2 * np.pi) * 360
                else:
                    angle = -np.arccos(dot_product) / (2 * np.pi) * 360
                score = np.cos(
                    (angle - (180 + hyph.end.edges(t0)[0].orientation_begin(t0, 30)))
                    / 360
                    * 2
                    * np.pi
                ) + np.cos(
                    (360 + angle - hyph2.end.edges(t0)[0].orientation_begin(t0, 30))
                    / 360
                    * 2
                    * np.pi
                )
                if (
                    np.linalg.norm(hyph2.end.pos(t0) - hyph.end.pos(t0)) <= 500
                    and score >= 0.5
                ):
                    hyph_anas_tip_tip.append((hyph, hyph2, t0))
    hyph_tiptip_set = {c[0] for c in hyph_anas_tip_tip}
    disapearing_hyph_len1 = [
        hyph
        for hyph in exp_clean.hyphaes
        if len(hyph.end.ts()) == 1
        and hyph.ts[-1] != len(exp_clean.nx_graph) - 1
        and hyph not in hyph_tiptip_set
    ]
    print(
        f"Found {len(hyph_tiptip_set)} tip-tip anastomosis, found {len(disapearing_hyph_len1)} tips that appear at only one timestep and then disapear and are not anastomosing"
    )
    nx_graph_cleans = [nx.Graph.copy(nx_g) for nx_g in exp.nx_graph]
    exp_clean.nx_graph = nx_graph_cleans
    for hyph in disapearing_hyph_len1:
        exp_clean.nx_graph[hyph.ts[0]].remove_node(hyph.end.label)
        exp_clean.hyphaes.remove(hyph)
    exp_clean.nx_graph = [prune_graph(g,0.01) for g in exp_clean.nx_graph]
    for i, g in enumerate(exp_clean.nx_graph):
        reconnect_degree_2(g, exp_clean.positions[i])
    exp_clean.nodes = []
    labels = {int(node) for g in exp_clean.nx_graph for node in g}
    for label in labels:
        exp_clean.nodes.append(Node(label, exp_clean))
    for hyph in exp_clean.hyphaes:
        hyph.update_ts()
    #     exp_clean_relabeled= clean_exp_with_hyphaes(exp_clean)
    to_remove_hyphae = set()
    for hyphae in exp_clean.hyphaes:
        for t in hyphae.ts:
            try:
                hyphae.get_nodes_within(t)
            except nx.exception.NetworkXNoPath:
                to_remove_hyphae.add(hyphae)
                print(
                    f"clean tips end error for hyphae {hyphae} on position {hyphae.end.pos(t),hyphae.root.pos(t)}"
                )
    for hyphae in to_remove_hyphae:
        exp_clean.hyphaes.remove(hyphae)
    return exp_clean

def solve_degree4(exp):
    hyphae_with_degree4 = {}
    exp_clean = exp  # better to modify in place
    articulation_points = [
        list(nx.articulation_points(nx_g)) for nx_g in exp_clean.nx_graph
    ]
    nx_graph_cleans = [nx.Graph.copy(nx_g) for nx_g in exp.nx_graph]
    exp_clean.nx_graph = nx_graph_cleans
    for hyph in exp.hyphaes:
        t0 = hyph.ts[-1]
        nodes, edges = hyph.get_nodes_within(t0)
        hyphae_with_degree4[hyph] = []
        for node in nodes:
            if exp.get_node(node).degree(t0) >= 4:
                hyphae_with_degree4[hyph].append(exp.get_node(node))
    roots = [hyph.root for hyph in exp.hyphaes]
    ends = [hyph.end for hyph in exp.hyphaes]
    solved_node = []
    solved = []
    iis = {t: 2 for t in range(len(exp_clean.nx_graph))}
    for hyph in hyphae_with_degree4.keys():
        for node in hyphae_with_degree4[hyph]:
            can_be_removed = True
            if 0 in node.ts():
                can_be_removed = False
            if len(node.ts()) <= 1:
                can_be_removed = False
            else:
                for t in node.ts():
                    if node.degree(t) == 4:
                        pairs = []
                        for edge in node.edges(t):
                            mini = np.inf
                            for edge_candidate in node.edges(t):
                                angle = np.cos(
                                    (
                                        edge.orientation_begin(t, 100)
                                        - edge_candidate.orientation_begin(t, 100)
                                    )
                                    / 360
                                    * 2
                                    * np.pi
                                )
                                if angle < mini:
                                    winner = edge_candidate
                                    mini = angle
                            if (edge, winner) not in pairs and (
                                winner,
                                edge,
                            ) not in pairs:
                                pairs.append((edge, winner))
                        for pair in pairs:
                            can_be_removed *= (
                                pair[0].end.degree(t) != 1 or pair[1].end.degree(t) != 1
                            )
                            can_be_removed *= (
                                pair[0].end.label not in articulation_points[t]
                                or pair[1].end.label not in articulation_points[t]
                            )
                        if len(pairs) > 2:
                            can_be_removed *= False
            if (
                node not in roots
                and node not in ends
                and node not in solved_node
                and can_be_removed
            ):
                solved_node.append(node)
                for t in node.ts():
                    if node.degree(t) == 4:
                        solved.append((t, node.neighbours(t)))
                        pairs = []
                        for edge in node.edges(t):
                            mini = np.inf
                            for edge_candidate in node.edges(t):
                                angle = np.cos(
                                    (
                                        edge.orientation_begin(t, 100)
                                        - edge_candidate.orientation_begin(t, 100)
                                    )
                                    / 360
                                    * 2
                                    * np.pi
                                )
                                if angle < mini:
                                    winner = edge_candidate
                                    mini = angle
                            if (edge, winner) not in pairs and (
                                winner,
                                edge,
                            ) not in pairs:
                                pairs.append((edge, winner))
                        for pair in pairs:
                            right_n = pair[0].end
                            left_n = pair[1].end
                            right_edge = pair[0].pixel_list(t)
                            left_edge = list(reversed(pair[1].pixel_list(t)))
                            right_edge_width = pair[0].width(t)
                            left_edge_width = pair[1].width(t)
                            pixel_list = left_edge + right_edge[1:]
                            info = {"weight": len(pixel_list), "pixel_list": pixel_list, "width" : ((len(right_edge)*right_edge_width)+(len(left_edge)*left_edge_width))/(left_edge_width+right_edge_width)}
                            if right_n != left_n:
                                exp_clean.nx_graph[t].add_edges_from(
                                    [(left_n.label, right_n.label, info)]
                                )
                        exp_clean.nx_graph[t].remove_node(node.label)
                        if (
                            len(list(nx.connected_components(exp_clean.nx_graph[t])))
                            >= iis[t]
                        ):
                            iis[t] += 1
                            S = [
                                list(c)
                                for c in nx.connected_components(exp_clean.nx_graph[t])
                            ]
                            len_connected = [len(c) for c in S]
                            print(S[np.argmin(len_connected)])
                            print(
                                t,
                                node,
                                pairs,
                                len(
                                    list(nx.connected_components(exp_clean.nx_graph[t]))
                                ),
                            )
    exp_clean.nx_graph = [prune_graph(g,0.1) for g in exp_clean.nx_graph]
    exp_clean.nodes = []
    labels = {int(node) for g in exp_clean.nx_graph for node in g}
    for label in labels:
        exp_clean.nodes.append(Node(label, exp_clean))
    #     exp_clean_relabeled= clean_exp_with_hyphaes(exp_clean)
    print(len(solved_node))
    return (solved, solved_node)

def resolve_ambiguity_two_ends(hyphaes, bottom_threshold=0.98):
    root_hyph = {}
    hyphae_two_ends = [hyph for hyph in hyphaes if hyph.root.degree(hyph.ts[0]) == 1]
    print(f"{len(hyphae_two_ends)} hyphae with two ends have been detected")
    to_remove = []
    x_boundaries = hyphaes[0].experiment.boundaries_x
    y_boundaries = hyphaes[0].experiment.boundaries_y
    counter_problem = 0
    counter_problem_solved = 0
    for hyph in hyphae_two_ends:
        t0 = hyph.ts[0]
        if not hyph.root.pos(t0)[0] >= bottom_threshold * x_boundaries[1]:
            counter_problem += 1
            nodes, edges = hyph.get_nodes_within(t0)
            mini = np.inf
            found = False
            for i, edge in enumerate(edges):
                if edge.end.degree(t0) == 4:
                    next_edge = edges[i + 1]
                    angle = np.cos(
                        (
                            edge.orientation_end(t0, 50)
                            - next_edge.orientation_begin(t0, 50)
                        )
                        / 360
                        * 2
                        * np.pi
                    )
                    if angle < mini:
                        found = True
                        maxi = angle
                        root_candidate = edge.end
            if found:
                counter_problem_solved += 1
                root_hyph[hyph] = root_candidate
    print(
        f"Among the {len(hyphaes)}, {counter_problem} hyphaes had two real ends, {counter_problem_solved} ambiguity were solved by finding a degree 4 node"
    )
    ends = {hyph.end: hyph for hyph in hyphaes}
    for hyph in root_hyph.keys():
        if hyph.root in ends:
            ends[hyph.root].root = root_hyph[hyph]
        hyph.root = root_hyph[hyph]
    for hyph in hyphaes[0].experiment.hyphaes:
        hyph.update_ts()
    return root_hyph

def clean_and_relabel(exp):
    hyphaes, problems = get_hyphae(exp)
    exp.hyphaes = hyphaes

    # exp_clean = clean_exp_with_hyphaes(exp)
    # #     equ_class, ambig, connection = resolve_ambiguity(exp_clean.hyphaes)
    # #     new_graph, newposs = relabel_nodes_after_amb(
    # #         connection, exp_clean.nx_graph, exp_clean.positions
    # #     )
    # #     exp_clean.nx_graph = new_graph
    # #     exp_clean.positions = newposs
    # #     exp_clean.nodes = []
    # labels = {int(node) for g in exp_clean.nx_graph for node in g}
    # for label in labels:
    #     exp_clean.nodes.append(Node(label, exp_clean))
    # exp_clean_relabeled = clean_exp_with_hyphaes(exp_clean)
    return exp

def resolve_ambiguity(hyphaes):
    #     problems=[]
    #     safe=[]
    #     for hyph in hyphaes:
    #         if len(hyph.root.ts())<len(hyph.ts):
    #             problems.append(hyph)
    #         else:
    #             safe.append(hyph)
    to_remove = []
    for hyph in hyphaes:
        hyph.update_ts()
        if len(hyph.ts) == 0:
            to_remove.append(hyph)
    for hyph in to_remove:
        hyphaes.remove(hyph)
    safe = hyphaes
    ambiguities = []
    connection = {hyph: [] for hyph in safe}
    for hyph in safe:
        root = hyph.root
        for hyph2 in safe:
            if (
                hyph2.root == root
                and hyph2.end != hyph.end
                and (hyph2, hyph) not in ambiguities
            ):
                ambiguities.append((hyph, hyph2))
    #         t0=hyph.ts[0]
    #         nodes = hyph.get_nodes_within(t0)
    #         nodes_within_initial[hyph.end]=nodes
    #     for hyph in safe:
    #         nodes = nodes_within_initial[hyph.end]
    #         root,first = nodes[0],nodes[1]
    #         for hyph2 in safe:
    #             if hyph2.end != hyph.end:
    #                 nodes2 = nodes_within_initial[hyph2.end]
    #                 if root in nodes2 and first in nodes2:
    #                     ambiguities.append(hyph,hyph2)
    for ambig in ambiguities:
        common_ts = sorted(set(ambig[0].ts).intersection(set(ambig[1].ts)))
        if len(common_ts) >= 1:
            continue
        else:
            hyph1 = ambig[0]
            hyph2 = ambig[1]
            if hyph1.ts[-1] <= hyph2.ts[0]:
                t1 = hyph1.ts[-1]
                t2 = hyph2.ts[0]
            else:
                t1 = hyph1.ts[0]
                t2 = hyph2.ts[-1]
            if np.linalg.norm(hyph1.end.pos(t1) - hyph2.end.pos(t2)) <= 300:
                connection[hyph1].append(hyph2)
    equ_classes = []
    put_in_class = set()
    for hyph in connection.keys():
        if not hyph in put_in_class:
            equ = {hyph}
            full_equ_class = False
            i=0
            while not full_equ_class:
                i+=1
                if i>=100:
                    print(i)
                full_equ_class = True
                for hypha in list(equ):
                    for hyph2 in connection[hypha]:
                        if hyph2 not in equ:
                            equ.add(hyph2)
                            full_equ_class = False
            if not np.any([hyphaa in put_in_class for hyphaa in equ]):
                for hyphaa in equ:
                    put_in_class.add(hyphaa)
                equ_classes.append(equ)
    connect = {}
    for hyph in safe:
        found = False
        for equ in equ_classes:
            if hyph in equ:
                found = True
                connect[hyph.end.label] = np.min([hyphaa.end.label for hyphaa in equ])
        if not found:
            connect[hyph.end.label] = hyph.end.label
    return (equ_classes, ambiguities, connect)