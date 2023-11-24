import networkx as nx


def num_hypha(exp, args):
    return ("num_hypha", len(exp.hyphaes))


def prop_lost_tracks_junction(exp, args):
    lost = 0
    tracked = 0
    lapse = args[0]
    # for node in exp.nodes:
    #     t0 = node.ts()[0]
    #     if node.degree(t0) >=3 and t0 + lapse < exp.ts:
    #         if node.is_in(t0+lapse):
    #             tracked+=len(node.ts())
    #         else:
    #             lost += len(node.ts())
    for t in range(exp.ts - lapse):
        for node in exp.nodes:
            if node.is_in(t) and node.degree(t) >= 3:
                if node.is_in(t + lapse):
                    tracked += 1
                else:
                    lost += 1
    return (f"prop_lost_track_junction_lape{lapse}", lost / (lost + tracked))


def prop_lost_tracks_tips(exp, args):
    lost = 0
    tracked = 0
    lapse = args[0]
    # for node in exp.nodes:
    #     t0 = node.ts()[0]
    #     if node.degree(t0) ==1 and t0 + lapse < exp.ts:
    #         if node.is_in(t0+lapse):
    #             tracked+=len(node.ts())
    #         else:
    #             lost += len(node.ts())
    for t in range(exp.ts - lapse):
        for node in exp.nodes:
            if node.is_in(t) and node.degree(t) == 1:
                if node.is_in(t + lapse):
                    tracked += 1
                else:
                    lost += 1
    return (f"prop_lost_track_tips_lape{lapse}", lost / (lost + tracked))


def prop_inconsistent_root(exp, args):
    return ("inconsist_root", len(exp.inconsistent_root) / len(exp.hyphaes))


def number_of_timepoints_withing_boundaries(exp, args):
    return ("num_timepoint_within", int(exp.reach_out))


def number_of_timepoints(exp, args):
    return ("number_timepoints", int(exp.ts))
