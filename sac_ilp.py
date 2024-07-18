"""
To use the HiGHS solver, you need to install it first and pass the path to the solver.
Follow instructions here: https://ergo-code.github.io/HiGHS/dev/interfaces/cpp/

Command to run:
    python sac_ilp.py --in_file=GPT_modules_info.json --memory_budget=7.5
    python sac_ilp.py --in_file=GPT_modules_info.json --memory_budget=7.5 \
        --solver=HiGHS --solver_path=/home/xuanzh/local/HiGHS/build/bin/highs
"""

import argparse
import json
import logging
from typing import Dict, List

import numpy as np
from aggregate_stats import ModStats
from pulp import (
    COIN_CMD,
    HiGHS_CMD,
    lpDot,
    LpInteger,
    LpMinimize,
    LpProblem,
    lpSum,
    LpVariable,
    PULP_CBC_CMD,
    value,
)

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.INFO)

# Create a stream handler to print log messages to the terminal
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add the handler to the logger
logger.addHandler(handler)


class Node(ModStats):
    index: int = 0  # index according to forward pre-order
    pos_fw_post_order: int = 0  # index according to forward post-order


class Graph:
    def __init__(self, name: str, n: int) -> None:
        self.name: str = name
        self.nodes: List[Node] = []
        self.name2node: Dict[str, Node] = {}
        self.ad_matrix = np.zeros((n, n))
        self.fw_post_order: List[str] = []

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        self.name2node[node["fqn"]] = node


def parse_input(filename: str) -> Graph:
    with open(filename, "r") as f:
        module_info = json.load(f)

    # assertion and number of nodes
    assert len(module_info["modstats"]) == len(module_info["fw_pre_order"])
    n_nodes = len(module_info["modstats"])

    # create graph
    model_name = filename.split("_")[0]
    g = Graph(model_name, n_nodes)
    g.fw_post_order = module_info["fw_post_order"]

    # sort the modules by pre-order and add them to the graph
    module_info["modstats"] = sorted(
        module_info["modstats"],
        key=lambda x: module_info["fw_pre_order"].index(x["fqn"]),
    )
    for i, mod_info in enumerate(module_info["modstats"]):
        node: Node = mod_info
        node["index"] = i
        node["pos_fw_post_order"] = g.fw_post_order.index(node["fqn"])
        g.add_node(node)

    # set up ancestor-descendant matrix
    def is_self_or_submodule(name_descendant: str, name_ancestor: str) -> bool:
        # if name_descendant is a submodule of name_ancestor, or if they are the same
        return (
            name_descendant == name_ancestor or name_ancestor + "." in name_descendant
        )

    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if is_self_or_submodule(g.nodes[j]["fqn"], g.nodes[i]["fqn"]):
                g.ad_matrix[i][j] = 1
            else:
                break

    return g


def sac_milp(
    g: Graph, memory_budget: int, solver: COIN_CMD, verbose: bool = False
) -> None:
    """
    MILP to decide which modules to AC and how much memory to discard.
    Objective: minimize recomputation time.
    Constratint: memory budget (in bytes).
    #TODO: link doc with formulation
    #TODO: change unit for memory from bytes to MiB or GiB
    """
    num_nodes = len(g.nodes)
    M = 10**2  # note: numerical issue may occur if M is too big

    # Create a MILP problem
    prob = LpProblem("SAC", LpMinimize)

    # Create decision variables
    y = LpVariable.matrix("y", list(range(num_nodes)), 0, 1, LpInteger)
    r = LpVariable.matrix("r", list(range(num_nodes)), 0, 1)
    d = LpVariable.matrix("d", list(range(num_nodes)), 0)
    a = LpVariable.matrix("a", list(range(num_nodes)), 0)
    m = LpVariable.matrix("m", list(range(num_nodes)), 0)
    rcp = LpVariable.matrix("rcp", list(range(num_nodes)), 0)
    rct = LpVariable.matrix("rct", list(range(num_nodes)), 0)
    peak_mem = LpVariable("peak_mem", 0)

    # Add constraints
    # [Constraint] No nested AC units
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if g.ad_matrix[i][j] == 1:
                prob += y[i] + y[j] <= 1

    # [Constraint] Do not AC leaf modules
    for i in range(num_nodes):
        if g.nodes[i]["is_leaf"]:
            prob += y[i] == 0

    # [Constraint] Express amount of discarded activation memory
    for i in range(num_nodes):
        ACM_i = g.nodes[i]["ac_memory"]
        IA_i = g.nodes[i]["act_fw_per_module"]
        prob += d[i] == ACM_i * r[i] - (ACM_i - IA_i) * y[i]

    # [Constraint] Express total activation memory in the backward pass
    for i in range(num_nodes):
        AG_i = g.nodes[i]["act_grad_per_module"]
        TA_i = g.nodes[i]["act_total"]
        ACM_i = g.nodes[i]["ac_memory"]
        IA_i = g.nodes[i]["act_fw_per_module"]
        # related to discarded amount of memory
        pos = g.nodes[i]["pos_fw_post_order"]
        coeff = np.zeros(num_nodes)
        for p in range(pos):
            j = g.name2node[g.fw_post_order[p]]["index"]
            coeff[j] = 1
        if g.nodes[i]["is_leaf"]:
            continue
        prob += a[i] + lpDot(coeff, d) == TA_i + AG_i

    # [Constraint] Express the total amount memory at each module
    P_1 = g.nodes[0]["param_per_module"]
    for i in range(num_nodes):
        TG_i = g.nodes[i]["grad_total"]
        prob += m[i] - a[i] == P_1 + TG_i

    # [Constraint] Express peak memory
    for i in range(num_nodes):
        prob += peak_mem >= m[i]

    # [Constraint] Ensure correctness of r_i
    for i in range(num_nodes):
        prob += y[i] >= r[i]
        if g.nodes[i]["is_leaf"]:
            continue
        ACM_i = g.nodes[i]["ac_memory"]
        IA_i = g.nodes[i]["act_fw_per_module"]
        prob += r[i] >= (ACM_i - IA_i) / ACM_i * y[i]

    # [Constraint] Express percentage of recomputation time
    for i in range(num_nodes):
        for s in range(g.nodes[i]["n_segments"]):
            slope = g.nodes[i]["slopes"][s]
            intercept = g.nodes[i]["intercepts"][s]
            prob += rcp[i] - slope * r[i] >= intercept

    # [Constraint] Express recomputation time rec_i = y_i * (rep_i * FCP_i)
    for i in range(num_nodes):
        ACTT_i = g.nodes[i]["ac_runtime"]
        prob += rct[i] <= M * y[i]
        prob += rct[i] <= ACTT_i * rcp[i]
        prob += rct[i] >= ACTT_i * rcp[i] - M * (1 - y[i])

    # [Constraint] Peak memory should be below budget
    prob += peak_mem <= memory_budget

    # Set Objeictive
    prob += lpSum(rct)

    # Solve
    prob.solve(solver)

    # Print solution
    ac_decisions = {}
    for i in range(num_nodes):
        if round(y[i].varValue) == 1:
            ac_decisions[g.nodes[i]["fqn"]] = round(r[i].varValue, 4)
    logger.info(f"AC decisions are {json.dumps(ac_decisions, indent=2)}")
    logger.info(f"recomputation time is {round(value(prob.objective), 2)} ms")
    logger.info(f"peak memory is {display_bytes(peak_mem.varValue, 'GiB')}")

    if verbose:
        logger.info("\n\n --------- DETAILS ---------")
        for i in range(num_nodes):
            if g.nodes[i]["is_leaf"]:
                continue
            y_i = y[i].varValue
            r_i = r[i].varValue
            d_i = d[i].varValue
            a_i = a[i].varValue
            m_i = m[i].varValue
            rcp_i = rcp[i].varValue if rcp[i].varValue else 0
            rct_i = rct[i].varValue
            logger.info(
                ("AC" if round(y_i) == 1 else "  ")
                + f" {g.nodes[i]['fqn']:<40}: "
                + f"r_i = {r_i:.4f} "
                + f"a_i = {display_bytes(a_i, 'GiB'):<10} "
                + f"d_i = {display_bytes(d_i, 'GiB'):<10} "
                + f"m_i = {display_bytes(m_i, 'GiB'):<10} "
                + f"rcp_i = {rcp_i:8.4f} "
                + f"rct_i = {rct_i:8.4f} "
            )


def display_bytes(b: int, unit: str = "B") -> str:
    """
    return a string that represent the number of bytes in a desired unit
    """
    if unit == "KiB":
        return f"{b/2**10:.2f} KiB"
    if unit == "MiB":
        return f"{b/2**20:.2f} MiB"
    if unit == "GiB":
        return f"{b/2**30:.2f} GiB"
    return f"{b:.2f} bytes"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--in_file",
        help="Input file with module information",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--solver",
        help="Solver for MILP",
        required=False,
        choices=["CBC", "HiGHS"],
        default="CBC",
        type=str,
    )

    parser.add_argument(
        "--solver_path",
        help="Path to solver binary",
        required=False,
        type=str,
        default="",
    )

    parser.add_argument(
        "--memory_budget",
        help="Memory budget in GiB",
        required=False,
        type=float,
        default=70,
    )

    parser.add_argument(
        "--verbose",
        help="Verbosity level",
        action="store_true",
    )

    args = parser.parse_args()
    return args


def main():
    # parse the input
    args = parse_args()

    # get the json file by running `python aggregate_stats.py`
    g = parse_input(args.in_file)

    # setup and solve the problem
    solver = PULP_CBC_CMD(msg=args.verbose)
    if args.solver == "HiGHS":
        try:
            if args.solver_path:
                solver = HiGHS_CMD(path=args.solver_path, msg=args.verbose)
            else:
                solver = HiGHS_CMD(msg=args.verbose)
        except Exception:
            logger.error("HiGHS solver not found. Using CBC instead.")
    sac_milp(
        g,
        memory_budget=args.memory_budget * 2**30,
        solver=solver,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
