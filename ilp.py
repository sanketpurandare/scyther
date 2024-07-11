import json
import logging
import time
from typing import Dict, List

import numpy as np
from aggregate_stats import ModStats

from scipy.optimize import Bounds, LinearConstraint, milp

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


def fsdp_milp(g: Graph, selective_ac: bool = False, verbose: bool = False) -> None:
    """
    MILP to decide FSDP & AC units with the goal of minimizing peak memory consumption.
    #TODO: link doc with formulation
    """

    # Parameters
    world_size = 8
    K = 25 * 2**20  # number of bytes in 25 MiB  #TODO not hard code, maybe P_1/W?
    M = 100 * 2**30  # number of bytes in 100 GiB
    SCIPY_TIME_LIMIT_SEC = 30000
    r = 0.6  # memory_budget #TODO per-module budget

    # Decision Variables
    num_nodes = len(g.nodes)
    num_vars = num_nodes * 6 + 2

    def _x_var(i: int) -> int:
        return i

    def _p_var(i: int) -> int:
        return num_nodes + i

    def _g_var(i: int) -> int:
        return num_nodes * 2 + i

    def _m_var(i: int) -> int:
        return num_nodes * 3 + i

    def _a_var(i: int) -> int:
        return num_nodes * 4 + i

    def _y_var(i: int) -> int:
        return num_nodes * 5 + i

    def _peak_mem_var() -> int:
        return -2

    def _max_w_var() -> int:
        return -1

    # Define constraints for the optimization.
    constraints = []

    # [Constraint] specify the amount of parameter taken care of by each module for FSDP
    for i in range(num_nodes):
        P_i = g.nodes[i]["param_per_module"] + g.nodes[i]["grad_per_module"]
        # if not FSDP unit, then taking care of zero parameters
        A1 = np.zeros(num_vars)
        A1[i] = M
        A1[_p_var(i)] = -1
        constraints.append(LinearConstraint(A=A1, lb=0))
        # if FSDP unit, total taken care of by subtree is >= total parameter
        A2 = np.zeros(num_vars)
        A2[i] = -P_i
        for j in range(num_nodes):
            if g.ad_matrix[i][j] == 1:
                A2[_p_var(j)] = 1
        constraints.append(LinearConstraint(A=A2, lb=0))
        # total taken care of by subtree is <= total parameter
        A3 = np.zeros(num_vars)
        for j in range(num_nodes):
            if g.ad_matrix[i][j] == 1:
                A3[_p_var(j)] = 1
        constraints.append(LinearConstraint(A=A3, ub=P_i))

    # [Constraint] specify the amount of gradients taken care of by each module for FSDP
    for i in range(num_nodes):
        G_i = g.nodes[i]["grad_per_module"]
        # if not FSDP unit, then taking care of zero gradients
        A1 = np.zeros(num_vars)
        A1[i] = M
        A1[_g_var(i)] = -1
        constraints.append(LinearConstraint(A=A1, lb=0))
        # if FSDP unit, total taken care of by subtree is >= total gradients
        A2 = np.zeros(num_vars)
        A2[i] = -G_i
        for j in range(num_nodes):
            if g.ad_matrix[i][j] == 1:
                A2[_g_var(j)] = 1
        constraints.append(LinearConstraint(A=A2, lb=0))
        # total taken care of by subtree is <= total gradients
        A3 = np.zeros(num_vars)
        for j in range(num_nodes):
            if g.ad_matrix[i][j] == 1:
                A3[_g_var(j)] = 1
        constraints.append(LinearConstraint(A=A3, ub=G_i))

    # [Constraint] specify the amount memory (activation + params & grads) at each module
    P_1 = g.nodes[0]["param_per_module"]
    for i in range(num_nodes):
        TG_i = g.nodes[i]["grad_total"]
        RHS = (P_1 + TG_i) / world_size
        A = np.zeros(num_vars)
        A[_m_var(i)] = 1
        A[_a_var(i)] = -1
        for j in range(num_nodes):
            if g.ad_matrix[j][i] == 1:
                A[_p_var(j)] = -1
                A[_g_var(j)] = -1
        constraints.append(LinearConstraint(A=A, lb=RHS, ub=RHS))

    # [Constraint] peak memory
    for i in range(num_nodes):
        A = np.zeros(num_vars)
        A[_peak_mem_var()] = 1
        A[_m_var(i)] = -1
        constraints.append(LinearConstraint(A=A, lb=0))

    # [Constraint] maximum sharded params+grads among all FSDP units
    for i in range(num_nodes):
        A = np.zeros(num_vars)
        A[_max_w_var()] = 1
        A[_p_var(i)] = -1
        A[_g_var(i)] = -1
        constraints.append(LinearConstraint(A=A, lb=0))

    # [Constraint] ensure composibility of AC and FSDP units
    # AC units need to be within FSDP boundary
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if g.ad_matrix[i][j] == 1:
                A = np.zeros(num_vars)
                A[_y_var(i)] = 1
                A[j] = 1
                constraints.append(LinearConstraint(A=A, ub=1))

    # [Constraint] No nested AC units
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if g.ad_matrix[i][j] == 1:
                A = np.zeros(num_vars)
                A[_y_var(i)] = 1
                A[_y_var(j)] = 1
                constraints.append(LinearConstraint(A=A, ub=1))

    # [Constraint] No AC if SAC is set to be false:
    if not selective_ac:
        for i in range(num_nodes):
            A = np.zeros(num_vars)
            A[_y_var(i)] = 1
            constraints.append(LinearConstraint(A=A, lb=0, ub=0))

    # [Constraint] No nested FSDP units
    for i in range(1, num_nodes):
        for j in range(i + 1, num_nodes):
            if g.ad_matrix[i][j] == 1:
                A = np.zeros(num_vars)
                A[_x_var(i)] = 1
                A[_x_var(j)] = 1
                constraints.append(LinearConstraint(A=A, ub=1))

    # [Constraint] total activation (grad) memory in the backward pass
    for i in range(num_nodes):
        AG_i = g.nodes[i]["act_grad_per_module"]
        TA_i = g.nodes[i]["act_total"]
        A = np.zeros(num_vars)
        A[_a_var(i)] = 1
        pos = g.nodes[i]["pos_fw_post_order"]
        for p in range(pos):
            j = g.name2node[g.fw_post_order[p]]["index"]
            IA_j = g.nodes[j]["act_fw_per_module"]
            A[_y_var(j)] = IA_j * r
        RHS = TA_i + AG_i
        constraints.append(LinearConstraint(A=A, lb=RHS, ub=RHS))

    # Bounds
    lb = np.concatenate([np.ones(1), np.zeros(num_vars - 1)])
    ub = np.concatenate(
        [
            np.ones(num_nodes),  # x_i
            np.full((num_nodes * 4,), np.inf),  # p_i, g_i, m_i, and a_i
            np.ones(num_nodes),  # y_i
            np.full((2,), np.inf),  # peak_mem and max_w
        ]
    )
    bounds = Bounds(lb, ub)

    # Integrality
    integrality = np.concatenate(
        [
            np.ones(num_nodes),  # x_i
            np.zeros(num_nodes * 4),  # p_i, g_i, m_i, and a_i
            np.ones(num_nodes),  # y_i
            np.zeros(2),  # peak_mem and max_w
        ]
    )

    # Objective
    c = np.concatenate(
        [
            np.full((num_nodes,), K),
            np.zeros(num_nodes * 4),  # p_i, g_i, m_i, and a_i
            np.ones(num_nodes),  # y_i
            np.ones(2),  # peak_mem, max_w
        ]
    )

    # Optimization
    options = {"time_limit": SCIPY_TIME_LIMIT_SEC}
    start_time = time.time()
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options=options,
    )
    end_time = time.time()
    logger.info("Completed MILP in {:.2f} sec".format(end_time - start_time))

    # Display the status of the MILP
    if result.status == 0:
        logger.error("  Succeeded!")
    elif result.status == 1:
        logger.error("  Iteration or time limit reached.")
    elif result.status == 2:
        logger.error("  Problem is infeasible.")
    elif result.status == 3:
        logger.error("  Problem is unbounded.")
    elif result.status == 4:
        logger.error(f"  Other Msg: {result.message}")

    # Display the objective and decisions of the MILP
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

    logger.info(f"Peak memory is: {display_bytes(result.x[-2], 'GiB')}")
    logger.info(
        f"FSDP units are: {[g.nodes[i]['fqn'] for i in range(num_nodes) if result.x[i] == 1]}"
    )
    logger.info(
        f"AC units are {[g.nodes[i]['fqn'] for i in range(num_nodes) if result.x[_y_var(i)] == 1]}"
    )

    # Display more details if needed
    if not verbose:
        return
    logger.info("\n\n --------- DETAILS ---------")
    for i in range(num_nodes):
        x_i = round(result.x[i])
        y_i = round(result.x[_y_var(i)])
        a_i = result.x[_a_var(i)]
        m_i = result.x[_m_var(i)]
        p_i = abs(result.x[_p_var(i)])
        g_i = abs(result.x[_g_var(i)])
        logger.info(
            ("FSDP" if x_i == 1 else "    ")
            + " "
            + ("AC" if y_i == 1 else "  ")
            + f" {g.nodes[i]['fqn']:<40}: "
            + f"a_i = {display_bytes(a_i, 'GiB'):<10} "
            + f"m_i = {display_bytes(m_i, 'GiB'):<10} "
            + f"p_i = {display_bytes(p_i, 'GiB'):<10} "
            + f"g_i = {display_bytes(g_i, 'GiB'):<10} "
        )


if __name__ == "__main__":
    # get the json file by running `python aggregate_stats.py`
    g = parse_input("GPT_modules_info.json")
    result = fsdp_milp(g, selective_ac=True, verbose=True)
