import json
import logging
import time
from typing import Dict, List

import numpy as np
from aggregate_stats import ModStats

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.optimize._optimize import OptimizeResult

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
    # per module activation size, excluding children's activations
    act_excl: int = 0


class Graph:
    def __init__(self, name: str, n: int) -> None:
        self.name: str = name
        self.nodes: List[Node] = []
        self.name2node: Dict[str, Node] = {}
        self.ad_matrix = np.zeros((n, n))

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

    # sort the modules by pre-order and add them to the graph
    module_info["modstats"] = sorted(
        module_info["modstats"],
        key=lambda x: module_info["fw_pre_order"].index(x["fqn"]),
    )
    for mod_info in module_info["modstats"]:
        node: Node = mod_info
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

    # get per-module intermediate activations exclusive for the module
    # (i.e., removing children’s intermediate activations)
    for i in range(n_nodes - 1, -1, -1):
        g.nodes[i]["act_excl"] = g.nodes[i]["act_fw_per_module"] - sum(
            g.nodes[j]["act_excl"]
            for j in range(i + 1, n_nodes)
            if g.ad_matrix[i][j] == 1
        )

    return g


def fsdp_milp(g: Graph, verbose: bool = False) -> OptimizeResult:
    """
    MILP to decide FSDP & AC units with the goal of minimizing peak memory consumption.

    ## Notations
    * N={v_1,v_2,...,v_n} is the set of nodes (or modules)
    * AD is a matrix representation of Ancestor-Descendant relationships
        - AD_{i,j}=1 if v_i is an ancestor/supmodule (self included) of v_j

    ## Parameters
    * P_i is the per-module unsharded parameter memory of module v_i
    * G_i is the per-module unsharded gradient memory of module v_i
    * T_i = P_i + G_i
    * TG_i is the total unsharded gradient memory post module v_i in the backward pass
    * W is the world size
    * K is the penalty constant (in unit of bytes) for large number of FSDP units
      - e.g., if K=2**30, it is worth to save 1GB with 1 additional FSDP unit
    * M is a large number
    * TA_i is the total activation up to but excluding the current module
    * IA_i is the per-module intermediate activations
    * IAe_i is the per-module intermediate activations exclusive for the module
      - i.e., removing children’s intermediate activations
    * AG_i is the per-module activation gradients
    * R is the amount of intermediate activations to be discarded is a module is an AC unit

    ## Decision Variables
    * x_i is the binary indicator variable for whether module v_i is an FSDP unit
    * w_i is the unsharded parameter+gradient memory communicated at module v_i
    * m_i is the total memory during the backward pass at module v_i.
    * a_i is the activation memory during the backward pass at module v_i
    * y_i is the binary decision variable indicating if SAC policy is applied to module v_i
    * peak_mem is the peak memory
    * max_w is the maximum unsharded parameter+gradient memory of all FSDP units

    ## Objective
    ```
    minimize 	peak_mem + max_w + K sum_i x_i + sum_i y_i
    ```

    ## Constraints
    ```
    # constraint 1 -- to express w_i
    M x_i - w_i                              >= 0      forall i in [n]
    sum_{j: AD[i,j]==1} w_j - T_i * x_i      >= 0      forall i in [n]
    sum_{j: AD[i,j]==1} w_j                  <= T_i    forall i in [n]

    # constraint 2 -- to express m_i
    m_i - sum_{j: AD[j,i]==1} w_j - a_i      =  P_1/W + TG_i/W   forall i in [n]

    # constraint 3 -- to express peak_mem
    peak_mem - m_i                           >= 0      forall i in [n]

    # constraint 4 -- to express max_w
    max_w - w_i                              >= 0      forall i in [n]

    # constraint 5 -- to ensure composibility and compatibility
    y_i + x_j                                <= 1      forall i,j where i<>j and AD[i,j]=1
    y_i + y_j                                <= 1      forall i,j where i<>j and AD[i,j]=1

    # constraint 6 -- to express a_i
    a_i + sum_{j: j<i, AD[j,i]=0} R*IA_j*y_i =  TA_i + AG_i + IAe_i     forall i in [n]
    ```

    ## Bounds and Integrality
    ```
    x_1      =  1
    x_i      in {0, 1}   forall i in [n]
    w_i      >= 0        forall i in [n]
    m_i      >= 0        forall i in [n]
    a_i      >= 0        forall i in [n]
    y_i      in {0,1}    forall i in [n]
    peak_mem >= 0
    max_w    >= 0
    ```
    """

    # Parameters
    world_size = 8
    K = 25 * 2**20  # number of bytes in 20MB  #TODO not hard code, maybe P_1/W?
    M = 100 * 2**30  # number of bytes in 100GB
    SCIPY_TIME_LIMIT_SEC = 30000
    r = 0.8  # memory_budget #TODO per-module budget

    # variables
    # x_1, ..., x_n, w_1, ..., w_n, m_1, ..., m_n, a_1, ..., a_n, y_0, ..., y_n,
    # peak_mem, max_w
    num_nodes = len(g.nodes)
    num_vars = num_nodes * 5 + 2

    def _w_var(i: int) -> int:
        return num_nodes + i

    def _m_var(i: int) -> int:
        return num_nodes * 2 + i

    def _a_var(i: int) -> int:
        return num_nodes * 3 + i

    def _y_var(i: int) -> int:
        return num_nodes * 4 + i

    def _peak_mem_var() -> int:
        return -2

    def _max_w_var() -> int:
        return -1

    # Define constraints for the optimization.
    constraints = []

    # constraints 1:
    # specify the amount of parameter / gradient taken care of by each module for FSDP
    for i in range(num_nodes):
        T_i = g.nodes[i]["param_per_module"] + g.nodes[i]["grad_per_module"]
        # if not FSDP unit, then taking care of zero parameters and gradients
        A1 = np.zeros(num_vars)
        A1[i] = M
        A1[_w_var(i)] = -1
        constraints.append(LinearConstraint(A=A1, lb=0))
        # if FSDP unit, total taken care of by subtree is >= total parameter + gradients
        A2 = np.zeros(num_vars)
        A2[i] = -T_i
        for j in range(num_nodes):
            if g.ad_matrix[i][j] == 1:
                A2[_w_var(j)] = 1
        constraints.append(LinearConstraint(A=A2, lb=0))
        # total taken care of by subtree is <= total parameter + gradients
        A3 = np.zeros(num_vars)
        for j in range(num_nodes):
            if g.ad_matrix[i][j] == 1:
                A3[_w_var(j)] = 1
        constraints.append(LinearConstraint(A=A3, ub=T_i))

    # constraints 2:
    # specify the amount memory (activation + params & grads) at each module
    P_1 = g.nodes[0]["param_per_module"]
    for i in range(num_nodes):
        TG_i = g.nodes[i]["grad_total"]
        RHS = (P_1 + TG_i) / world_size
        A = np.zeros(num_vars)
        A[_m_var(i)] = 1
        A[_a_var(i)] = -1
        for j in range(num_nodes):
            if g.ad_matrix[j][i] == 1:
                A[_w_var(j)] = -1
        constraints.append(LinearConstraint(A=A, lb=RHS, ub=RHS))

    # constraints 3:
    # peak memory
    for i in range(num_nodes):
        A = np.zeros(num_vars)
        A[_peak_mem_var()] = 1
        A[_m_var(i)] = -1
        constraints.append(LinearConstraint(A=A, lb=0))

    # constraints 4:
    # maximum sharded params+grads among all FSDP units
    for i in range(num_nodes):
        A = np.zeros(num_vars)
        A[_max_w_var()] = 1
        A[_w_var(i)] = -1
        constraints.append(LinearConstraint(A=A, lb=0))

    # constraints 5:
    # ensure composibility / compatibility of AC and FSDP units
    for i in range(num_nodes):
        # AC units need to be within FSDP boundary
        for j in range(i + 1, num_nodes):
            if g.ad_matrix[i][j] == 1:
                A1 = np.zeros(num_vars)
                A1[_y_var(i)] = 1
                A1[j] = 1
                constraints.append(LinearConstraint(A=A1, ub=1))
        # No nested AC units
        for j in range(i + 1, num_nodes):
            if g.ad_matrix[i][j] == 1:
                A2 = np.zeros(num_vars)
                A2[_y_var(i)] = 1
                A2[_y_var(j)] = 1
                constraints.append(LinearConstraint(A=A2, ub=1))

    # constraints 6:
    # total activation (grad) memory in the backward pass
    for i in range(num_nodes):
        IAe_i = g.nodes[i]["act_excl"]
        AG_i = g.nodes[i]["act_grad_per_module"]
        TA_i = g.nodes[i]["act_total"]
        A = np.zeros(num_vars)
        A[_a_var(i)] = 1
        for j in range(i):
            if g.ad_matrix[j][i] == 1:
                continue
            IA_j = g.nodes[j]["act_fw_per_module"]
            A[_y_var(j)] = IA_j * r
        RHS = TA_i + AG_i + IAe_i
        constraints.append(LinearConstraint(A=A, lb=RHS, ub=RHS))

    # Bounds
    lb = np.concatenate([np.ones(1), np.zeros(num_vars - 1)])
    ub = np.concatenate(
        [
            np.ones(num_nodes),  # x_i
            np.full((num_nodes * 3,), np.inf),  # w_i, m_i, and a_i
            np.ones(num_nodes),  # y_i
            np.full((2,), np.inf),  # peak_mem and max_w
        ]
    )
    bounds = Bounds(lb, ub)

    # Integrality
    integrality = np.concatenate(
        [
            np.ones(num_nodes),  # x_i
            np.zeros(num_nodes * 3),  # w_i, m_i, and a_i
            np.ones(num_nodes),  # y_i
            np.zeros(2),  # peak_mem and max_w
        ]
    )

    # Objective
    c = np.concatenate(
        [
            np.full((num_nodes,), K),
            np.zeros(num_nodes * 3),  # w_i, m_i, and a_i
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
        w_i = abs(result.x[_w_var(i)])
        logger.info(
            ("FSDP" if x_i == 1 else "    ")
            + " "
            + ("AC" if y_i == 1 else "  ")
            + f" {g.nodes[i]['fqn']:<40}: "
            + f"a_i = {display_bytes(a_i, 'GiB'):<10} "
            + f"m_i = {display_bytes(m_i, 'GiB'):<10} "
            + f"w_i = {display_bytes(w_i, 'GiB'):<10} "
        )


if __name__ == "__main__":
    # get the json file by running `python aggregate_stats.py`
    g = parse_input("GPT_modules_info.json")
    result = fsdp_milp(g, verbose=True)
