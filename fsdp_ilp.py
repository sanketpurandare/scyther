"""
To use the HiGHS solver, you need to install it first and pass the path to the solver.
Follow instructions here: https://ergo-code.github.io/HiGHS/dev/interfaces/cpp/

Command to run:
    python ilp.py --in_file=GPT_modules_info.json --memory_budget=7.5
"""

import argparse
import logging

import numpy as np

from ilp_utils import display_bytes, Graph, parse_input
from pulp import (
    COIN_CMD,
    HiGHS_CMD,
    lpDot,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpVariable,
    PULP_CBC_CMD,
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


def fsdp_milp(
    graph: Graph,
    world_size: int,
    solver: COIN_CMD,
    selective_ac: bool = False,
    verbose: bool = False,
) -> None:
    """
    MILP to decide FSDP units, AC units and how much memory to discard.
    Objective: minimize recomputation time.
    Constratint: memory budget (in bytes).
    """

    # TODO: link doc with formulation
    # TODO: add sac functionality
    # TODO: change unit for memory from bytes to MiB or GiB

    num_nodes = len(graph.nodes)
    M = 80 * 2**30  # note: numerical issue may occur if M is too big

    # TODO: no need for K and r later
    K = 25 * 2**20  # number of bytes in 25 MiB

    # Create a MILP problem
    prob = LpProblem("FSDP", LpMinimize)

    # Create decision variables
    x = LpVariable.matrix("x", list(range(num_nodes)), 0, 1, LpInteger)
    p = LpVariable.matrix("p", list(range(num_nodes)), 0)
    g = LpVariable.matrix("g", list(range(num_nodes)), 0)
    a = LpVariable.matrix("a", list(range(num_nodes)), 0)
    m = LpVariable.matrix("m", list(range(num_nodes)), 0)
    peak_mem = LpVariable("peak_mem", 0)
    max_w = LpVariable("max_w", 0)

    # Add constraints
    # [Constraint] Root module is always an FSDP unit
    prob += x[0] == 1

    # [Constraint] Express parameter taken care of by each module for FSDP
    for i in range(num_nodes):
        P_i = graph.nodes[i]["param_per_module"]
        prob += p[i] <= M * x[i]
        coeff = np.zeros(num_nodes)
        for j in range(i, num_nodes):
            if graph.ad_matrix[i][j] == 1:
                coeff[j] = 1
        prob += P_i * x[i] <= lpDot(p, coeff)
        prob += P_i >= lpDot(p, coeff)

    # [Constraint] Express gradient taken care of by each module for FSDP
    for i in range(num_nodes):
        G_i = graph.nodes[i]["grad_per_module"]
        prob += g[i] <= M * x[i]
        coeff = np.zeros(num_nodes)
        for j in range(i, num_nodes):
            if graph.ad_matrix[i][j] == 1:
                coeff[j] = 1
        prob += G_i * x[i] <= lpDot(g, coeff)
        prob += G_i >= lpDot(g, coeff)

    # [Constraint] Express the total amount memory at each module
    P_1 = graph.nodes[0]["param_per_module"]
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"]
        coeff = np.zeros(num_nodes)
        for j in range(num_nodes):
            if graph.ad_matrix[j][i] == 1:
                coeff[j] = 1
        prob += (
            m[i] == (P_1 + TG_i) / world_size + lpDot(p, coeff) + lpDot(g, coeff) + a[i]
        )

    # [Constraint] Express total activation memory in the backward pass
    for i in range(num_nodes):
        AG_i = graph.nodes[i]["act_grad_per_module"]
        TA_i = graph.nodes[i]["act_total"]
        prob += a[i] == TA_i + AG_i

    # [Constraint] Express peak memory
    for i in range(num_nodes):
        prob += peak_mem >= m[i]

    # [Constraint] Express maximum FSDP shard
    for i in range(num_nodes):
        prob += max_w >= p[i] + g[i]

    # Set Objeictive
    prob += lpDot(np.ones(num_nodes) * K, x) + peak_mem + max_w

    # Solve
    prob.solve(solver)

    # Print solution
    fsdp_decisions = set()
    for i in range(num_nodes):
        if round(x[i].varValue) == 1:
            fsdp_decisions.add(graph.nodes[i]["fqn"])
    logger.info(f"FSDP decisions are {fsdp_decisions}")
    logger.info(f"peak memory is {display_bytes(peak_mem.varValue, 'GiB')}")

    if verbose:
        logger.info("\n\n --------- DETAILS ---------")
        for i in range(num_nodes):
            if graph.nodes[i]["is_leaf"]:
                continue
            x_i = x[i].varValue
            p_i = p[i].varValue
            g_i = g[i].varValue
            a_i = a[i].varValue
            m_i = m[i].varValue
            logger.info(
                ("FSDP" if round(x_i) == 1 else "    ")
                + f" {graph.nodes[i]['fqn']:<40}: "
                + f"p_i = {display_bytes(p_i, 'GiB'):<10} "
                + f"g_i = {display_bytes(g_i, 'GiB'):<10} "
                + f"a_i = {display_bytes(a_i, 'GiB'):<10} "
                + f"m_i = {display_bytes(m_i, 'GiB'):<10} "
            )


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
        "--world_size",
        help="Number of GPUs",
        required=False,
        type=int,
        default=8,
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
    fsdp_milp(
        g,
        world_size=args.world_size,
        # memory_budget=args.memory_budget * 2**30,
        solver=solver,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
