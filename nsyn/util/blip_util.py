# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import re
import tempfile

import numpy as np
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.PDAG2DAG import pdag2dag

from nsyn.util.flags import BLIP_JAR_PATH, BLIP_JAVA_XMX, BLIP_MACHINE_CORES
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.util.blip_util")

PARENT_SET_IDENTIFICATION_TIMEOUT = 600
STRUCTURE_OPTIMIZATION_TIMEOUT = 600
RES_FILE_PATTERN = re.compile(r"(\d+):\s*-?\d+\.\d+\s*(?:\(([\d,]+)\))?")


def generate_dat(ndarray: np.ndarray, tmp: tempfile.TemporaryDirectory) -> None:
    """
    The method to generate a .dat file for BLIP.

    """
    dat_path = os.path.join(tmp.name, "data.dat")

    # Generate column names
    columns = [f"X{i+1}" for i in range(ndarray.shape[1])]
    out_str = " ".join(columns) + "\n"

    # Process each column
    processed_data = []
    for col_idx in range(ndarray.shape[1]):
        col = ndarray[:, col_idx]

        # Check if the column is of integer type
        if np.issubdtype(col.dtype, np.integer):
            processed_data.append(col)
        else:
            # Create a dictionary to map values to indices
            unique, indices = np.unique(col, return_inverse=True)
            processed_data.append(indices)

    # Convert list of arrays to a 2D array
    processed_ndarray = np.column_stack(processed_data)

    # Calculate cardinality for each column
    card = [
        len(np.unique(processed_ndarray[:, i]))
        for i in range(processed_ndarray.shape[1])
    ]
    out_str += " ".join(map(str, card)) + "\n"

    # Concatenate row data
    row_data = "\n".join(" ".join(map(str, row)) for row in processed_ndarray)
    out_str += row_data

    with open(dat_path, "w") as f:
        f.write(out_str)


def parent_set_iden(tmp: tempfile.TemporaryDirectory) -> None:
    """
    The method to generate a .jkl file for BLIP, which contains the parent set of each variable in the causal graph.

    Args:
        tmp (tempfile.TemporaryDirectory): The temporary directory to store the .jkl file.

    Returns:
        None
    """
    logger.warning(
        f"Identifying parent set... Using {BLIP_MACHINE_CORES} cores and "
        f"-Xmx{BLIP_JAVA_XMX}. Please make sure you have enough CPU cores and memory."
    )
    dat_path = os.path.join(tmp.name, "data.dat")
    jkl_path = os.path.join(tmp.name, "parent_set.jkl")

    os.system(
        f"java -Xmx{BLIP_JAVA_XMX} -jar {BLIP_JAR_PATH} scorer.is -d {dat_path} -j {jkl_path} -t {PARENT_SET_IDENTIFICATION_TIMEOUT} -b {BLIP_MACHINE_CORES}"
    )

    if not os.path.exists(jkl_path):
        logger.error(f"Failed to generate parent set file {jkl_path}")


def general_struc_opt(tmp: tempfile.TemporaryDirectory) -> None:
    # java -jar blip.jar solver.winasobs.adv -smp ent -d data/child-5000.dat -j data/child-5000.jkl -r data/child.wa.res -t 10 -b 0
    logger.warning(
        f"Identifying parent set... Using {BLIP_MACHINE_CORES} cores and "
        f"-Xmx{BLIP_JAVA_XMX}. Please make sure you have enough CPU cores and memory."
    )
    dat_path = os.path.join(tmp.name, "data.dat")
    jkl_path = os.path.join(tmp.name, "parent_set.jkl")
    res_path = os.path.join(tmp.name, "result.res")

    os.system(
        f"java -Xmx{BLIP_JAVA_XMX} -jar {BLIP_JAR_PATH} solver.winasobs.adv -smp ent  -d {dat_path} -j {jkl_path} -r {res_path} -t {STRUCTURE_OPTIMIZATION_TIMEOUT} -b {BLIP_MACHINE_CORES}"
    )

    if not os.path.exists(res_path):
        logger.error(f"Failed to generate result file {res_path}")


def get_blip(tmp: tempfile.TemporaryDirectory) -> GeneralGraph:
    """
    The method to parse the result file of BLIP.

    Args:
        tmp (tempfile.TemporaryDirectory): The temporary directory to store the .jkl file.

    Returns:
        GeneralGraph: The causal graph.
    """
    res_path = os.path.join(tmp.name, "result.res")
    with open(res_path) as f:
        raw_text = f.read()
        logger.info(f"BLIP output raw text:\n{raw_text}")
        lines = [
            line.strip()
            for line in raw_text.split("\n")
            if not line.startswith("Score") and line.strip() != ""
        ]
    nodes = [GraphNode("X%d" % (i + 1)) for i in range(len(lines))]
    G = GeneralGraph(nodes=nodes)
    for line in lines:
        line_match = RES_FILE_PATTERN.match(line)
        if line_match is None:
            logger.error(f"Invalid line {line}")
            continue
        child_idx = int(line_match.group(1))
        parent_indices = (
            [int(n) for n in line_match.group(2).split(",")]
            if line_match.group(2)
            else []
        )
        child = nodes[child_idx]
        for parent_idx in parent_indices:
            parent = nodes[parent_idx]
            G.add_edge(
                Edge(
                    node1=parent,
                    node2=child,
                    end1=Endpoint.TAIL,
                    end2=Endpoint.ARROW,
                )
            )
    G = pdag2dag(G)
    G = dag2cpdag(G)
    return G


def run_blip(data: np.ndarray) -> GeneralGraph:
    """
    Run BLIP on a given dataframe.

    Args:
        data (np.ndarray): The input data.

    Returns:
        GeneralGraph: The causal graph.
    """
    tmp = tempfile.TemporaryDirectory()
    generate_dat(data, tmp)
    parent_set_iden(tmp)
    general_struc_opt(tmp)
    dag = get_blip(tmp)
    return dag


if __name__ == "__main__":
    # run_blip(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), "../blip/data/child.res")
    # run_blip(
    #     np.array([[1, 2, 3], [1, 2, 3]]),
    # )
    sample_res = """0: -96100.56  (1,7,8,38,44,46)
1: -56266.72  (8,10,23,39)
2: -19959.74  (43,44)
3: -105682.59  (0,7,8,36,41,44)
4: -1175.23  (6,38)
5: -40534.75  (0,3,28,41,43,44)
6: -11735.89  (10,12,46)
7: -111799.65  (39)
8: -102858.35  (7,23,27,39,46)
9: -94779.94  (7,8,10,11,38,41)
10: -19906.35  (16,37)
11: -108316.73  (3,7,8,36,41,44)
12: -4869.10  (11,16,18,27,37,46)
13: -102684.80  (7,8,9,11,14,41)
14: -13356.30  (10,16,37)
15: -51813.74  (16,25)
16: -26567.62 
17: -33177.79  (15,18,23,24,26,46)
18: -21599.33  (14,16,27)
19: -3401.72  (15,17,23,24)
20: -437.50  (16,19,35,46)
21: -1855.21  (15,17,19,23)
22: -242.83  (20,21,27,46)
23: -37041.11  (15,16,18,25)
24: -29588.95  (15,16,23,25,39)
25: -5673.38  (14,16,27,31,46)
26: -78556.80  (11,15,27,38,41,44)
27: -36284.87  (10,14,16,37)
28: -93782.46  (8,9,11,29,38,41)
29: -7277.37  (16,18,27,37,46)
30: -55871.90  (11,31,34,38,41,44)
31: -8132.24  (16,18,27,29,46)
32: -13666.36  (24,26,27,33,39)
33: -6301.37  (14,16,18,27,46)
34: -4350.13  (26,27,32,35)
35: -6844.89  (16,27,29,39,46)
36: -78616.18  (7,8,37,44)
37: -15783.87  (16)
38: -8162.32  (36,37)
39: -64964.44  (15,18,23,27,46)
40: -17033.86  (23,31,39,46)
41: -116738.00  (0,7,8,44)
42: -58691.21  (0,3,7,41,44)
43: -118119.06  (0,3,8,38,41,44)
44: -89178.39  (7,8,46)
45: -316.87  (30,31)
46: -30185.07  (10,14,16,18,27,37)
"""

    tmp = tempfile.TemporaryDirectory()
    with open(os.path.join(tmp.name, "result.res"), "w") as f:
        f.write(sample_res)
    dag = get_blip(tmp)
    logger.info(dag)
    logger.info(dag.nodes)
    logger.info(dag.get_num_edges())
