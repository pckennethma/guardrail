from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

import networkx as nx

from nsyn.util.base_model import BaseModel
from nsyn.util.errors import UndirectedEdgeException
from nsyn.util.logger import get_logger
from nsyn.util.mec import MEC, CyclicGraphException

logger = get_logger(name="nsyn.util.dag")


def _mec2juliagr(mec: MEC, instance_path: str) -> None:
    """
    Converts an MEC to a JuliaGraph instance.

    :param mec: An instance of the MEC class representing a partial DAG.
    :param instance_path: The path to the instance file.
    """

    num_node = mec.get_node_num()

    edges = mec.get_edges()
    logger.debug(edges)
    julia_edges: list[tuple[int, int]] = []
    for edge in edges:
        # check if the edge is already inserted
        if (edge[0], edge[1]) in julia_edges:
            continue
        julia_edges.append((edge[0], edge[1]))
        # append another direction for undirected edges
        if not edge[2]:
            julia_edges.append((edge[1], edge[0]))
    julia_edges.sort()

    num_edge = len(julia_edges)

    content = ""

    with open(instance_path, "w") as f:
        content += f"{num_node} {num_edge}\n\n"
        for j_edge in julia_edges:
            content += f"{j_edge[0]+1} {j_edge[1]+1}\n"
        f.write(content)

        logger.debug(f"Writing to {instance_path}:\n{content}")


def _juliagr2nxgraph(instance_path: str) -> nx.MultiDiGraph:
    """
    Converts a JuliaGraph instance to a DAG.
    """

    with open(instance_path) as f:
        lines = f.readlines()
        num_node_str, num_edge_str = lines[0].split()
        num_node = int(num_node_str)
        int(num_edge_str)

        edges = []
        for line in lines[2:]:
            u, v = line.split()
            edges.append((int(u) - 1, int(v) - 1))

        dag = nx.MultiDiGraph()
        dag.add_nodes_from(range(num_node))
        for edge in edges:
            # check if the edge is already in the graph
            if dag.has_edge(edge[0], edge[1]):
                continue
            # check if the edge is directed
            if (edge[1], edge[0]) in edges:
                dag.add_edge(edge[0], edge[1], directed=False)
            else:
                dag.add_edge(edge[0], edge[1], directed=True)

        return dag


def _run_julia(
    cpdag_instance_path: str,
    output_folder: str,
) -> None:
    """
    Runs the Julia code to enumerate all Markov equivalent DAGs.

    julia lib/fastmecenumeration/run_pdag_enum.jl [CPDAG instance path] [output folder]

    """
    import subprocess

    # Run the Julia code
    subprocess.run(
        [
            "julia",
            "lib/fastmecenumeration/run_pdag_enum.jl",
            cpdag_instance_path,
            output_folder,
        ],
        check=True,
    )


class DAG(BaseModel):
    graph: nx.MultiDiGraph

    @classmethod
    def from_mec(cls, mec: MEC, oriented_edges: List[Tuple[int, int]]) -> DAG:
        """
        Initializes a DAG from an MEC and a list of oriented edges.

        :param mec: An instance of the MEC class representing a partial DAG.
        :param oriented_edges: A list of tuples representing the oriented edges.
        """
        # Copy the graph from the MEC
        graph: nx.MultiDiGraph = mec.graph.copy()

        # Orient the specified edges
        for edge in oriented_edges:
            u, v = edge
            if graph.has_edge(u, v) and not graph.get_edge_data(u, v, key="directed"):
                # Orient the edge
                graph.add_edge(u, v, directed=True)
                graph.remove_edge(v, u)
                if not cls.is_acyclic(graph=graph):
                    # If the graph becomes cyclic, revert the change and raise an exception
                    graph.remove_edge(u, v)
                    graph.add_edge(v, u, directed=False)
                    raise CyclicGraphException(
                        "Orienting the edge ({}, {}) creates a cycle.".format(u, v)
                    )

        if any(not d["directed"] for u, v, d in graph.edges(data=True)):
            raise UndirectedEdgeException("Not all edges are directed.")

        return cls(graph=graph)

    @classmethod
    def from_juliagr(cls, dag_instance_path: str) -> DAG:
        """
        Initializes a DAG from a JuliaGraph instance.

        :param dag_instance_path: The path to the JuliaGraph instance file.
        """
        return cls(graph=_juliagr2nxgraph(instance_path=dag_instance_path))

    @staticmethod
    def is_acyclic(graph: nx.MultiDiGraph) -> bool:
        """
        Checks if the graph is acyclic. Only considers directed edges.
        """
        directed_edges = []
        for u, v, d in graph.edges(data=True):
            if not d["directed"]:
                raise UndirectedEdgeException("Not all edges are directed.")
            else:
                directed_edges.append((u, v))

        directed_graph = nx.DiGraph(directed_edges)
        return nx.is_directed_acyclic_graph(directed_graph)

    def get_edges(self) -> List[Tuple[int, int]]:
        """
        Returns a list of edges in the graph with their types (directed or undirected).
        """
        return [(u, v) for u, v, d in self.graph.edges(data=True) if d["directed"]]

    def get_parents(self, node: int) -> List[int]:
        """
        Returns a list of parents of the specified node.
        """
        return [
            u for u, v, d in self.graph.edges(data=True) if v == node and d["directed"]
        ]

    def get_node_num(self) -> int:
        """
        Returns the number of nodes in the graph.
        """
        return self.graph.number_of_nodes()

    def __hash__(self) -> int:
        sorted_edges = sorted(self.get_edges())
        return hash(tuple(sorted_edges))

    @classmethod
    def enumerate_markov_equivalent_dags(cls, mec: MEC) -> List[DAG]:
        """
        Enumerates all Markov equivalent DAGs using Meek's rules.
        """
        dags = []

        with tempfile.TemporaryDirectory() as temp_dir:
            cpdag_instance_path = os.path.join(temp_dir, "cpdag.gr")
            dag_output_folder = os.path.join(temp_dir, "dag")
            os.makedirs(dag_output_folder, exist_ok=True)
            _mec2juliagr(mec=mec, instance_path=cpdag_instance_path)
            _run_julia(
                cpdag_instance_path=cpdag_instance_path,
                output_folder=dag_output_folder,
            )
            for filename in os.listdir(dag_output_folder):
                dag_instance_path = os.path.join(dag_output_folder, filename)
                if os.path.isfile(dag_instance_path):
                    dag = cls.from_juliagr(dag_instance_path=dag_instance_path)
                    dags.append(dag)
            logger.info(f"Enumerated {len(dags)} DAGs.")
        return dags
