from __future__ import annotations

from typing import List, Set, Tuple, cast

import networkx as nx

from nsyn.util.base_model import BaseModel
from nsyn.util.errors import CyclicGraphException


class MEC(BaseModel):
    graph: nx.MultiDiGraph

    @classmethod
    def create_empty(cls) -> MEC:
        """
        Creates an empty MEC.
        """
        return cls(graph=nx.MultiDiGraph())

    def add_edge(self, u: int, v: int, directed: bool = False) -> None:
        """
        Adds an edge to the graph. The edge can be directed or undirected.
        Checks for cycles and raises an exception if a cycle is created.

        :param u: The starting node of the edge
        :param v: The ending node of the edge
        :param directed: Boolean indicating if the edge is directed
        """
        # Add the nodes if they don't exist
        if u not in self.graph.nodes:
            self.graph.add_node(u)
        if v not in self.graph.nodes:
            self.graph.add_node(v)
        # Add the edge (temporarily)
        if directed:
            self.graph.add_edge(u, v, directed=True)
        else:
            self.graph.add_edge(u, v, directed=False)
            self.graph.add_edge(v, u, directed=False)

        # Check for cycles
        if self.is_acyclic():
            return
        else:
            # Remove the edge and raise an exception
            self.graph.remove_edge(u, v)
            if not directed:
                self.graph.remove_edge(v, u)
            raise CyclicGraphException("Adding the edge creates a cycle in the graph.")

    def is_acyclic(self) -> bool:
        """
        Checks if the graph is acyclic. Only considers directed edges.
        """
        directed_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d["directed"]
        ]
        directed_graph = nx.DiGraph(directed_edges)
        return nx.is_directed_acyclic_graph(directed_graph)

    def remove_edge(self, u: int, v: int) -> None:
        """
        Removes an edge from the graph.

        :param u: The starting node of the edge
        :param v: The ending node of the edge
        """
        # Removing all edges between u and v, both directed and undirected
        self.graph.remove_edges_from([(u, v), (v, u)])

    def get_edges(self) -> List[Tuple[int, int, bool]]:
        """
        Returns a list of edges in the graph with their types (directed or undirected).
        """
        return [(u, v, d["directed"]) for u, v, d in self.graph.edges(data=True)]

    def draw_graph(self) -> None:
        """
        Draws the graph using networkx and matplotlib.
        """
        import matplotlib.pyplot as plt

        pos = nx.spring_layout(self.graph)  # positions for all nodes

        # Draw nodes and labels
        nx.draw_networkx_nodes(self.graph, pos)
        nx.draw_networkx_labels(self.graph, pos)

        # Separate directed and undirected edges
        directed_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d["directed"]
        ]
        undirected_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if not d["directed"]
        ]

        # Draw directed and undirected edges
        nx.draw_networkx_edges(self.graph, pos, edgelist=directed_edges, arrows=True)
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=undirected_edges, arrows=False, style="dashed"
        )

        plt.show()

    def get_node_num(self) -> int:
        """
        Returns the number of nodes in the graph.
        """
        return self.graph.number_of_nodes()

    def get_undirected_edges(self) -> List[Tuple[int, int]]:
        """
        Returns a list of all undirected edges in the graph, without duplicates.
        """
        undirected_edges_set: Set[Tuple[int, int]] = set()
        for u, v, d in self.graph.edges(data=True):
            if not d["directed"]:
                # Add edge as a sorted tuple to avoid duplicates
                edge = tuple(sorted([cast(int, u), cast(int, v)]))
                undirected_edges_set.add(edge)  # type: ignore [arg-type]

        return list(undirected_edges_set)
