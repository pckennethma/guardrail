from uuid import uuid4

import networkx as nx
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from nsyn.util.dag import DAG
from nsyn.util.errors import CyclicGraphException, UnknownDirectionException
from nsyn.util.logger import get_logger
from nsyn.util.mec import MEC

logger = get_logger(name="nsyn.util.convert")

_MAX_ATTEMPTS = 10
_TIMEOUT = 10


def mec2ggraph(
    mec: MEC,
) -> GeneralGraph:
    """
    Converts an MEC to a GeneralGraph.

    :param mec: An instance of the MEC class representing a partial DAG.
    """
    nodes = []
    for i in range(mec.get_node_num()):
        gnode = GraphNode(str(uuid4()))
        gnode.add_attribute("id", i)
        nodes.append(gnode)
    ggraph = GeneralGraph(nodes=nodes)
    for u, v, directed in mec.get_edges():
        edge = Edge(
            node1=nodes[u],
            node2=nodes[v],
            end1=Endpoint.TAIL,
            end2=Endpoint.ARROW if directed else Endpoint.TAIL,
        )
        ggraph.add_edge(edge=edge)
    return ggraph


def ggraph2mec(
    ggraph: GeneralGraph,
) -> MEC:
    """
    Converts a GeneralGraph to an MEC.

    :param ggraph: An instance of the GeneralGraph class representing a partial DAG.
    """
    mec = MEC.create_empty()
    node_num = len(ggraph.nodes)
    for i in range(node_num):
        for j in range(i + 1, node_num):
            """
            graph[j,i]=1 and graph[i,j]=-1 indicate i -> j;
            graph[i,j] = graph[j,i] = -1 indicate i — j;
            graph[i,j] = graph[j,i] = 1 indicates i <-> j.
            """
            if ggraph.graph[j, i] == 1 and ggraph.graph[i, j] == -1:
                mec.add_edge(i, j, directed=True)
            elif ggraph.graph[j, i] == -1 and ggraph.graph[i, j] == -1:
                mec.add_edge(i, j, directed=False)
            elif ggraph.graph[j, i] == -1 and ggraph.graph[i, j] == 1:
                mec.add_edge(j, i, directed=True)
            elif ggraph.graph[j, i] == 0 and ggraph.graph[i, j] == 0:
                continue
            else:
                raise UnknownDirectionException(
                    f"Unknown direction between {i} and {j}: {ggraph.graph[i, j]}-{ggraph.graph[j, i]}."
                )
    return mec


def dag2ggraph(
    dag: DAG,
) -> GeneralGraph:
    """
    Converts a DAG to a GeneralGraph.

    :param dag: An instance of the DAG class representing a partial DAG.
    """
    nodes = []
    for i in range(dag.get_node_num()):
        gnode = GraphNode(str(uuid4()))
        gnode.add_attribute("id", i)
        nodes.append(gnode)
    ggraph = GeneralGraph(nodes=nodes)
    for u, v in dag.get_edges():
        edge = Edge(
            node1=nodes[u],
            node2=nodes[v],
            end1=Endpoint.TAIL,
            end2=Endpoint.ARROW,
        )
        ggraph.add_edge(edge=edge)
    return ggraph


def ggraph2dag(
    ggraph: GeneralGraph,
) -> DAG:
    """
    Converts a GeneralGraph to a DAG.

    :param ggraph: An instance of the GeneralGraph class representing a partial DAG.
    """
    mec = ggraph2mec(ggraph)
    return DAG(graph=mec.graph)


def mec2dag(mec: MEC) -> DAG:
    """
    Converts an MEC to a DAG.

    :param mec: An instance of the MEC class representing a partial DAG.
    """
    dag = nx.MultiDiGraph()
    dag.add_nodes_from(mec.graph.nodes())

    for u, v, d in mec.graph.edges(data=True):
        if d["directed"]:
            dag.add_edge(u, v, directed=True)

    und_edges = mec.get_undirected_edges()
    logger.info(f"Orienting {len(und_edges)} edges: {und_edges}")
    for u, v in und_edges:
        dag.add_edge(u, v, directed=True)
        if nx.is_directed_acyclic_graph(dag):
            continue
        else:
            dag.remove_edge(u, v)
            dag.add_edge(v, u, directed=True)
            if not nx.is_directed_acyclic_graph(dag):
                raise CyclicGraphException(
                    "mec2dag: Orienting the edge ({}, {}) creates a cycle.".format(u, v)
                )

    return DAG(graph=dag)
