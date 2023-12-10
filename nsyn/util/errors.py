class CyclicGraphException(Exception):
    """Custom exception for cyclic graphs."""

    pass


class UnknownDirectionException(Exception):
    """Custom exception for unknown directions."""

    pass


class UndirectedEdgeException(Exception):
    """Custom exception for undirected edges in a directed graph."""

    pass
