from __future__ import annotations

import os
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple, cast

import pandas as pd
from p_tqdm import p_uimap

from nsyn.dsl.prog import DSLProg
from nsyn.dsl.stmt import DSLStmt
from nsyn.learner import BaseLearner
from nsyn.sampler import AbstractSampler
from nsyn.util.base_model import BaseModel
from nsyn.util.dag import DAG
from nsyn.util.logger import get_logger
from nsyn.util.mec import MEC

logger = get_logger(name="nsyn.search")

_NUM_WORKERS = int(os.getenv("NSYN_NUM_WORKERS", "4"))


class Search(BaseModel):
    """
    Search class for synthesizing programs from data using specified learning and sampling algorithms.

    This class encapsulates the logic for searching over a space of potential programs that best describe a given dataset. It leverages a learning algorithm and a sampling algorithm to explore different program configurations and select the most appropriate ones.

    Attributes:
        LearningAlgo (BaseLearner): An instance of a learning algorithm used for program synthesis.
        SamplingAlgo (AbstractSampler): An instance of a sampling algorithm used in the search process.
        input_data (pd.DataFrame): The dataset used for program synthesis.
        columns (List[str]): A list of column names from the input data.
        stmt_cache (Dict[Tuple[FrozenSet, int], Optional[DSLStmt]]): A cache to store DSL statements.
        epsilon (float): A threshold parameter used in the learning algorithm.
        search_strategy (Literal['exact', 'stochastic']): The search strategy to use.

    Methods:
        create: A class method to create an instance of the Search class.
        run: Performs program synthesis using the specified learning and sampling algorithms.
        _synthesis_from_dag: Synthesizes a DSL program from a given DAG.
        _get_stmt: Retrieves or creates a DSL statement for given determinants and a dependent index.
        _stochastic_search_over_mec: Performs stochastic search over a given MEC. (Not implemented)
        _exact_search_over_mec: Performs exact search over a given MEC.
    """

    LearningAlgo: BaseLearner
    SamplingAlgo: AbstractSampler

    input_data: pd.DataFrame
    columns: List[str]

    stmt_cache: DictProxy | Dict[Tuple[FrozenSet, int], Optional[DSLStmt]]

    epsilon: float = 0.01
    min_support: int = 100
    search_strategy: Literal["exact", "stochastic"] = "exact"

    @classmethod
    def create(
        cls,
        learning_algorithm: BaseLearner,
        sampling_algorithm: AbstractSampler,
        input_data: pd.DataFrame,
        epsilon: float = 0.01,
        min_support: int = 10,
        search_strategy: Literal["exact", "stochastic"] = "exact",
    ) -> Search:
        """
        Class method to create an instance of the Search class.

        Args:
            learning_algorithm (BaseLearner): The learning algorithm to use.
            sampling_algorithm (AbstractSampler): The sampling algorithm to use.
            input_data (pd.DataFrame): The input data for the search.
            epsilon (float): A threshold parameter used in the learning algorithm.
            search_strategy (Literal['exact', 'stochastic']): The search strategy to use.

        Returns:
            Search: An instance of the Search class.
        """
        stmt_cache: DictProxy[Tuple[FrozenSet, int], Optional[DSLStmt]] | Dict[
            Tuple[FrozenSet, int], Optional[DSLStmt]
        ]
        if _NUM_WORKERS > 1:
            logger.info(f"Using {_NUM_WORKERS} workers")
            manager = Manager()
            stmt_cache = manager.dict()
        else:
            stmt_cache = {}
        return cls(
            LearningAlgo=learning_algorithm,
            SamplingAlgo=sampling_algorithm,
            input_data=input_data,
            columns=input_data.columns.tolist(),
            epsilon=epsilon,
            min_support=min_support,
            search_strategy=search_strategy,
            stmt_cache=stmt_cache,
        )

    def _synthesis_from_dag(
        self, dag: DAG, retained_columns: Optional[List[str]]
    ) -> DSLProg:
        """
        Synthesizes a DSL program from a given Directed Acyclic Graph (DAG).

        Args:
            dag (DAG): The DAG used for synthesizing the program.
            retained_columns (List[str]): The list of columns to retain in the program.

        Returns:
            DSLProg: A DSL program synthesized from the DAG.
        """
        prog = DSLProg()
        if retained_columns is None:
            for dependent_idx, _ in enumerate(self.columns):
                determinants = dag.get_parents(dependent_idx)
                if stmt := self._get_stmt(determinants, dependent_idx):
                    prog.add_stmt(stmt)
        else:
            for dependent_idx, _ in enumerate(retained_columns):
                determinants_indices = dag.get_parents(dependent_idx)
                recovered_determinants_indices = [
                    self.columns.index(retained_columns[i])
                    for i in determinants_indices
                ]
                recovered_dependent_idx = self.columns.index(
                    retained_columns[dependent_idx]
                )
                if stmt := self._get_stmt(
                    recovered_determinants_indices, recovered_dependent_idx
                ):
                    prog.add_stmt(stmt)
        return prog

    def _get_stmt(
        self, determinants_indices: List[int], dependent_idx: int
    ) -> Optional[DSLStmt]:
        """
        Retrieves or creates a DSL statement for given determinants and a dependent index.

        Args:
            determinants_indices (List[int]): Indices of determinant columns.
            dependent_idx (int): The index of the dependent column.

        Returns:
            Optional[DSLStmt]: The DSL statement for the specified indices, if exists.
        """
        cache_key = (frozenset(determinants_indices), dependent_idx)
        if cache_key in self.stmt_cache:
            return self.stmt_cache[cache_key]

        if len(determinants_indices) == 0:
            self.stmt_cache[cache_key] = None
            return None

        stmt = DSLStmt.create(
            dependent=self.columns[dependent_idx],
            determinants=[self.columns[i] for i in determinants_indices],
        )
        stmt.fit(self.input_data, self.epsilon, self.min_support)
        self.stmt_cache[cache_key] = stmt if stmt.cardinality > 0 else None
        return self.stmt_cache[cache_key]

    def _stochastic_search_over_mec(
        self,
        mec: MEC,
        retained_columns: Optional[List[str]],
    ) -> Optional[DSLProg]:
        """
        Performs stochastic search over a given Markov Equivalence Class (MEC). (Not implemented)

        Args:
            mec (MEC): The MEC to perform the search over.
            retained_columns (Optional[List[str]]): The list of columns to retain in the program.

        Raises:
            NotImplementedError: Indicates that the method is not yet implemented.
        """
        raise NotImplementedError("TODO")

    def _exact_search_over_mec(
        self,
        mec: MEC,
        retained_columns: Optional[List[str]],
    ) -> Optional[DSLProg]:
        """
        Performs exact search over a given Markov Equivalence Class (MEC).

        Args:
            mec (MEC): The MEC to perform the search over.
            retained_columns (Optional[List[str]]): The list of columns to retain in the program.

        Returns:
            Optional[DSLProg]: The best DSL program found during the search, if any.
        """
        max_coverage = float("-inf")
        best_prog = None

        def _run_dag(dag: DAG) -> DSLProg:
            return self._synthesis_from_dag(dag, retained_columns)

        for uncast_prog in p_uimap(
            _run_dag, DAG.enumerate_markov_equivalent_dags(mec), num_cpus=_NUM_WORKERS
        ):
            prog = cast(DSLProg, uncast_prog)
            assert isinstance(prog, DSLProg)
            if prog.coverage > max_coverage:
                max_coverage = prog.coverage
                best_prog = prog
        return best_prog

    def run(self) -> Optional[DSLProg]:
        """
        Performs program synthesis using the specified learning and sampling algorithms.

        Returns:
            Optional[DSLProg]: The best DSL program found during the search, if any.
        """
        logger.info("Starting MEC learning")
        mec, retained_columns = self.LearningAlgo.learn(
            self.input_data, self.SamplingAlgo
        )
        logger.info("MEC learning complete.")
        logger.info(f"# Edge: {len(mec.get_edges())}")
        logger.info(f"# Undirected edge: {len(mec.get_undirected_edges())}")
        if self.search_strategy == "stochastic":
            prog = self._stochastic_search_over_mec(mec, retained_columns)
        else:
            prog = self._exact_search_over_mec(mec, retained_columns)
        logger.info("Search complete.")
        return prog
