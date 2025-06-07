# Guardrail Codebase Overview

This document provides a high level overview of the Guardrail project.

Guardrail is a research artifact for the paper *"Guardrail: Solidifying ML-Integrated SQL Queries with Automated Integrity Constraint Synthesis"*. The repository contains tooling for error detection in datasets and utilities for executing extended SQL queries that integrate machine learning models.

## Repository Layout

- **datasets/** – Example datasets and pre-generated splits.
- **example_query/** – Sample SQL queries used in experiments.
- **lib/** – External libraries such as the `fastmecenumeration` Julia module.
- **nsyn/** – Main Python source code.
- **scripts/** – Scripts for reproducing paper experiments.
- **src/** – Packaging entry point (contains `nsyn` for editable installs).

## Key Components

### `nsyn/app`
Entry points for running the system. `q2_executor.py` executes SQL queries with ML models, and `error_detector.py` applies synthesized programs to identify data errors.

### `nsyn/dataset`
Utilities for loading datasets and generating noisy splits.

### `nsyn/dsl`
Defines the Domain Specific Language used to express integrity constraints. Important classes include `DSLStmt`, `DSLProg`, and related condition and assignment helpers.

### `nsyn/learner`
Implements structure learning algorithms (PC, GES, BLIP) to infer causal graphs that drive program synthesis.

### `nsyn/search`
Given a learned Markov Equivalence Class (MEC), enumerates equivalent DAGs and constructs the best DSL program from them.

### `nsyn/sampler`
Sampling strategies used during learning. `AuxiliarySampler` performs circular-shift style sampling to augment the data.

### `nsyn/util`
Shared utilities such as the DAG/MEC helpers, logging, and visualization routines.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
2. Run a query executor example:
   ```bash
   python nsyn/app/q2_executor.py -h
   ```

For more details see the main [README](../README.md).

## Query Executor CLI

The script `nsyn/app/q2_executor.py` executes extended SQL files that may
invoke machine learning models. It accepts a few key options:

- `--query_path`/`-p` – path to a single `.sql` file or a directory of query
  files. When a directory is provided, all contained SQL files are processed
  sequentially.
- `--query`/`-q` – supply a query string directly on the command line.
- `--noisy_level`/`-n` – value used to replace occurrences of the placeholder
  `noisy` when reading a query file (defaults to `05`).

Running the script without specifying a query path or a query prints the help
text. Example invocations:

```bash
# Display CLI usage
python nsyn/app/q2_executor.py -h

# Execute a single query file with a specified noise level
python nsyn/app/q2_executor.py -p example_query/default/adult.sql -n 05

# Execute all queries in a directory
python nsyn/app/q2_executor.py -p example_query/default/ -n 05
```

Each SQL file can define model references at the bottom (e.g. `M1: some_model_id, autogluon`).
The Query2 grammar in `nsyn/app/q2_util/q2_grammar.py` parses these definitions
along with the query clauses to determine which models to invoke during
execution.
