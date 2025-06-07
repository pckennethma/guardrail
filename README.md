<p align="center">
    <h1 align="center">Guardrail</h1>
</p>

<p align="center">
	<img src="https://img.shields.io/github/license/pckennethma/guardrail?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/pckennethma/guardrail?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/pckennethma/guardrail?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/pckennethma/guardrail?style=flat&color=0080ff" alt="repo-language-count">
</p>

<p align="center">
	<em>This is a research artifact for paper "Guardrail: Automated Integrity Constraint Synthesis From Noisy Data" (SIGMOD 2026).</em>
</p>

<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
</p>
<hr>

## 📍 Overview

Welcome to the repository for Guardrail, a suite of tools and utilities designed for enhancing the robustness and reliability of ML-integrated SQL queries through automated integrity constraint synthesis. This project aims to mitigate the impact of data errors on machine learning models and their predictions by integrating constraint-based error detection and correction directly within the SQL querying process.

## 📂 Directory Structure

The repository is organized as follows:

```
prog-syn/
│
├── datasets/                   # Contains example datasets used for testing and validation
├── example_query/              # Example SQL queries demonstrating the usage of Guardrail
├── lib/                        # Core libraries and utilities
├── models/                     # Machine learning models used in the project
├── nsyn/                       # Main application source code
│   ├── app/
│   │   ├── __pycache__/
│   │   ├── error_detector_util/
│   │   ├── ml_backend/         # Backend services for machine learning tasks
│   │   ├── q2_util/
│   │   ├── error_detector.py
│   │   ├── q2_executor.py
│   ├── dataset/
│   ├── dsl/                    # Domain-specific language definitions for constraint synthesis
│   │   ├── __pycache__/
│   │   ├── assign.py
│   │   ├── branch.py
│   │   ├── condition.py
│   │   ├── prog.py
│   │   ├── stmt.py
│   │   ├── term.py
│   ├── tests/                  # Unit tests for the project
│   ├── util/
│   │   ├── __init__.py
│   │   ├── learner.py
│   │   ├── run.py
│   │   ├── sampler.py
│   │   ├── search.py
│   ├── nsyn.egg-info/          # Package metadata
├── scripts/                    # Scripts for running experiments in our paper
├── statistics/                 # Statistical analysis and validation tools
└── ...
```

## 🚀 Getting Started

To get started with Guardrail, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/pckennethma/prog-syn.git
    cd prog-syn
    ```

2. **Install:**

    ```sh
    pip install -r requirements.txt
    pip install -e .
    ```

3. **Run:**

    ```sh
    python nsyn/app/q2_executor.py -h
    ```

## 📖 Documentation

Documentation is currently under development. Please find an overview in [doc](docs/overview.md).

<p align="center">
    <em>Thank you for using Guardrail!</em>
</p>
