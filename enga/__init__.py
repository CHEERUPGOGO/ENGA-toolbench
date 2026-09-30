"""ENGA: Evolutionary Nested Greedy Algorithm for tool composition.

Implements the method described in "Nested Greedy Search for Tool Selection"
(ICLR 2026 submission) on top of ToolBench / ToolRet data.

Module map (paper section -> module):
  Preliminary Alg.1 (NGA)          -> enga/nga.py   (decode, oracle-greedy)
  Method 3.2 parameterized NGA     -> enga/nga.py   (decode with alpha)
  Method 3.3 evolutionary search   -> enga/es.py    (OpenAI-ES)
  Method 3.5 QD archive            -> enga/qd.py    (Archive)
  Method 3.4 budgeted LLM eval     -> enga/evaluator.py (EvalBudget)
  Evaluation RQ1-RQ5               -> scripts/run_experiment.py
"""

__version__ = "0.1.0"

from .config import Config  # noqa: F401
