# Binary Classification Engine for Mowja

This directory vendors the MIT-licensed `bcpipeline` implementation from the original repository:
https://github.com/mzydhaif11-ctrl/binary-classification-pipeline-

## Scope

- Kept as an independent, optional classification engine.
- Does not replace or modify the adaptive memory or RAG engines.
- Does not automatically train on conversations, memories, or user data.
- No API route is wired into the FastAPI server in this initial integration.
- Install the optional dependencies from `requirements-classification.txt` before importing this engine.

## Import

From the repository root:

```python
from engines.binary_classification import BinaryClassifier
```

## Important validation notes

This is a first-stage code integration, not a claim of production readiness. Before exposing it through an API, add tests for numeric/categorical inputs, missing values, single-class edge cases, probability calibration, serialization compatibility, and data leakage. The upstream implementation currently expects a binary target and uses median imputation plus standard scaling, so categorical columns require preprocessing before use.

The original MIT license is retained in this directory.
