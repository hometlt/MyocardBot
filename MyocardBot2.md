# MyocardBot2

## Vision

A modular ECG paper digitization system that reconstructs the most
probable ECG signal rather than merely extracting pixels.

## Core Philosophy

-   Treat digitization as probabilistic path reconstruction.
-   Never permanently erase information (text/grid become probability
    layers).
-   Every stage outputs both results and confidence.
-   Recover signals using image evidence plus ECG continuity
    constraints.

## Pipeline

1.  Image Loader
2.  Geometry Analyzer
3.  Grid Detector
4.  Layout Detector
5.  OCR & Semantic Analysis
6.  ECG Trace Probability Estimator
7.  Noise / Artifact Detector
8.  Cost Map Builder
9.  Path Solver
10. Gap Recovery
11. Beat Detector
12. Template Builder
13. Signal Optimizer
14. Quality Analyzer
15. Export

## Blackboard Architecture

``` python
state.image
state.geometry
state.grid
state.layout
state.text
state.trace_probability
state.noise
state.artifacts
state.cost_map
state.paths
state.signals
state.quality
```

## Cost Map

Each pixel stores probabilities: - P(trace) - P(text) - P(grid) -
P(noise) - P(artifact)

## OCR Strategy

Do not erase text. OCR identifies semantic objects and contributes
penalties to the cost map instead.

## Path Optimization

Use dynamic programming / shortest-path optimization with costs from
image evidence, continuity, curvature, text, grid, noise and template
similarity.

## ECG Constraints

Use physiological priors only for ambiguous regions.

## Robustness

Support text over traces, crossings, folds, glare, blur, missing ink,
scanned and photographed ECGs.

## Design Principles

-   Modular
-   Confidence propagation
-   Replaceable algorithms
-   Debug visualizations
-   Classical CV first
-   Future AI modules optional
