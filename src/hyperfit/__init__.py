"""Identification of hyperelastic strain energy models from three sources.

Modules:
    sources: Remote data sources, URLs and checksums.
    data: Download, cache, parse and preprocess the data.
    kinematics: Deformation gradient, invariants and stretches.
    trees: DEAP expression trees, SymPy conversion, tree edit distance.
    models: Strain energy model variants.
    mechanics: Stresses and tangents by automatic differentiation.
    fem: Finite element energy, weak-form residual and forward solver.
    parameters: Parameter vectors, bounds and expected values.
    fitting: Linear least squares, Levenberg-Marquardt and CMA-ES.
    graph: The governing equation as a directed acyclic graph.
    evaluation: Error metrics and physical checks.
    experiment: Steps of the computational experiment.
    plotting: Figures for the notebook.
"""
