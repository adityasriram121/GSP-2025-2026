"""Code package initializer.

This makes the `code` directory importable as a package so worker
processes can import modules as `code.<module>`.
"""

__all__ = [
    "data_generation",
    "centralized_training",
    "federated_training",
    "comprehensive_trials",
    "metrics_visualization",
    "model_manager",
    "main_pipeline",
]
