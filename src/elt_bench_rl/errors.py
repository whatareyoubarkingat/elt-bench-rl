class InfrastructureError(RuntimeError):
    """An external failure; exclude the episode from RL rather than assigning zero."""


class PolicyError(ValueError):
    """An action outside the model's allowed capability set."""
