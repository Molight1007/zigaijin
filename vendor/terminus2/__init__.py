from .terminus_2 import Terminus2


class AgentHarness(Terminus2):
    """Official Harbor Terminus-2 exposed under the runner's agent name."""


__all__ = ["Terminus2", "AgentHarness"]
