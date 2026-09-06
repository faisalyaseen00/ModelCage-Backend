"""Sandbox runtime abstraction."""


class SandboxRuntime:
    """Defines the execution environment for a sandbox."""

    def __init__(self, image=None, cpu_limit=None, memory_limit=None):
        self.image = image
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit

    def run(self, command):
        """Execute a command in the sandbox runtime."""
        return {"command": command, "status": "queued"}
