"""Sandbox lifecycle management."""


class SandboxLifecycle:
    """Tracks the runtime lifecycle state of a sandbox."""

    def __init__(self):
        self.state = "created"

    def start(self):
        """Mark sandbox as started."""
        self.state = "running"
        return self.state

    def stop(self):
        """Mark sandbox as stopped."""
        self.state = "stopped"
        return self.state
