"""Sandbox policy configuration."""


class SandboxPolicy:
    """Defines constraints and rules for sandbox operation."""

    def __init__(self, allow_network=True, max_runtime_seconds=300, allowed_commands=None):
        self.allow_network = allow_network
        self.max_runtime_seconds = max_runtime_seconds
        self.allowed_commands = allowed_commands or []

    def is_allowed(self, command):
        """Check if a command is permitted by policy."""
        if not self.allowed_commands:
            return True
        return command in self.allowed_commands
