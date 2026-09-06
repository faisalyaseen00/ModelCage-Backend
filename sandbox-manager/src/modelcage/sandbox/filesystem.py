"""Sandbox filesystem abstraction."""


class SandboxFilesystem:
    """Manages the view of the sandbox filesystem."""

    def __init__(self, root_dir="/workspace"):
        self.root_dir = root_dir

    def list_dir(self, path=None):
        """List files within a sandbox directory."""
        target = path or self.root_dir
        return {"path": target, "entries": []}

    def write_file(self, path, content):
        """Write content to a file in the sandbox."""
        return {"path": path, "written": bool(content), "size": len(content or "")}
