"""Configurable evidence-based observations, never malware verdicts."""
from dataclasses import dataclass
import ipaddress


@dataclass(frozen=True)
class ObservationPolicy:
    enabled_groups: tuple = ("process", "network", "filesystem", "privilege", "memory", "reconnaissance")
    approved_write_roots: tuple = ("/tmp",)
    sensitive_paths: tuple = ("/etc/shadow", "/etc/passwd", "/root/.ssh", "/proc/kcore")
    allowed_executables: tuple = ("/usr/local/bin/python3", "/usr/bin/python3", "python3")


def under(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def observe(events, policy=None):
    policy = policy or ObservationPolicy()
    exits = {e["entry_sequence"]: e for e in events if e.get("entry_sequence") is not None}
    observations = []
    writable = {}
    for event in events:
        if event.get("phase") != "enter":
            continue
        name, args = event["syscall_name"], event["arguments"]
        exit_event = exits.get(event["sequence"])
        outcome = "attempted" if exit_event is None else "failed" if exit_event.get("error_status") else "succeeded"
        evidence = [event["sequence"]] + ([exit_event["sequence"]] if exit_event else [])
        def add(group, code, reason, priority="normal"):
            if group in policy.enabled_groups:
                observations.append({"analysis_id": event["analysis_id"], "observation_id": len(observations)+1,
                    "group": group, "reason_code": code, "priority": priority, "outcome": outcome,
                    "source_sequences": evidence, "reason": reason, "syscall_name": name})
        path = args.get("pathname") or args.get("fd_path")
        # Only absolute decoded paths have reliable namespace-relative semantics here.
        if name in {"execve", "execveat"} and path and path not in policy.allowed_executables:
            add("process", "unexpected_process_execution", "Executed or attempted an executable outside the configured allowlist")
        if name in {"fork", "vfork"} or (name in {"clone", "clone3"} and args.get("flags") is not None and not args["flags"] & 0x10000):
            add("process", "process_creation", "Process creation observed; CLONE_THREAD not set where flags are available")
        if name in {"socket", "connect", "sendto", "sendmsg", "bind", "listen", "accept", "accept4"}:
            external = False
            address = args.get("address")
            if isinstance(address, dict) and address.get("ip"):
                ip = ipaddress.ip_address(address["ip"])
                external = not ip.is_loopback and not ip.is_unspecified
            add("network", "network_attempt", "Offline workload network operation; denial is retained" + ("; external address decoded" if external else ""),
                "high" if external and name in {"connect", "sendto", "sendmsg"} else "normal")
        if isinstance(path, str) and path.startswith("/"):
            if any(under(path, root) for root in policy.sensitive_paths):
                add("reconnaissance", "sensitive_path_access", "Decoded path matches a configured sensitive path inside the container")
            modifying = name in {"write", "writev", "pwrite64", "unlink", "unlinkat", "rename", "renameat", "renameat2", "chmod", "fchmod", "mkdir", "mkdirat", "creat"}
            if name in {"open", "openat"} and args.get("flags", 0) & (3 | 0x40 | 0x200 | 0x400):
                modifying = True
            if modifying:
                outside = not any(under(path, root) for root in policy.approved_write_roots)
                add("filesystem", "filesystem_modification", "Modification of a decoded container path" + (" outside approved temporary locations" if outside else ""), "high" if outside else "normal")
        if name in {"unlink", "unlinkat", "rename", "renameat", "renameat2", "chmod", "fchmod", "fchmodat", "mkdir", "mkdirat", "truncate", "ftruncate", "write", "writev", "pwrite64"} and not path:
            add("filesystem", "filesystem_modification", "Modification attempt observed; path unavailable, no path-based conclusion")
        if name == "memfd_create":
            add("memory", "anonymous_file_creation", "Anonymous file creation observed; not inherently malicious")
        if name in {"setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid", "capset", "mount", "umount2", "unshare", "setns", "ptrace", "bpf", "init_module", "finit_module", "keyctl"}:
            isolation = name in {"mount", "umount2", "unshare", "setns", "ptrace", "bpf", "init_module", "finit_module"}
            add("privilege", "isolation_change_attempt" if isolation else "privilege_boundary_attempt",
                "Boundary-related operation observed; outcome is not proof of compromise", "high")
        if name in {"mmap", "mprotect"} and isinstance(args.get("prot"), int):
            prot = args["prot"]
            key = (event["process_id"], args.get("addr"))
            if prot & 2:
                writable[key] = event["sequence"]
            if prot & 4 and (prot & 2 or key in writable):
                add("memory", "executable_memory_pattern", "Writable/executable memory combination observed; JIT activity may be benign")
                if observations and observations[-1]["source_sequences"] == evidence and key in writable:
                    observations[-1]["source_sequences"] = sorted(set(evidence + [writable[key]]))
    return observations
