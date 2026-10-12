"""ModelCage-wide sandbox policy settings."""

# ModelCage sandboxes must run under gVisor. Docker's host default remains
# independent from this application policy.
MODELCAGE_RUNTIME = "runsc"

# Phase 3.1 filesystem isolation. The only writable filesystem exposed to a
# sandbox is Docker-managed tmpfs storage at /tmp.
READ_ONLY_ROOT_FILESYSTEM = True
WRITABLE_TMPFS = {"/tmp": "rw"}
FORBIDDEN_MOUNT_OPTIONS = frozenset({"binds", "mounts", "volumes", "volumes_from"})

# Phase 3.2 resource isolation. Values are passed to Docker as bytes and
# nanoseconds of CPU time so inspection can validate them without conversion.
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
CPU_LIMIT_NANO_CPUS = 1_000_000_000
PID_LIMIT = 64
FORBIDDEN_RESOURCE_OPTIONS = frozenset(
    {
        "cpu_period",
        "cpu_quota",
        "cpu_rt_period",
        "cpu_rt_runtime",
        "cpu_shares",
        "cpuset_cpus",
        "cpuset_mems",
        "kernel_memory",
        "mem_limit",
        "mem_reservation",
        "mem_swappiness",
        "memswap_limit",
        "nano_cpus",
        "pids_limit",
    }
)

# Phase 3.3 privilege isolation. Sandboxes receive no Linux capabilities and
# may never be privileged.
CAP_DROP = ["ALL"]
PRIVILEGED = False
FORBIDDEN_PRIVILEGE_OPTIONS = frozenset({"cap_add", "cap_drop", "privileged"})

# Phase 3.4 network isolation. Docker's none network has no external or host
# network attachment.
NETWORK_MODE = "none"
FORBIDDEN_NETWORK_OPTIONS = frozenset(
    {
        "dns",
        "dns_opt",
        "dns_search",
        "extra_hosts",
        "links",
        "mac_address",
        "network",
        "network_disabled",
        "network_mode",
        "networking_config",
        "ports",
        "port_bindings",
        "publish_all_ports",
    }
)

# Options that can expose host resources or change the namespace boundary.
# ModelCage owns the complete container configuration; these are rejected even
# when Docker would otherwise accept them.
FORBIDDEN_ISOLATION_OPTIONS = frozenset(
    {
        "cgroup_parent",
        "cgroupns",
        "devices",
        "device_cgroup_rules",
        "device_requests",
        "group_add",
        "init",
        "ipc",
        "pid_mode",
        "security_opt",
        "sysctls",
        "uts_mode",
        "user",
        "userns_mode",
    }
)

# Phase 3.5 execution isolation. This is intentionally not caller-configurable.
EXECUTION_TIMEOUT_SECONDS = 30

# Phase 6A controlled pickle runner image. Callers cannot select the image or
# runner command used for model deserialization.
MODELCAGE_RUNNER_IMAGE = "modelcage-runner:phase6a"

# Temporary development pause, requested while debugging execution. Re-enable
# this flag to restore SHA-256 calculation and content-addressed quarantine reuse.
# Quarantine copying and sandbox isolation remain enabled.
MODEL_HASHING_ENABLED = False
