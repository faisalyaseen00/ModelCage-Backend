Vendored from https://github.com/google/gvisor/tree/release-20260907.0/pkg/sentry/seccheck/points
Copyright The gVisor Authors, Apache License 2.0 (headers retained).
Only protobuf import paths changed for local packaging. Python generated with
`grpcio-tools==1.76.0`; imports made package-relative. Runtime dependency protobuf.
Wire reference: same release's sinks/remote/wire/wire.go and server/server.go.

Checked against release-20261019.0 on 2026-11-18: common.proto, sentry.proto and
syscall.proto differ only in comments and import paths, and `runsc trace metadata`
lists the same 992 syscall tracepoints, so these generated decoders are unchanged.
