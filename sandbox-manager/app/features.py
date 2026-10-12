"""Reproducible offline entry-only baseline and experimental n-gram features."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def ngrams(events, *, per_thread=False):
    streams = defaultdict(list)
    analysis_ids = set()
    seen_sequences = set()
    last = 0
    for event in events:
        analysis_ids.add(event["analysis_id"])
        if len(analysis_ids) > 1:
            raise ValueError("Cannot mix analyses")
        sequence = event["sequence"]
        if sequence in seen_sequences or sequence <= last:
            raise ValueError("Input must have unique ordered sequence numbers")
        last = sequence
        seen_sequences.add(sequence)
        if event.get("event_source") != "gvisor_seccheck" or not event.get("tracepoint", "").startswith("syscall/"):
            raise ValueError("Non-syscall source in raw stream")
        if event.get("phase") not in {"enter", "exit"}:
            raise ValueError("Unknown syscall phase")
        if event["phase"] == "enter":
            key = f'{event["process_id"]}:{event["thread_id"]}:{event.get("thread_start_time_ns")}' if per_thread else "collector_order"
            streams[key].append(event["syscall_name"])
    output = {}
    for key, names in sorted(streams.items()):
        features = {}
        for n in range(1, 5):
            counts = Counter(tuple(names[i:i+n]) for i in range(len(names)-n+1))
            features[str(n)] = [{"syscalls": list(words), "frequency": count, **({"present": True} if n == 1 else {})}
                               for words, count in sorted(counts.items())]
        output[key] = features
    return {"schema_version": 1, "sequence_policy": "per_thread" if per_thread else "collector_receive_order",
            "baseline": "1-gram presence/frequency", "experimental": [2, 3, 4], "streams": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--per-thread", action="store_true")
    args = parser.parse_args()
    print(json.dumps(ngrams([json.loads(line) for line in args.trace.read_text().splitlines()],
                           per_thread=args.per_thread), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
