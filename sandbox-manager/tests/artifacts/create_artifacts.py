"""Create the two harmless Phase 6A pickle specimens."""

from pathlib import Path
import pickle


class HarmlessMarker:
    def __reduce__(self):
        return (Path.touch, (Path("/tmp/modelcage-malicious-marker"),))


target = Path(__file__).parent
(target / "benign_model.pkl").write_bytes(pickle.dumps({"model": "benign"}))
(target / "malicious_test_model.pkl").write_bytes(pickle.dumps(HarmlessMarker()))