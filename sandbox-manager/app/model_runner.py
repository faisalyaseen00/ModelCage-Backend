"""Source for the only Python program ModelCage permits inside the POC sandbox."""

import re

# Plain pickles are loaded with pickle.load; PyTorch checkpoints with torch.load.
PICKLE_SUFFIXES = frozenset({".pkl", ".pickle"})
TORCH_SUFFIXES = frozenset({".bin", ".pt", ".pth"})
MODEL_SUFFIXES = PICKLE_SUFFIXES | TORCH_SUFFIXES


def loading_failure(output, exit_code):
    """Summarize known loading errors without publishing arbitrary model output.

    Output is supplied by an untrusted process; these are reported diagnostics,
    not proof of the artifact's safety or the cause of failure.
    """
    tail = (output or '')[-4096:]
    missing_class = re.search(r"AttributeError: Can't get attribute '([A-Za-z_][A-Za-z0-9_.]{0,119})'", tail)
    if missing_class:
        return (f"Model loading reported missing class '{missing_class[1]}'. "
                "Provide the original class source and its dependencies for the sandbox.")
    missing_module = re.search(r"ModuleNotFoundError: No module named '([A-Za-z_][A-Za-z0-9_.]{0,119})'", tail)
    if missing_module:
        return f"Model loading reported missing dependency '{missing_module[1]}' in the sandbox image."
    if 'UnpicklingError:' in tail or 'EOFError:' in tail:
        return 'Model loading reported an invalid, incompatible or incomplete pickle.'
    return f'Model workload exited with code {exit_code}; capture and execution status are separate.'

RUNNER_SOURCE = '''import os
from pathlib import Path
import pickle
import sys

if len(sys.argv) != 2:
    raise SystemExit("expected exactly one model path")

Path(sys.argv[1]).with_name("runner.pid").write_text(str(os.getpid()), encoding="ascii")
if sys.argv[1].endswith((".bin", ".pt", ".pth")):
    # PyTorch checkpoints are loaded the way a victim's from_pretrained/torch.load
    # would, so pickle reducers inside the archive execute here, in the sandbox.
    import torch
    torch.load(sys.argv[1], map_location="cpu", weights_only=False)
else:
    with open(sys.argv[1], "rb") as model_file:
        pickle.load(model_file)
'''
