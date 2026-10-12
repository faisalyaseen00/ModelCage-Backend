"""Source for the only Python program ModelCage permits inside the POC sandbox."""

import re


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
with open(sys.argv[1], "rb") as model_file:
    pickle.load(model_file)
'''
