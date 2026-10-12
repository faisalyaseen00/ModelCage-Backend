# Phase 6A artifacts

Run `python create_artifacts.py` from this directory to generate:

- `benign_model.pkl`, a plain dictionary pickle.
- `malicious_test_model.pkl`, a harmless pickle specimen that creates only a
  `/tmp/modelcage-malicious-marker` file inside the disposable sandbox.

The test suite generates equivalent temporary artifacts automatically.