# Environment Checks

Starter probes; adapt to the active protocol. In this repo, prefer `uv run python`
so the command uses the project environment instead of a missing or unrelated
system `python`.

```bash
pwd
uv run python --version
uv --version
uv run python - <<'PY'
import importlib.util
for name in ['numpy', 'scipy', 'skimage', 'SimpleITK', 'matplotlib', 'pyvista', 'torch']:
    spec = importlib.util.find_spec(name)
    print(f'{name}: {bool(spec)}')
PY
uv run python - <<'PY'
try:
    import torch
    print('cuda:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
except Exception as exc:
    print('torch probe failed:', exc)
PY
```

If the project is not managed by `uv`, use the active interpreter explicitly
(e.g. `.venv/bin/python`) and record that choice. If a required probe fails,
either adapt the protocol to a supported path or stop with a clear installation
request. Do not continue with degraded analysis silently.
