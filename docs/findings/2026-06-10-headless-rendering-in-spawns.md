# Headless rendering in Meridian spawns works; the fragile part is Jupyter kernel transport, not PyVista offscreen rendering

**Date:** 2026-06-10
**Scope:** `src/microct_analysis/processing/rendering.py`, `jupyter-workbench` session transport

## Summary

I tested this from the current Meridian task/spawn environment and **PyVista/VTK offscreen rendering works**.
`render_surface_view()` in `src/microct_analysis/processing/rendering.py:294-339` uses
`pv.Plotter(off_screen=True)`, and that succeeds here without `DISPLAY`, Xvfb, or
`pv.start_xvfb()`.

The active backend is **`vtkEGLRenderWindow`**, not an on-screen X11 window. VTK logs a
`bad X server connection. DISPLAY=` warning first, but still falls back to EGL and writes the
PNG successfully.

The part that really depends on sandbox policy is **`jupyter-workbench` kernel transport**.
`jupyter-workbench open/exec` starts an IPython kernel and connects to it through a Jupyter
connection file that uses **`transport: "tcp"` and `ip: "127.0.0.1"`**. If a spawn profile
forbids localhost socket creation/connection, the failure will happen in kernel startup/client
channel connection, not in VTK offscreen rendering itself.

## What I tested

### 1. Direct offscreen PyVista render in this spawn

Command:

```bash
uv run python - <<'PY'
from pathlib import Path
import pyvista as pv
pv.OFF_SCREEN = True
sphere = pv.Sphere()
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(sphere)
out = '/tmp/test_offscreen_meridian.png'
plotter.screenshot(out)
plotter.close()
print(Path(out).exists(), Path(out).stat().st_size)
PY
```

Observed:

- warning: `vtkXOpenGLRenderWindow ... bad X server connection. DISPLAY=`
- output file exists: `/tmp/test_offscreen_meridian.png`
- output size: `44472` bytes

Backend probe:

```text
renwin_class vtkEGLRenderWindow
offscreen 1
```

### 2. `jupyter-workbench` session + offscreen render inside the kernel

Commands:

```bash
uv run jupyter-workbench open --session-id headless-test --root-dir /tmp/jwb-headless-test
uv run jupyter-workbench exec --session-id headless-test --root-dir /tmp/jwb-headless-test \
  "import pyvista as pv; from pathlib import Path; pv.OFF_SCREEN=True; \
   p=pv.Plotter(off_screen=True); p.add_mesh(pv.Sphere()); \
   out='/tmp/jwb-headless-test-render.png'; p.screenshot(out); p.close(); \
   print(Path(out).exists(), Path(out).stat().st_size)"
```

Observed:

- session opened successfully
- kernel execution succeeded
- render inside the workbench kernel also produced a PNG (`44472` bytes)
- backend inside the kernel was also `vtkEGLRenderWindow`

### 3. Interactive PyVista/trame path

Command:

```bash
uv run jupyter-workbench exec --session-id headless-test --root-dir /tmp/jwb-headless-test \
  "import pyvista as pv; pv.OFF_SCREEN=True; p=pv.Plotter(); p.add_mesh(pv.Sphere()); p.show(jupyter_backend='trame')"
```

Observed:

- execution succeeded
- `jupyter-workbench` recorded a visualization delta with a browser URL like:
  `http://localhost:37343/index.html?...`

That confirms the interactive review path also relies on a **localhost server/socket**, separate
from the EGL offscreen screenshot path.

## Environment facts

- `DISPLAY=None`
- `WAYLAND_DISPLAY=None`
- `PYVISTA_OFF_SCREEN=None`
- `vtk` version: `9.6.1`
- `pyvista` version: `0.47.3`
- `pv.start_xvfb()` is available but deprecated and **fails** here because Xvfb is not installed
- despite that, offscreen rendering still works via EGL

I also tested:

- `DISPLAY=:99` → still rendered successfully with `vtkEGLRenderWindow`
- `PYVISTA_OFF_SCREEN=true` → also rendered successfully

So Xvfb is **not required** in this environment.

## Code-path diagnosis

### `jupyter-workbench exec` call chain

- CLI entry: `jupyter_workbench/cli.py:164-188`
- execution service: `jupyter_workbench/core/execution_service.py:31-108`
- kernel start/connect/execute: `jupyter_workbench/adapters/kernel/jupyter_client_manager.py:26-106`

Key transport details:

- kernel startup: `jupyter_client_manager.py:28-45`
  - creates `KernelManager(connection_file=...)`
  - `manager.start_kernel(...)`
  - `client = manager.blocking_client()`
  - `client.start_channels()`
  - `client.wait_for_ready(...)`
- reconnect path: `jupyter_client_manager.py:52-57`
  - `BlockingKernelClient(connection_file=connection_file)`
  - `client.load_connection_file(...)`
  - `client.start_channels()`
- execution loop: `jupyter_client_manager.py:69-99`
  - `client.execute(code)`
  - `client.get_iopub_msg(...)`

The session connection file written by the open call was:

```json
{
  "ip": "127.0.0.1",
  "transport": "tcp",
  "shell_port": 57487,
  "iopub_port": 58783,
  "stdin_port": 36773,
  "control_port": 56233,
  "hb_port": 36467
}
```

So the kernel path is definitively **Jupyter over localhost TCP**, not stdio and not a Unix socket.

## Root cause / non-cause

### Not the cause

- `render_surface_view()` itself is not blocked here.
- Missing `DISPLAY` is not fatal.
- Xvfb is not required.

### Actual fragile dependency

The fragile dependency is **local socket permission** for Jupyter kernel channels (and for trame in
interactive mode).

If a different Meridian spawn profile blocks loopback TCP sockets, the likely breakpoints are:

- `client.start_channels()` / `client.wait_for_ready()` during session open
  (`jupyter_client_manager.py:42-45`)
- `client.start_channels()` during reconnect
  (`jupyter_client_manager.py:54-56`)
- `client.get_iopub_msg()` during execution
  (`jupyter_client_manager.py:76-79`)

I did **not** reproduce the reported `PermissionError` in this spawn. Based on the code path and the
observed `kernel.json`, that error would belong to the Jupyter client/socket layer, not the VTK
rendering layer.

## Resolution options, ranked

### 1. Recommended: run the landmarker in a spawn context that allows localhost kernel sockets

Best fit with the current architecture.

Why:

- keeps `jupyter-workbench exec` intact
- keeps interactive trame review available
- offscreen screenshot rendering already works

What needs to be allowed:

- writes to the session root (for `.jupyter-workbench/...` or equivalent)
- localhost TCP connections for Jupyter kernel channels
- localhost TCP for trame if interactive review is used

### 2. Acceptable fallback: keep offscreen screenshot rendering in spawned Python, but avoid `jupyter-workbench exec`

Use direct Python entrypoints like `prepare_landmark_session()`, `render_surface_view()`, and
`render_slice_view()` without a workbench kernel.

Tradeoff:

- avoids Jupyter socket dependence
- loses the current persistent notebook/session model and interactive review flow

### 3. Low value here: add Xvfb / force `DISPLAY=:99`

Not recommended.

Why:

- EGL already works
- `pv.start_xvfb()` fails due missing Xvfb, but rendering still succeeds
- this would treat the wrong problem

### 4. Possible but larger change: redesign workbench transport away from Jupyter TCP channels

Example directions: subprocess stdio transport or a different local IPC model.

Tradeoff:

- would remove the socket-policy sensitivity
- much larger architectural change than the evidence justifies

## Recommended execution context for the landmarker agent

Use the existing spawned agent context **only if** it keeps:

1. writable session storage, and
2. localhost socket access for `jupyter-workbench` kernel/trame channels.

Given today's tests, the rendering implementation in `processing/rendering.py` does **not** need to
change for headless execution. If the team still sees failures in another spawn mode, the next thing
to inspect is that mode's sandbox/network policy — specifically whether it denies loopback TCP.
