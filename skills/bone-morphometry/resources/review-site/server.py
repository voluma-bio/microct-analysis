"""Read-only review server, restricted to app assets and explicit exported evidence."""

import argparse
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, unquote

import numpy as np
from PIL import Image

from export import COLORS

APP = Path(__file__).resolve().parent


def slice_png(hu, labels, axis, index, low, high, overlay):
    if axis not in (0, 1, 2) or not 0 <= index < hu.shape[axis] or not -10000 <= low < high <= 30000:
        raise ValueError("Invalid slice or intensity window")
    plane = np.take(hu, index, axis=axis)
    gray = (np.clip((plane - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if overlay:
        lab = np.take(labels, index, axis=axis)
        palette = np.array([[0, 0, 0]] + [[int(c[j : j + 2], 16) for j in (1, 3, 5)] for c in COLORS])
        present = lab > 0
        rgb[present] = 0.6 * rgb[present] + 0.4 * palette[lab[present]]
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    arrays = {
        "filtered": (
            np.load(a.run / "prepared-3/hu.npy", mmap_mode="r"),
            np.load(a.run / "review-07-separated/labels.npy", mmap_mode="r"),
        ),
        "raw": (
            np.load(a.run / "intake/hu.npy", mmap_mode="r"),
            np.load(a.run / "review-07-separated/labels.npy", mmap_mode="r"),
        ),
        "tibia": (
            np.load(a.run / "review-08-anatomical/tibia-hu.npy", mmap_mode="r"),
            np.load(a.run / "review-08-anatomical/tibia-labels.npy", mmap_mode="r"),
        ),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            path = unquote(url.path)
            try:
                if path == "/api/slice":
                    q = parse_qs(url.query)

                    def get(k, d):
                        return q.get(k, [d])[0]

                    hu, labels = arrays[get("source", "filtered")]
                    data = slice_png(
                        hu,
                        labels,
                        int(get("axis", "0")),
                        int(get("index", "438")),
                        float(get("low", "0")),
                        float(get("high", "5000")),
                        get("overlay", "1") == "1",
                    )
                    kind = "image/png"
                else:
                    if path.startswith("/assets/"):
                        root = a.assets.resolve()
                        relative = path[len("/assets/") :]
                    elif path.startswith("/vendor/"):
                        root = (APP / "node_modules/three").resolve()
                        relative = path[len("/vendor/") :]
                    elif path in (
                        "/",
                        "/index.html",
                        "/app.js",
                        "/style.css",
                        "/parameters.json",
                        "/parameter-panel.js",
                    ):
                        root = APP
                        relative = path.lstrip("/") or "index.html"
                    else:
                        self.send_error(404)
                        return
                    target = (root / relative).resolve()
                    if not target.is_relative_to(root) or not target.is_file():
                        self.send_error(404)
                        return
                    data = target.read_bytes()
                    kind = {
                        ".html": "text/html",
                        ".js": "text/javascript",
                        ".css": "text/css",
                        ".json": "application/json",
                        ".png": "image/png",
                        ".md": "text/plain",
                        ".py": "text/plain",
                        ".pdf": "application/pdf",
                    }.get(target.suffix, "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, KeyError, IndexError):
                self.send_error(400, "Invalid review request")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Review server: http://127.0.0.1:{a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
