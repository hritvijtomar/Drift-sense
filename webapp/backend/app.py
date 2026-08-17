#!/usr/bin/env python3
"""
app.py
------
Drift-Sense web API + static frontend server.

Endpoints:
    GET  /                                 -> frontend (index.html)
    GET  /api/health                       -> liveness check
    GET  /api/examples                     -> curated demo case registry
    GET  /api/media/<kind>/<filename>      -> serve reference/search PNGs
    POST /api/upload                       -> validate + store an uploaded image, return a token
    GET  /api/run/stream                   -> Server-Sent Events stream of a real localization run

This process is the ONLY place a web request touches the localization
pipeline, and it does so exclusively through webapp/backend/
localization_service.py, which itself only calls the existing,
unmodified functions in localization/template_matching.py and
localization/feature_matcher.py. localize.py (the CLI) is untouched and
keeps working independently -- see webapp/backend/verify_parity.py.
"""
import os
import sys
import json
import uuid
import time
import shutil
import tempfile
import traceback

from flask import Flask, request, Response, jsonify, send_from_directory, abort, stream_with_context

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.join(BACKEND_DIR, "..")
REPO_ROOT = os.path.join(WEBAPP_DIR, "..")
FRONTEND_DIR = os.path.join(WEBAPP_DIR, "frontend")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "localization"))
from localization_service import run_localization, PipelineError, AMBIGUOUS_GAP, GT_MATCH_TOLERANCE_PX  # noqa: E402
import examples as examples_registry  # noqa: E402
from template_matching import DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX  # noqa: E402

# ---------------------------------------------------------------------------
# Upload handling config
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "driftsense_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_IMAGE_DIM = 8000  # px, sanity bound against decompression-bomb style files
UPLOAD_TOKEN_TTL_SEC = 60 * 60  # 1 hour

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(os.path.join(FRONTEND_DIR, "static"), path)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": time.time()})


@app.route("/api/config")
def config():
    """
    Exposes real backend constants the frontend needs to build an honest
    UI, instead of hardcoding guesses in JavaScript:

      - reference_scale_range: the reference:search scale sweep
        (localization/template_matching.py). The backend assumes the
        reference image is a ~1/scale zoomed-in depiction of the target
        pattern -- e.g. at scale~0.1, a reference is expected to be ~10x
        larger (in linear pixels) than how the same feature appears in
        the search image. This is why "Explore Your Own Images" upscales
        a cropped region before sending it as the reference (see
        webapp/README.md, "Upload mode & the scale convention").
      - ambiguous_gap_threshold / gt_match_tolerance_px: same constants
        used server-side for the ambiguity flag and failure
        classification, so any UI copy referencing them can never drift
        out of sync with the backend.
    """
    mid_scale = (DEFAULT_SCALE_MIN + DEFAULT_SCALE_MAX) / 2.0
    return jsonify({
        "reference_scale_range": [DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX],
        "recommended_reference_upscale_factor": round(1.0 / mid_scale, 1),
        "ambiguous_gap_threshold": AMBIGUOUS_GAP,
        "gt_match_tolerance_px": GT_MATCH_TOLERANCE_PX,
    })


# ---------------------------------------------------------------------------
# Curated examples
# ---------------------------------------------------------------------------
@app.route("/api/examples")
def list_examples():
    return jsonify({"examples": examples_registry.list_examples()})


@app.route("/api/media/<kind>/<path:filename>")
def media(kind, filename):
    if kind not in ("reference", "search"):
        abort(404)
    directory = os.path.join(OUTPUTS_DIR, kind)
    # send_from_directory safely rejects path traversal
    return send_from_directory(directory, filename)


# ---------------------------------------------------------------------------
# Upload mode
# ---------------------------------------------------------------------------
def _cleanup_stale_uploads():
    now = time.time()
    try:
        for name in os.listdir(UPLOAD_DIR):
            p = os.path.join(UPLOAD_DIR, name)
            try:
                if now - os.path.getmtime(p) > UPLOAD_TOKEN_TTL_SEC:
                    os.remove(p)
            except OSError:
                pass
    except FileNotFoundError:
        pass


@app.route("/api/upload", methods=["POST"])
def upload():
    _cleanup_stale_uploads()

    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file' in request."}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty upload."}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Allowed: "
                     f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    token = uuid.uuid4().hex
    tmp_path = os.path.join(UPLOAD_DIR, f"{token}_raw{ext}")
    f.save(tmp_path)

    # Decode with Pillow rather than OpenCV for the initial validation pass.
    # Pillow tolerates a much wider range of real-world JPEG/PNG variants
    # (CMYK, palette/indexed PNGs, progressive JPEGs, embedded color
    # profiles, EXIF-rotated phone photos) than OpenCV's default decoder,
    # which is the main source of "could not decode" failures on
    # arbitrary user uploads. We then normalize to a canonical grayscale
    # PNG so the localization pipeline (which uses cv2.imread) always
    # receives a format it's known to handle -- this is a genuine
    # compatibility fix, not a cosmetic one.
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(tmp_path) as im:
            im = ImageOps.exif_transpose(im)  # respect phone-camera orientation
            im = im.convert("L")              # grayscale, matches pipeline's IMREAD_GRAYSCALE
            w, h = im.size
            if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM or w < 4 or h < 4:
                os.remove(tmp_path)
                return jsonify({"error": f"Image dimensions out of supported range: {w}x{h}px."}), 400
            dest_path = os.path.join(UPLOAD_DIR, f"{token}.png")
            im.save(dest_path, format="PNG")
    except Image.DecompressionBombError:
        # Pillow raises this for images whose pixel count exceeds its own
        # decompression-bomb safety threshold. Catch it explicitly so an
        # oversized upload becomes a clean 400 response instead of a raw 500.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({
            "error": "Image is too large to process (exceeds decompression-bomb safety limit)."
        }), 400
    except (UnidentifiedImageError, OSError, ValueError) as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": "File could not be decoded as an image."}), 400
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify({
        "token": token + ".png",
        "width": w,
        "height": h,
        "url": f"/api/uploads/{token}.png",
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
    }), 413


@app.route("/api/uploads/<path:filename>")
def serve_upload(filename):
    # filename must be exactly "<32-hex-token><ext>" -- reject anything else
    # to avoid serving arbitrary files.
    name, ext = os.path.splitext(filename)
    if len(name) != 32 or not all(c in "0123456789abcdef" for c in name):
        abort(404)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        abort(404)
    return send_from_directory(UPLOAD_DIR, filename)


def _resolve_upload_token(token):
    """token is '<hex><ext>' as returned by /api/upload. Returns absolute
    path inside UPLOAD_DIR, or None if it doesn't exist / is invalid."""
    if not token:
        return None
    name, ext = os.path.splitext(token)
    if len(name) != 32 or not all(c in "0123456789abcdef" for c in name):
        return None
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return None
    path = os.path.join(UPLOAD_DIR, token)
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Run localization (Server-Sent Events)
# ---------------------------------------------------------------------------
def _sse_format(event_dict):
    return f"data: {json.dumps(event_dict)}\n\n"


@app.route("/api/run/stream")
def run_stream():
    """
    Query params:
        mode=example       & example_id=<id from /api/examples>
        mode=upload         & reference_token=<token> & search_token=<token>
        topk (optional, default 5)
    A GET (not POST) endpoint so EventSource can be used directly from
    the browser without extra client libraries.
    """
    mode = request.args.get("mode", "example")
    topk = int(request.args.get("topk", 5))
    topk = max(1, min(topk, 15))

    # Always initialize this before the mode branches. The streaming
    # generator closes over it, so it must exist for curated examples too.
    self_reference_factor = None

    if mode == "example":
        example_id = request.args.get("example_id", "")
        ex = examples_registry.get_example(example_id)
        if ex is None:
            return jsonify({"error": f"Unknown example_id '{example_id}'."}), 400
        reference_path = ex["reference_path"]
        search_path = ex["search_path"]
        gt = ex["ground_truth"]
    elif mode == "upload":
        ref_token = request.args.get("reference_token", "")
        search_token = request.args.get("search_token", "")
        reference_path = _resolve_upload_token(ref_token)
        search_path = _resolve_upload_token(search_token)
        if reference_path is None or search_path is None:
            return jsonify({"error": "Invalid or expired upload token(s). Re-upload and try again."}), 400
        # Optional self-supplied ground truth: when the reference was
        # produced by cropping a region directly out of the search image
        # (the normal "Explore Your Own Images" flow), the crop's center
        # in the search image's pixel space IS the true target location
        # by construction -- not an estimate. If present, we can honestly
        # report prediction error the same way we do for curated examples.
        gt = None
        sgx = request.args.get("self_gt_x")
        sgy = request.args.get("self_gt_y")
        if sgx is not None and sgy is not None:
            try:
                gt = (float(sgx), float(sgy))
            except ValueError:
                gt = None

        # The browser knows the actual reference upscale factor after the
        # 1600px safety cap. Pass it through only for the self-crop flow so
        # that upload-mode NCC searches tightly around the true crop scale.
        self_reference_factor = None
        srf = request.args.get("self_reference_factor")
        if srf is not None:
            try:
                value = float(srf)
                if value > 0:
                    self_reference_factor = value
            except ValueError:
                self_reference_factor = None
    else:
        return jsonify({"error": f"Unknown mode '{mode}'."}), 400

    def generate():
        try:
            gt_source = "self_crop" if mode == "upload" else "annotation"
            for ev in run_localization(reference_path, search_path, gt=gt, topk=topk, gt_source=gt_source, reference_scale_factor=self_reference_factor):
                yield _sse_format(ev)
        except PipelineError as e:
            yield _sse_format({
                "stage": e.stage or "unknown",
                "status": "failed",
                "message": str(e),
                "timestamp": time.time(),
                "data": {},
            })
        except Exception as e:  # pragma: no cover - defensive backstop
            traceback.print_exc()

            yield _sse_format({
                "stage": "unknown",
                "status": "failed",
                "message": f"Internal error: {type(e).__name__}: {e}",
                "timestamp": time.time(),
                "data": {
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)