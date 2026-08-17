# Drift-Sense &mdash; Interactive Web Console

A browser-based front end for the existing Drift-Sense localization
pipeline (`localization/template_matching.py` +
`localization/feature_matcher.py`). A visitor with no Python
environment and no knowledge of the source code can open one URL,
pick a real inspection scenario (or upload their own images), click
**Run Drift-Sense**, and watch the actual classical-CV pipeline
execute: NCC candidate generation, ORB+RANSAC verification, ranking,
and final selection, streamed live from the backend.

**Nothing in this app duplicates the localization algorithm.** The
web service imports and calls the exact same functions the CLI
(`localize.py`) uses. See "Architecture" below and
`webapp/backend/verify_parity.py`, which asserts the two produce
numerically identical output.

---

## 1. Architecture

```
localize.py  (CLI, UNCHANGED)
     |
     v
localization/inference.py :: localize()   <-- single source of truth,
     |                                          unchanged
     v
localization/template_matching.py  (multiscale_ncc_candidates)
localization/feature_matcher.py    (orb_verify, rank_candidates,
                                     is_ambiguous, select_final_candidate)
     ^
     |  (imported directly, not copied)
     |
webapp/backend/localization_service.py
     |  calls the SAME functions above, in the SAME order, with the
     |  SAME parameters -- and yields a structured event after each
     |  stage instead of returning once at the end.
     v
webapp/backend/app.py  (Flask)
     |  GET /api/run/stream  -> Server-Sent Events, one JSON event
     |                          per pipeline stage
     v
webapp/frontend/  (vanilla HTML/CSS/JS, no build step)
     EventSource consumes the stream and updates:
       - the pipeline tracker
       - the live technical/explanation feed
       - the candidate overlay on the search image canvas
       - the candidate ranking table
       - the final result + failure-classification panel
```

`localization/inference.py::localize()` and
`webapp/backend/localization_service.py::run_localization()` are two
different *callers* of the same underlying functions -- there is one
algorithm implementation, reused by both the CLI and the web service.

### Why this architecture

The PS/backend team's algorithm lives in exactly two files
(`template_matching.py`, `feature_matcher.py`). Those files were not
touched. `localization_service.py` is purely an orchestration/observability
layer: it calls `multiscale_ncc_candidates()` once (identical to the
CLI), then calls `orb_verify()` once per candidate *in the same loop
order `rank_candidates()` uses internally* (confirmed by
`verify_parity.py`), so the frontend can show verification happening
candidate-by-candidate instead of only seeing the final ranked list.

---

## 2. Event schema

Server-Sent Events on `GET /api/run/stream`. Each `data:` line is one
JSON object:

```json
{
  "stage": "candidate_generation",
  "status": "running",
  "message": "Human-readable status line",
  "timestamp": 1737000000.123,
  "data": { "...stage-specific fields..." }
}
```

`stage` is one of: `load`, `template_extraction`, `candidate_generation`,
`verification`, `ranking`, `decision`, `result`.
`status` is one of: `running`, `complete`, `failed`.

Stage-specific `data` payloads:

| stage | status | data |
|---|---|---|
| `load` | complete | `reference_shape`, `search_shape` (w/h in px) |
| `template_extraction` | complete | `template_w`, `template_h` |
| `candidate_generation` | complete | `candidate_count`, `candidates[]` (x, y, w, h, tl_x, tl_y, ncc_score, scale, angle_deg) |
| `verification` | running (×N) | `candidate_index`, `candidate` (adds orb_inliers, orb_total_matches, orb_ratio, final_score) |
| `ranking` | complete | `ranking[]` (candidates sorted by final_score, each with `rank`) |
| `decision` | complete | `selected`, `backend_ambiguous_flag`, `ambiguous_gap_threshold` |
| `result` | complete/failed | `result` (x, y, confidence, ambiguous, runtime_sec, candidates_considered), `diagnosis` (null when no ground truth is available) |

`diagnosis` is present for curated examples (`gt_source: "annotation"`)
and for upload-mode runs where the user cropped their reference
directly out of the search image (`gt_source: "self_crop"` -- see
section 3a). It is `null` only when no ground truth exists at all.

This schema is intentionally generic (`stage`/`status`/`message`/`data`)
so a future pipeline stage (e.g. a learned descriptor stage replacing
ORB) can emit new `data` fields without changing the frontend's event
handling contract.

---

## 3. Curated demo examples

Registered in `webapp/backend/examples.py`, built entirely from real
files under `outputs/annotations/`, `outputs/reference/`,
`outputs/search/`. Chosen by inspecting `evaluation/results.json`:

| id | category | why chosen |
|---|---|---|
| `finfet_005` | success | correct, **not** flagged ambiguous by the backend -- cleanest possible win |
| `finfet_009` | success | correct, but backend's ambiguity flag is `true` -- demonstrates the flag isn't "wrong" |
| `dram_002` | success | periodic DRAM array, correctly disambiguated despite repeat-pitch aliasing |
| `dram_018` | candidate-generation miss | ground truth never entered the top-5 NCC candidates (matches the supplied DRAM diagnosis exactly: error=400.0px) |
| `dram_020` | ranking/selection failure | ground truth WAS candidate rank 4, lost by a 0.215 margin (matches the supplied diagnosis exactly) |

Add a new curated example by adding a 4-tuple
`(annotation_id, label, note, expected_category)` to `CURATED` in
`webapp/backend/examples.py`. `annotation_id` must match a file in
`outputs/annotations/<id>.json`.

---

## 3a. Upload mode & the scale convention

`localization/template_matching.py::multiscale_ncc_candidates` sweeps
reference scales in a narrow range (`DEFAULT_SCALE_MIN`/`MAX`, default
0.07-0.15 -- see `/api/config`). This is not arbitrary: it encodes the
assumption that the **reference image is a ~10x zoomed-in depiction**
of a feature that occupies a much smaller area in the search image
(this matches the supplied dataset's own generation convention --
every `outputs/annotations/*.json` has `"scale_factor": 10`).

This matters for "Explore Your Own Images": if you simply crop a
region directly out of the search image at 1:1 scale and feed it back
in as the reference, the algorithm's scale sweep will never find it
(a same-scale crop is outside its search range by construction -- this
was verified empirically, not assumed). So the upload flow:

1. Lets the user upload **one** image and drag a box over the pattern
   they want located elsewhere in it.
2. Client-side (`app.js`), upscales that crop by
   `recommended_reference_upscale_factor` from `/api/config` (bicubic,
   via `<canvas>`) before uploading it as the reference. This makes the
   crop's pixel content match what the pipeline expects a "reference
   chip" to look like -- no new information is fabricated, the crop is
   just resampled to the scale the algorithm was designed around.
3. Because the crop was cut directly from the search image, its center
   in the search image's pixel space **is** the true target location,
   not an estimate -- so accuracy (error vs. ground truth, and the same
   4-category failure classification used for curated examples) can be
   shown honestly for user-uploaded images too. This is passed to the
   backend as `self_gt_x`/`self_gt_y` on `/api/run/stream`, and the
   resulting `diagnosis.gt_source` is `"self_crop"` (vs `"annotation"`
   for curated examples) so the UI can label it correctly.

Verified end-to-end: cropping a known region from
`outputs/search/dram_002_search.png`, upscaling 9.1x, and running it
through `/api/run/stream` recovers the crop's exact source location
(0.0px error, confidence 0.86, not flagged ambiguous).

---

## 4. Failure-analysis classification

Ground truth is only used **after** the real prediction has already
been computed -- it never feeds back into the algorithm. For curated
examples, `localization_service.classify_failure()` compares the
ranked candidate list against ground truth using the exact convention
from `evaluation/diagnose_dram.py` (the script that produced the
supplied DRAM diagnosis): a candidate is "the true one" if it's within
15px of ground truth.

Four possible outcomes, never collapsed into one generic status:

1. **`success`** &mdash; the selected candidate is within tolerance of ground truth.
2. **`candidate_generation_miss`** &mdash; ground truth is not within tolerance of *any* retained candidate. The failure is upstream of ranking.
3. **`ranking_selection_failure`** &mdash; ground truth *was* a retained candidate, but the winning candidate's final score beat it by more than the backend's own ambiguity gap (0.08, from `feature_matcher.is_ambiguous`'s default). Not a near tie.
4. **`true_ambiguity`** &mdash; ground truth was a retained candidate and the score margin to the winner is *smaller* than 0.08 -- a genuine statistical tie under the backend's own rule.

Verified against the supplied `DRAM_Failure_Diagnosis.md`: running the
service on pairs 018/020/022/024/028 reproduces the exact category and
margin numbers in that document (see "Testing performed" in
`CLAUDE_HANDOFF.md`).

---

## 5. Running locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt -r webapp/requirements.txt

python3 webapp/backend/app.py
# -> Running on http://0.0.0.0:5050
```

Open `http://localhost:5050` in a browser. No separate frontend build
step -- `webapp/frontend/` is served directly by Flask.

Environment variables (all optional):

| var | default | purpose |
|---|---|---|
| `PORT` | `5050` | HTTP port |
| `FLASK_DEBUG` | `0` | set to `1` for Flask's auto-reload/debugger during development |

### Verifying the install

```bash
# 1. Confirm the CLI is unaffected:
python3 localize.py --reference outputs/reference/finfet_005_ref.png \
                     --search outputs/search/finfet_005_search.png

# 2. Confirm the web service matches the CLI exactly:
python3 webapp/backend/verify_parity.py
```

---

## 6. Deployment

The app is a single Flask process serving both the API and the static
frontend -- no separate frontend deploy, no database, no external
services.

### Current production deployment: Render, native Python runtime

The live deployment (`https://drift-sense-g88m.onrender.com` at time of
writing) uses Render's **native Python runtime**, not the Dockerfile.
This matters because Render's native runtime installs from the
**repository-root `requirements.txt`**, not `webapp/requirements.txt` --
both files are kept in sync on purpose (see the comments in each), but
if you ever change one, change both.

Deployment config is now committed as code in `render.yaml` (a Render
"Blueprint") instead of living only in the dashboard's Start Command
field, so it survives in Git history and a fresh service can be
recreated from the repo alone:

- **Build command**: `pip install -r requirements.txt`
- **Start command**: `gunicorn --chdir webapp/backend --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 180 app:app`
- **Health check path**: `/api/health`

To deploy/redeploy: in the Render dashboard, **New +** -> **Blueprint**,
point it at this repo. If a service already exists under the same
name, Render's Settings -> Blueprint tab will offer to sync this file
onto it.

`--workers 1 --threads 1` is deliberate, not a placeholder -- see
"Memory & the 512MB limit" below before changing it.

### Alternative: Docker

`webapp/Dockerfile` is also maintained and works, for platforms that
prefer/require a container image (Fly.io, Google Cloud Run, AWS App
Runner, Railway, etc.):

```bash
# from the repository root (build context matters -- must include
# localization/, outputs/, webapp/)
docker build -f webapp/Dockerfile -t drift-sense .
docker run -p 8000:8000 drift-sense
```

Unlike the Render-native path, the Docker image regenerates the full
30-pair dataset at build time (`generate_dataset.py`, seeded and
deterministic) rather than relying on the 5 curated pairs being
committed to Git -- either approach produces byte-identical curated
images, so this is purely a build-time-vs-git-size tradeoff.

### CORS / networking

The frontend is served from the same origin as the API, so no CORS
configuration is required for normal use. `app.py` still sets
permissive CORS headers (`Access-Control-Allow-Origin: *`) in case you
choose to split the frontend onto a different host/CDN later.

### Storage

Uploaded images are written to a temp directory
(`tempfile.gettempdir()/driftsense_uploads`) with a random token
filename and are cleaned up on a 1-hour TTL on the next upload
request. On most container platforms this directory is ephemeral,
which is the desired behavior here (no user data is meant to persist).

---

## 6a. Memory & the 512MB limit

Render's Free tier caps a service at 512MB RSS; exceeding it gets the
process killed ("Ran out of memory... while running your code" in
Render's event log), which surfaces to the browser as a mid-stream 502
on `/api/run/stream`.

**Root cause, found by profiling, not guessing:** `multiscale_ncc_candidates`
(`localization/template_matching.py`) tracks, per pixel of the search
image, which of the 45 (scale x angle) combinations currently has the
best NCC score there. The original implementation stored this as a
Python tuple `(scale, angle, w, h)` in an `HxW`, `dtype=object` NumPy
array, written via a **Python-level `for y, x in zip(*np.where(mask))`
loop** -- for a search image of `H*W` pixels, that's up to `H*W`
individual Python tuple allocations, repeated up to 45 times (once per
scale/angle combination), since on early iterations most of the image
still "wins" against the initial `-1` sentinel score.

Measured directly (`resource.getrusage`, not estimated) on a synthetic
pair matching the real production failure's reported dimensions
(reference 7994x5810, search 3848x2160 = 8.3 megapixels):

| | peak RSS | time |
|---|---|---|
| Original (Python-loop + object array) | **931 MB** | 31.3s |
| Vectorized fix (below) | **343 MB** | 14.9s |

**The fix**: replaced the per-pixel Python loop and `dtype=object` array
with a small integer "winner index" map (`dtype=int32`) updated via a
single vectorized boolean-mask assignment per (scale, angle) --
`sub_winner[mask] = combo_idx` -- plus a short Python list of the 45
`(scale, angle, w, h)` combos looked up by index only for the final
top-K peaks (at most `topk`, not `H*W`, tuple constructions). This is
the **same algorithm** -- same score comparisons, same non-max
suppression, same tie-break -- just without per-pixel Python object
churn. Verified bit-for-bit identical against the original
implementation on **all 30 dataset pairs** before merging (every
candidate's x, y, score, scale, angle, w, h matched exactly); see
`localization/template_matching.py`'s docstring for the full
before/after comparison methodology.

**Defense in depth beyond the fix**: `webapp/backend/app.py` allows
uploads up to 8000px per side (64 megapixels) -- 8x larger than the
already-fixed 8.3MP case above, which would still risk exceeding
512MB even after vectorization. `localization_service.py` adds a
`MAX_SEARCH_PIXELS = 6_000_000` working-resolution cap: if the
uploaded search image exceeds it, **both** the reference and search
images are downscaled by the same factor (preserving the algorithm's
assumed reference:search scale relationship -- see section 3a) before
running the pipeline, and every coordinate in every SSE event is
scaled back to the ORIGINAL image's pixel space before being sent to
the browser, so the API/frontend contract never changes and displayed
coordinates are always correct. Verified end-to-end with a planted,
precisely-known target in a synthetic 24-megapixel image: the pipeline
correctly reported the exact original-image coordinates (0.0px error)
at 260MB peak RSS. The absolute worst case the upload endpoint allows
(8000x8000, 64MP) was also tested and peaked at 242MB.

Every curated example (1000x1000 = 1 megapixel) is far under this cap
and takes the untouched `scale_back=1.0` code path, so none of this
changes curated-example behavior -- confirmed via `verify_parity.py`
and by re-running the DRAM diagnosis reproduction (section 4) after
these changes; results were unchanged.

**Gunicorn workers/threads**: production runs `--workers 1 --threads 1`.
This is deliberate: N workers or threads means up to N large-image
localization runs could execute *concurrently* within the memory
budget, multiplying peak RSS by N. A worker process duplicates the
entire Python/OpenCV/Flask baseline; a thread only costs the memory of
the request actually in flight, so if you move to a larger instance,
raise `--threads` before `--workers`.

**Observability**: `localization_service.py` logs one line at the start
and end of every run (image sizes, whether downscaling engaged, runtime,
peak RSS, result summary) via a `driftsense.localization` logger to
stderr, which Render/Gunicorn capture into the service's log stream
automatically -- no per-candidate/per-scale spam, just enough to
diagnose a future slow or memory-heavy request without needing to
reproduce it blind.

### If 512MB genuinely isn't enough later

The fixes above make the current classical CV pipeline fit comfortably
within 512MB for any input the upload endpoint accepts. If a future ML
model (mentioned as a possible next step) needs meaningfully more
memory than this, that's a different, larger budget question than
what was fixed here -- the honest guidance is to size the instance to
the model's actual measured footprint (the same `resource.getrusage`
technique used above works for that too) rather than guessing.

---

## 7. Extending

**Add a new pipeline stage** (e.g. a future learned-descriptor stage
replacing ORB): give it a `stage` key, emit `running`/`complete`
events with whatever `data` fields make sense, and add a row to the
`STAGES` array in `webapp/frontend/static/app.js` plus (optionally) a
case in the `explain()` function. The tracker, feed, and SSE plumbing
require no other changes -- they're generic over stage names.

**Add a new curated example:** add a tuple to `CURATED` in
`webapp/backend/examples.py` (see section 3).

**Change ranking/verification behavior:** edit
`localization/feature_matcher.py` directly (the single source of
truth) -- `localization_service.py` will pick up the change
automatically since it imports the real functions and even reads
`ncc_weight`/`orb_weight`/`gap_threshold` from their default
parameter values rather than hardcoding them.
