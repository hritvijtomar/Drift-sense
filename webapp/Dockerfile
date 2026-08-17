# Drift-Sense web console -- single-container deployment.
# Build context MUST be the repository root (so localization/, outputs/,
# etc. are available), e.g.:
#   docker build -f webapp/Dockerfile -t drift-sense .
#   docker run -p 8000:8000 drift-sense
FROM python:3.11-slim

# opencv-python needs these system libs even in "headless" mode on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY webapp/requirements.txt ./webapp-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r webapp-requirements.txt

COPY . .

# outputs/reference/*.png and outputs/search/*.png are gitignored (kept out
# of the repo to stay lean -- see .gitignore). The webapp's curated examples
# reference specific files under outputs/ by id (e.g. dram_018, finfet_005),
# so regenerate the exact same deterministic set here at build time --
# generate_dataset.py is seeded, so this reproduces byte-for-byte the same
# images whether built locally or from a fresh GitHub clone.
RUN python3 generate_dataset.py --out_dir outputs --n_pairs 30 --seed 42

ENV PORT=8000
EXPOSE 8000

# 1 worker, 1 thread: this is a memory-constrained deployment (Render Free
# = 512MB). A single localization run on a large uploaded image can peak
# around ~250-350MB by itself (see webapp/README.md "Memory & the 512MB
# limit" for the measurements behind this number). Gunicorn threads/workers
# share -- or in the worker case, duplicate -- that budget: N threads or
# workers means up to N of these large-image runs could execute
# CONCURRENTLY, multiplying peak memory by N. Since this is a demo
# deployment (not a high-traffic service), we deliberately process one
# request at a time rather than risk an OOM kill under concurrent load.
# If you move to a larger instance, --threads can be raised safely (each
# thread only costs the memory of a request actually in flight, unlike
# --workers which duplicates the whole Python/OpenCV/Flask baseline).
CMD ["sh", "-c", "gunicorn --chdir webapp/backend --bind 0.0.0.0:${PORT} --workers 1 --threads 1 --timeout 180 app:app"]
