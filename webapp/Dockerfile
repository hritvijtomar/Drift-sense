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

# 2 workers keep memory bounded; each localization run is CPU-bound and
# takes a few seconds, so avoid oversubscribing on small deployment tiers.
CMD ["sh", "-c", "gunicorn --chdir webapp/backend -w 2 --threads 4 -b 0.0.0.0:${PORT} --timeout 120 app:app"]
