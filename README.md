# SALIGP Bloom Duplicate Detection

SALIGP Bloom is a document duplicate-detection project with a Python/FastAPI backend and a React frontend. The current application lets users upload documents or CSV data, converts raw files into SALIGP similarity features, runs pairwise duplicate predictions, and displays/export results in the web UI.

The active production model path is an Improved Genetic Programming (IGP) model persisted at `saligp/outputs/saligp_igp_model.pkl` and loaded by `saligp.api.app`. Some older research/training modules still contain baseline or experimental code, including RandomForest comparisons, but RandomForest is not the deployed prediction model for the API.

## Current Capabilities

- Upload two or more documents and compare them pairwise for duplicate detection.
- Upload CSV/TSV files containing either SALIGP feature rows or text-pair rows.
- Extract text from `txt`, `md`, `html`, `json`, `csv`, `tsv`, `docx`, and extractable-text `pdf` files.
- Generate normalized SALIGP features:
  - `filename_similarity`
  - `content_similarity`
  - `metadata_similarity`
  - `size_similarity`
  - `tfidf_similarity`
  - `embedding_similarity`
  - `sha256_match`
  - `overall_similarity`
- Serve predictions through FastAPI.
- Use a React/Vite frontend for uploads, result filtering, sorting, CSV export, and report generation.
- Store supporting outputs and trained artifacts under `saligp/outputs`.

## How The App Works

```text
Uploaded files
    |
    v
DocumentProcessor
    extracts text and file metadata
    |
    v
TextFeatureExtractor
    creates pairwise SALIGP feature rows
    |
    v
SALIGPClassifier
    loads/runs the persisted IGP model
    |
    v
FastAPI response
    duplicate flag, confidence, scores, metadata
    |
    v
React frontend
    dashboard, predictions table/cards, exports, report
```

For `N` uploaded documents, the document upload endpoint compares every pair, so the number of comparisons is:

```text
N * (N - 1) / 2
```

For example, 100 files produce 4,950 pair predictions. The feature extractor now precomputes reusable per-document text features and computes TF-IDF once per upload batch, which keeps the 100-file path much faster than recomputing everything for each pair.

## Repository Layout

```text
.
|-- main.py                         # Root wrapper for running the full pipeline script
|-- requirements.txt                # Backend dependencies
|-- render.yaml                     # Render deployment config for the FastAPI API
|-- frontend/                       # React + Vite UI
|-- saligp/
|   |-- api/                        # FastAPI app/server code
|   |-- text_processing/            # Document extraction and feature generation
|   |-- pipeline/                   # Prediction classifiers and pipeline integration
|   |-- genetic_programming/        # DEAP IGP training/loading code
|   |-- bloom_filter/               # Bloom filter support code
|   |-- role_hierarchy/             # Ownership/access tracking support code
|   |-- clustering/                 # Geometric analysis support code
|   |-- active_learning/            # Active learning support code
|   |-- evaluation/                 # Evaluation utilities
|   |-- validation/                 # Dataset validation utilities
|   |-- visualizations/             # Plot/report generation helpers
|   |-- outputs/                    # Generated artifacts and model files
|   `-- tests/                      # Backend tests
`-- saligp-100-inputfiles/          # Local sample documents for batch testing
```

## Backend Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run The API

From the project root:

```bash
uvicorn saligp.api.app:app --host 127.0.0.1 --port 8000
```

The API expects a non-empty model artifact at:

```text
saligp/outputs/saligp_igp_model.pkl
```

You can override that path with:

```bash
set SALIGP_MODEL_PATH=D:\path\to\saligp_igp_model.pkl
```

On macOS/Linux:

```bash
export SALIGP_MODEL_PATH=/path/to/saligp_igp_model.pkl
```

## API Endpoints

### Health

```http
GET /health
```

Returns:

```json
{"status": "ok"}
```

### Predict From Feature Row

```http
POST /predict
```

Body:

```json
{
  "pair_id": 1,
  "features": {
    "filename_similarity": 0.8,
    "content_similarity": 0.75,
    "metadata_similarity": 0.5,
    "size_similarity": 0.9,
    "tfidf_similarity": 0.7,
    "embedding_similarity": 0.72,
    "sha256_match": 0,
    "overall_similarity": 0.67
  }
}
```

### Predict From Two Texts

```http
POST /predict-text
```

Body:

```json
{
  "pair_id": 1,
  "left_text": "First document text",
  "right_text": "Second document text",
  "left_filename": "left.txt",
  "right_filename": "right.txt"
}
```

### Batch Prediction

```http
POST /predict-batch
```

Accepts:

- Rows with the eight SALIGP feature columns.
- Legacy text feature columns, which are mapped into the current SALIGP feature schema.
- Text-pair rows using columns such as `left_text`/`right_text`, `text_a`/`text_b`, `document_a`/`document_b`, or `source_text`/`target_text`.

### Upload Documents Or CSV

```http
POST /upload
```

Accepted input:

- Two or more document files for pairwise comparison.
- One CSV/TSV file containing feature rows or text-pair rows.

The response includes `total_pairs`, `duplicates_found`, and formatted result rows for the frontend.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server usually runs at:

```text
http://localhost:5173
```

Useful scripts:

```bash
npm run dev
npm run build
npm run preview
```

## Frontend Features

- Dashboard with pipeline/system summary information.
- Predictions page with:
  - drag-and-drop document upload
  - result summary cards
  - top-level `Generate Report` and `Export CSV` actions
  - duplicate/unique filtering
  - confidence and uncertainty sorting
  - per-pair result cards
- Analytics page with supporting charts and summaries.

Some dashboard/analytics values are static or derived from existing output artifacts. Treat them as project summaries unless they are wired to live API data.

## Running The Full Pipeline

The root script delegates to `saligp/main.py`:

```bash
python main.py
```

That script runs the research/training workflow: validation, clustering, active learning support steps, IGP training, Bloom filter support, role hierarchy setup, evaluation, and visualizations.

This pipeline is separate from the normal API serving path. For day-to-day app use, run the API and frontend.

## Model Notes

The deployed API path uses:

- `saligp/api/app.py`
- `saligp/api/server.py`
- `saligp/pipeline/saligp_classifier.py`
- `saligp/genetic_programming/gp_trainer.py`
- `saligp/outputs/saligp_igp_model.pkl`

The IGP model is a DEAP-based symbolic tree over the eight SALIGP features. Its prediction score is thresholded at `0.5`.

RandomForest appears in parts of the repository as:

- baseline comparison/evaluation support
- older simplified GP fallback code in `gp_trainer_simple.py`
- cluster-specific experimental integration code

Those references should not be read as the current production prediction model.

## Batch Performance

The app still performs all-pairs comparison for uploaded documents. That is accurate for deduplication, but it scales quadratically.

Approximate pair counts:

| Files | Pairs |
| ---: | ---: |
| 10 | 45 |
| 50 | 1,225 |
| 100 | 4,950 |
| 200 | 19,900 |

For larger batches, the next likely improvement is a blocking/pre-filter stage that avoids comparing obviously unrelated documents.

## Testing And Verification

Backend tests live under:

```text
saligp/tests/
```

Run them with:

```bash
python -m pytest saligp/tests
```

If `pytest` is not installed:

```bash
pip install pytest
python -m pytest saligp/tests
```

Frontend build verification:

```bash
cd frontend
npm run build
```

## Deployment

`render.yaml` is configured to deploy the API service on Render:

```text
uvicorn saligp.api.app:app --host 0.0.0.0 --port $PORT
```

The deployment must include or generate the model artifact referenced by `SALIGP_MODEL_PATH`.

## Current Limitations

- Document uploads compare every pair, so very large batches can still take time.
- PDF extraction depends on extractable text and `pypdf` support.
- The API does not currently implement authentication around the role hierarchy support code.
- Some research modules and historical outputs remain in the repo for context, but not every generated chart or metric is part of the deployed inference path.
- Metrics in existing output files should be interpreted as results from the project dataset/workflow, not as a guarantee of production performance on unseen document collections.

## Author

Karthik V

## Status

Active project. Last README refresh: June 12, 2026.
