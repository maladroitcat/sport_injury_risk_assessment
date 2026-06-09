# Sports Impact Injury Risk Assessment

Computer vision project for classifying short sports-impact videos into injury risk levels:

- `low`
- `moderate`
- `high`

The modeling work implements three complementary approaches:

1. Naive majority-class baseline
2. Strict classical computer vision model using HOG + motion-summary features
3. ResNet50 transfer-learning model with a small temporal classification head

## Setup

Create/activate the virtual environment, then install dependencies:

```bash
source env/bin/activate
pip install -r requirements.txt
```

The raw videos live in `data/cv_module_videos/` and metadata lives in `data/metadata.csv`.

## Modeling Workflow

Run these commands from the repository root.

### 1. Create shared folds

```bash
python scripts/create_folds.py
```

This writes reusable stratified fold allocations to `artifacts/splits/folds.csv`.

### 2. Train the naive baseline

```bash
python scripts/train_baseline.py
```

The baseline always predicts the majority class in each training fold, expected to be `moderate`.

### 3. Extract strict classical CV features

```bash
python scripts/extract_classical_features.py
```

This samples 16 frames per video, converts them to grayscale, extracts HOG features, computes motion-summary features, and saves one feature vector per video.

### 4. Train classical CV classifiers

```bash
python scripts/train_classical.py
```

This trains and compares:

- logistic regression
- linear SVM
- random forest

The selected model is based on cross-validated macro F1, with high-risk recall as the tie-breaker.

### 5. Extract ResNet50 frame embeddings

```bash
python scripts/extract_resnet_features.py
```

This uses pretrained ResNet50 as a frozen frame encoder for the deep learning model only.

### 6. Train the deep ResNet50 temporal head

```bash
python scripts/train_deep.py
```

This trains a small neural classifier over mean-pooled ResNet50 frame embeddings.

## Experiments

### Frame-count experiment

```bash
python scripts/run_frame_count_experiment.py
```

Compares classical HOG + motion features using 4, 8, 16, and 32 frames per video.

### Transformation experiment

```bash
python scripts/run_transformation_experiment.py
```

Compares plain HOG + motion preprocessing against CLAHE contrast-normalized preprocessing.

## Error Analysis

After training a model, export five mispredictions and representative frames:

```bash
python scripts/export_error_analysis.py \
  --predictions artifacts/models/classical/best_predictions.csv
```

Outputs are written under `artifacts/error_analysis/`.

## Local API + UI

Run the FastAPI model backend from this repository:

```bash
conda run -n cv-project uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The backend exposes:

- `GET /health`
- `GET /models`
- `POST /predict`

`POST /predict` accepts multipart form data with:

- `video`: `.mp4` or `.mov`
- `sport`: `hockey`, `basketball`, `soccer`, `football`, or `rugby`
- `impact_type`: `collision`, `object_hit`, `fall`, or `twist`
- `body_region`: `head_face`, `upper_body`, or `lower_body`
- `model_version`: optional model selection, defaults to `resnet50_mild_blur_4f`

Uploaded clips are limited to 250 MB by default. Override with `MAX_UPLOAD_BYTES` if needed.

Deployment model artifacts live under `model_artifacts/`. The default model is `resnet50_mild_blur_4f`; the selectable alternatives are `classical_hog_motion_svm_16f` and the `baseline_moderate` no-inference baseline.

Run the separate Next.js UI from `/Users/yiqian/src/injury-risk-ui`:

```bash
npm run dev
```

For local testing, set `/Users/yiqian/src/injury-risk-ui/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Notebooks

The `notebooks/` folder contains report-friendly exploratory views:

- `01_eda.ipynb`: class balance, metadata relationships, and sampled video frames
- `02_model_results.ipynb`: model comparison plots, confusion matrices, and transformation experiment results
- `03_error_analysis.ipynb`: five mispredicted videos with representative frame strips

Start Jupyter with:

```bash
jupyter notebook
```

## Generated Artifacts

Generated models, metrics, cached features, plots, and exported frames are written to `artifacts/`, which is ignored by git.
