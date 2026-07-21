# Workflow-CI — Kriteria 3 Submission MSML

Re-training otomatis model klasifikasi *rice variety* menggunakan **MLflow Project** yang
dipantik oleh **GitHub Actions**.

## Tautan Docker Hub

```
https://hub.docker.com/r/hip2array/rice-variety-model
```

Image dibangun otomatis oleh workflow dengan `mlflow models build-docker`, lalu di-push dengan
dua tag: `latest` dan nomor run CI.

Menjalankan hasilnya:

```bash
docker pull hip2array/rice-variety-model:latest
docker run -p 5001:8080 hip2array/rice-variety-model:latest
```

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml          # workflow CI (GitHub Actions)
└── MLProject/
    ├── MLProject                     # manifest MLflow Project
    ├── conda.yaml                    # spesifikasi environment
    ├── modelling.py                  # skrip training
    ├── rice_preprocessing/   # dataset siap latih (train.csv, test.csv)
    └── artifacts/                    # keluaran training (model + metrics.json)
```

Catatan: kriteria menuliskan folder workflow sebagai `.workflow`. Path fungsional yang dikenali
GitHub Actions adalah `.github/workflows/`, jadi itu yang dipakai.

## Menjalankan secara lokal

```bash
pip install mlflow==2.19.0 scikit-learn==1.5.2 pandas==2.2.3 numpy==1.26.4
mlflow run MLProject --env-manager=local
```

Dengan parameter kustom:

```bash
mlflow run MLProject --env-manager=local -P n_estimators=200 -P max_depth=15
```

Hasil uji lokal (RandomForest, `n_estimators=100`, `max_depth=10`, `min_samples_split=5`):

| Metrik | Nilai |
|---|---|
| test_accuracy | 0.8860 |
| test_precision | 0.6875 |
| test_recall | 0.2973 |
| test_f1 | 0.4151 |
| train_accuracy | 0.9853 |

## Pemicu workflow

- `push` ke `main` yang menyentuh `MLProject/**`
- `pull_request` ke `main` (training saja, tanpa commit balik dan tanpa push image)
- `workflow_dispatch` manual, bisa mengatur `n_estimators` dan `max_depth`

## Secrets yang dibutuhkan

Diatur di **Settings → Secrets and variables → Actions**:

| Secret | Isi |
|---|---|
| `DOCKERHUB_USERNAME` | `hip2array` |
| `DOCKERHUB_TOKEN` | Access Token dari Docker Hub (**bukan** password akun) |

Tanpa kedua secret ini, tahap build dan push Docker dilewati otomatis; training dan penyimpanan
artefak tetap berjalan.
