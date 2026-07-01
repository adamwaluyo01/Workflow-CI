# Workflow-CI

Repository untuk Kriteria 3 - Membuat Workflow CI.

Workflow GitHub Actions menjalankan training model menggunakan MLflow Project melalui perintah:

```bash
mlflow run MLProject --env-manager=local
```

Struktur utama:

```text
.github/workflows/main.yml
MLProject/
├── MLProject
├── conda.yaml
├── requirements.txt
├── modelling.py
└── namadataset_preprocessing/
```
