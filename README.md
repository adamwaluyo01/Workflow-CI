# Workflow-CI Adam Waluyo

Repository ini digunakan untuk Kriteria 3 submission **Membangun Sistem Machine Learning**.

Workflow CI menjalankan MLflow Project untuk melakukan re-training model Customer Churn secara otomatis saat push ke branch `main` atau saat workflow dijalankan manual.

## Struktur

```text
Workflow-CI
├── .github/workflows/main.yml
├── .workflow
└── MLProject
    ├── modelling.py
    ├── conda.yaml
    ├── MLProject
    ├── requirements.txt
    └── namadataset_preprocessing
```

## Cara run lokal

```bash
cd Workflow-CI
pip install -r MLProject/requirements.txt
mlflow run MLProject --env-manager=local
```

## Output

Workflow akan menghasilkan artefak MLflow dan mengunggahnya ke GitHub Actions Artifacts dengan nama:

```text
customer-churn-mlflow-artifacts
```
