# Session: 2026-03-16 — Infrastructure Decisions

## Decisions made

### Orchestration: Airflow on a cheap GCP VM
- GCP e2-small (~$15/month) running Airflow with LocalExecutor
- Learning Airflow is an explicit goal — Cloud Composer ($300+/month) hides too many internals, and serverless alternatives (Cloud Workflows, Cloud Functions) skip Airflow entirely
- Seasonal project: can spin the VM up for Feb-March, shut down after

### Data warehouse: BigQuery
- Serverless — no server to manage, just storage + SQL queries
- Accessed from Airflow tasks via `google-cloud-bigquery` Python client

### DAG structure
Three tasks with dependencies:
```
extract >> retrain >> predict
```
- **Extract**: Daily download (Kaggle + ESPN)
- **Retrain**: Update Elo ratings with new game results
- **Predict**: Generate tournament win probabilities (submission CSV)

### Alternatives considered
| Option | Cost | Why not |
|--------|------|---------|
| Cloud Composer | ~$300+/month | Overkill, abstracts away Airflow internals |
| Cloud Scheduler + Cloud Functions | ~$5/month | No Airflow learning, fine for simple jobs |
| Cloud Scheduler + Cloud Run Jobs + Workflows | ~$5/month | Lightweight DAG runner but not Airflow |

## Next up
- Set up GCP VM + Airflow
- Write the DAG (`extract >> retrain >> predict`)
- Connect pipeline to BigQuery
