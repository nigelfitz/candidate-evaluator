# Candidate Evaluator - Local Development Setup

## Installing Dependencies

### For Local Development (Windows with SQLite):
```bash
pip install -r requirements-local.txt
```

This excludes `psycopg2-binary` and `gunicorn` which are only needed for production PostgreSQL and deployment.

### For Production (Railway with PostgreSQL):
Railway uses `requirements.txt` automatically, which includes all production dependencies.

## Why Two Requirements Files?

- **requirements.txt**: Full production dependencies including PostgreSQL driver
- **requirements-local.txt**: Local development dependencies (SQLite only, no PostgreSQL)

This prevents installation errors on Windows where psycopg2-binary tries to compile from source.
