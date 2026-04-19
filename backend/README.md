# Backend

`backend` is the FastAPI application for ReviewOp. It exposes admin and user APIs, performs hybrid review inference, manages batch jobs, serves analytics and graph data, and loads the exported ProtoNet bundle when implicit-aspect inference is enabled.

## Responsibilities

- Authenticate admin and user requests.
- Accept user-submitted product reviews.
- Run explicit and implicit aspect/sentiment inference.
- Process CSV batch jobs asynchronously.
- Serve admin analytics, product statistics, exports, and knowledge graph data.
- Preserve review/prediction records needed for analytics and research demos.

## Code Map

| Path | Responsibility |
| --- | --- |
| `app.py` | FastAPI app assembly, router registration, middleware, and startup behavior. |
| `core/` | Configuration, database setup, bootstrap logic, and shared dependencies. |
| `routes/` | API route modules for admin, user portal, analytics, jobs, inference, graph, and exports. |
| `services/` | Business logic for inference, analytics, batch processing, products, reviews, and graph generation. |
| `models/` | Database models and request/response schemas where applicable. |
| `scripts/` | Operational or maintenance scripts. |
| `tests/` | Backend test coverage. |
| `requirements.txt` | Python dependency lock/input for local setup. |

## Local Startup

From the repository root, the easiest path is:

```powershell
.\run-project.ps1
.\run-services.ps1
```

Manual startup:

```powershell
.\backend\venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Runtime Configuration

Database settings are configured through backend configuration and environment variables. Do not hardcode credentials or commit secrets.

Important runtime switches:

| Variable | Purpose |
| --- | --- |
| `REVIEWOP_ENABLE_IMPLICIT` | Enables or disables ProtoNet implicit-aspect inference. |
| `REVIEWOP_TRUSTED_BUNDLE_ROOTS` | Adds trusted directories for loading ProtoNet bundles. |

If implicit inference is disabled or unavailable, the backend uses a disabled implicit client instead of failing at import time.

## API Areas

| Area | Description |
| --- | --- |
| Admin auth | Admin login/session flow used by protected admin routes. |
| User portal | User login, profile bootstrap, product browsing, review submission, editing, and soft deletion. |
| Inference | Single-review hybrid inference and CSV batch inference. |
| Jobs | Background batch job status and results. |
| Analytics | Dashboard metrics, product stats, exports, and aggregate review analysis. |
| Graph | Knowledge graph and relationship endpoints for admin exploration. |

Admin analytics, graph, inference, job, and export endpoints require admin authorization. User portal endpoints require user authentication where applicable.

## Batch Processing

CSV batch uploads are scheduled through the backend batch-job service instead of running synchronously inside the request handler. This keeps API requests responsive and makes status polling the preferred client interaction pattern.

## Review Deletion Semantics

User review deletion is soft-delete based. The user-facing review is hidden through `deleted_at`, while linked ML review and prediction records can remain available for analytics integrity.

## ProtoNet Integration

The backend expects the exported ProtoNet bundle at:

```text
protonet/metadata/model_bundle.pt
```

Bundle loading is restricted to trusted roots. Use `REVIEWOP_TRUSTED_BUNDLE_ROOTS` only for directories that contain trusted project-generated model bundles.

## Validation

Backend import check:

```powershell
.\backend\venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'backend'); import app; print('backend import ok')"
```

Run tests when changing backend behavior:

```powershell
.\backend\venv\Scripts\python.exe -m pytest backend\tests
```
