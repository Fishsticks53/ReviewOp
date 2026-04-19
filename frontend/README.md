# Frontend

`frontend` is the Vite React application for ReviewOp. It provides the admin dashboard, analytics and graph exploration pages, authentication flows, product browsing, and user review workflows.

## Responsibilities

- Render admin analytics, product metrics, graph views, exports, and batch-job status.
- Render user-facing product and review flows.
- Call backend APIs through shared request helpers.
- Attach authentication tokens for protected admin and user requests.
- Surface partial-load warnings instead of silently hiding backend failures.

## Code Map

| Path | Responsibility |
| --- | --- |
| `src/api/request.js` | Shared request helper, token attachment, error handling, and file downloads. |
| `src/api/client.js` | Barrel export for API modules used by pages. |
| `src/api/user.js` | User-facing auth, product, and review API helpers. |
| `src/api/analytics.js` | Admin analytics and export API helpers. |
| `src/api/runtime.js` | Runtime/admin API helpers. |
| `src/admin/` | Admin portal pages, hooks, dashboard components, graph views, and job UI. |
| `src/pages/` | User-facing pages and route-level views. |
| `src/components/` | Shared UI components. |
| `scripts/build.mjs` | Production build wrapper used by `npm run build`. |
| `vite.config.js` | Vite configuration. |

## Local Startup

Install dependencies from the repository root:

```powershell
.\run-project.ps1
```

Start backend and frontend together:

```powershell
.\run-services.ps1
```

Manual frontend startup:

```powershell
npm --prefix frontend run dev
```

The Vite dev server usually runs at:

```text
http://127.0.0.1:5173
```

## Build

```powershell
npm --prefix frontend run build
```

Preview a production build:

```powershell
npm --prefix frontend run preview
```

## API Layer

Use the shared API modules instead of calling `fetch` directly from pages:

- `request()` centralizes base URL handling, JSON parsing, and auth token attachment.
- `downloadFile()` attaches the admin bearer token for protected exports.
- `client.js` re-exports user, runtime, and analytics helpers so existing page imports stay stable.

When adding new API calls, place them in the closest domain module and re-export them from `client.js` if route components need the barrel import.

## Admin Portal

The admin portal is split between layout and data orchestration:

- `src/admin/pages/AdminPortal.jsx` owns the page shell and rendering structure.
- `src/admin/pages/useAdminPortal.js` owns admin portal state loading, warnings, and actions.

Keep new admin data-loading behavior in the hook unless it is purely presentational.

## User Portal

User-facing pages rely on backend APIs for:

- login and profile bootstrap,
- product listings and product detail,
- aspect-filtered review lists,
- review submission and editing,
- direct review lookup for reply/edit flows.

Review deletion is handled by the backend as a soft delete; the frontend should treat deleted reviews as unavailable rather than assuming linked analytics records were removed.

## Validation

Run the production build after frontend changes:

```powershell
npm --prefix frontend run build
```

If a page depends on backend data, also verify the backend is running at `http://127.0.0.1:8000`.
