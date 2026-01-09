# Forge Agent Frontend

A simple, operator-focused UI for the Forge Agent API.

## Principles

- **Frontend is dumb** - No reasoning, no retries, no decision-making
- **API is single source of truth** - Frontend displays exactly what the API returns
- **Readability > cleverness** - This is an operator UI
- **Every step must be inspectable** - Every failure must be visible

## Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000` and proxies API requests to `http://localhost:8000`.

## Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts          # Thin API client (no retries, no transformation)
│   ├── pages/
│   │   ├── RunPage.tsx        # Create new run
│   │   ├── RunsListPage.tsx  # Browse historical runs
│   │   └── RunDetailPage.tsx # Inspect single run
│   ├── components/
│   │   ├── PlanViewer.tsx     # Display plan steps
│   │   ├── ExecutionViewer.tsx # Display execution results
│   │   ├── DiagnosticsViewer.tsx # Display planner diagnostics
│   │   └── JsonBlock.tsx      # Pretty-print JSON with copy
│   ├── types/
│   │   └── api.ts             # TypeScript types (mirror API schemas)
│   ├── App.tsx                # Main app with routing
│   └── main.tsx               # Entry point
```

## Features

- **Create Run**: Submit goals and view plan + execution results
- **Browse Runs**: List historical runs with pagination
- **Inspect Run**: View complete run details including plan, diagnostics, and execution
- **No Auto-Refresh**: Manual refresh only
- **No Polling**: No background updates
- **Explicit Failures**: All failures shown clearly, not hidden

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Notes

- Frontend never makes decisions - it only displays what the API returns
- Execution failures are shown as normal results (not UI errors)
- Empty plans are clearly labeled
- All data is inspectable via JSON blocks

