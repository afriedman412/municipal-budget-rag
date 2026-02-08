# web_monitor.py
"""
Web-based pipeline monitor.

Run with: python -m pipeline.web_monitor
Or: uvicorn pipeline.web_monitor:app --host 0.0.0.0 --port 8000

Reads from:
- pipeline_metrics.json (real-time metrics from running pipeline)
- pipeline_state.db (job counts and failure details)
"""

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .config import PipelineConfig
from .metrics_writer import PipelineMetrics, read_metrics
from .state import StateDB

app = FastAPI(title="Pipeline Monitor")

# Paths - configurable via environment
METRICS_PATH = Path("pipeline_metrics.json")
STATE_DB_PATH = Path("pipeline_state.db")


def get_state_db() -> StateDB:
    """Get StateDB instance."""
    return StateDB(STATE_DB_PATH)


@app.get("/api/metrics")
def api_metrics() -> JSONResponse:
    """Get current pipeline metrics."""
    metrics = read_metrics(METRICS_PATH)

    if metrics is None:
        return JSONResponse({
            "status": "no_data",
            "message": "No metrics file found. Is the pipeline running?"
        })

    # Calculate derived values
    elapsed = time.time() - metrics.started_at if metrics.started_at else 0
    since_update = time.time() - metrics.last_update_at if metrics.last_update_at else 999

    extract_rate = metrics.extracted_chunks / elapsed if elapsed > 0 else 0
    embed_rate = metrics.embedded_chunks / elapsed if elapsed > 0 else 0

    # Determine status
    if since_update > 30:
        status = "stale"
    elif metrics.phase == "stopped":
        status = "stopped"
    elif metrics.producer_done and metrics.queue_depth == 0:
        status = "done"
    elif metrics.errors > 0:
        status = "running_with_errors"
    else:
        status = "running"

    return JSONResponse({
        "status": status,
        "elapsed_seconds": round(elapsed, 1),
        "since_update_seconds": round(since_update, 1),
        "phase": metrics.phase,
        "current_file": metrics.current_file,
        "current_files": metrics.current_files,
        "extracted_docs": metrics.extracted_docs,
        "extracted_chunks": metrics.extracted_chunks,
        "extract_rate": round(extract_rate, 1),
        "embedded_docs": metrics.embedded_docs,
        "embedded_chunks": metrics.embedded_chunks,
        "embed_rate": round(embed_rate, 1),
        "batches_started": metrics.batches_started,
        "batches_done": metrics.batches_done,
        "queue_depth": metrics.queue_depth,
        "producer_done": metrics.producer_done,
        "errors": metrics.errors,
        "download_count": metrics.download_count,
        "download_total": metrics.download_total,
        "extract_done": metrics.extract_done,
        "extract_total": metrics.extract_total,
    })


@app.get("/api/jobs")
def api_jobs() -> JSONResponse:
    """Get job counts from SQLite."""
    try:
        state = get_state_db()
        stats = state.get_stats()
        return JSONResponse({
            "status": "ok",
            **stats
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })


@app.get("/api/failures")
def api_failures() -> JSONResponse:
    """Get recent failures."""
    try:
        state = get_state_db()
        failures = state.get_failed(limit=20)
        summary = state.get_failure_summary()
        return JSONResponse({
            "status": "ok",
            "failures": [
                {
                    "s3_key": f.s3_key,
                    "stage": f.stage_failed,
                    "error": f.error_message[:200] if f.error_message else None,
                    "attempts": f.attempts,
                }
                for f in failures
            ],
            "summary": summary,
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Monitor</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }
        h1 {
            color: #00d9ff;
            margin: 0 0 20px 0;
            font-size: 24px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #0f3460;
        }
        .card h2 {
            margin: 0 0 15px 0;
            font-size: 14px;
            text-transform: uppercase;
            color: #888;
            letter-spacing: 1px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
        }
        .metric:last-child { border-bottom: none; }
        .metric .label { color: #aaa; }
        .metric .value {
            font-weight: bold;
            font-variant-numeric: tabular-nums;
        }
        .status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status.running { background: #00aa55; color: white; }
        .status.running_with_errors { background: #ff9900; color: black; }
        .status.done { background: #00d9ff; color: black; }
        .status.stale { background: #ff4444; color: white; }
        .status.stopped { background: #666; color: white; }
        .status.no_data { background: #333; color: #888; }
        .current-file {
            font-family: monospace;
            font-size: 12px;
            color: #00d9ff;
            word-break: break-all;
            margin-top: 5px;
        }
        .failures {
            max-height: 300px;
            overflow-y: auto;
        }
        .failure {
            background: #1a1a2e;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 12px;
        }
        .failure .filename {
            color: #ff9900;
            font-weight: bold;
        }
        .failure .error {
            color: #888;
            margin-top: 5px;
            word-break: break-all;
        }
        .rate { color: #00d9ff; }
        #last-update {
            position: fixed;
            bottom: 10px;
            right: 20px;
            font-size: 11px;
            color: #666;
        }
    </style>
</head>
<body>
    <h1>📊 Pipeline Monitor</h1>

    <div class="grid">
        <div class="card">
            <h2>Status</h2>
            <div class="metric">
                <span class="label">State</span>
                <span id="status" class="status no_data">Loading...</span>
            </div>
            <div class="metric">
                <span class="label">Phase</span>
                <span id="phase" class="value">-</span>
            </div>
            <div class="metric">
                <span class="label">Elapsed</span>
                <span id="elapsed" class="value">-</span>
            </div>
            <div class="metric">
                <span class="label">ETA</span>
                <span id="eta" class="value">-</span>
            </div>
            <div class="metric">
                <span class="label">Current File</span>
                <span class="value"></span>
            </div>
            <div id="current-file" class="current-file">-</div>
        </div>

        <div class="card">
            <h2>Extraction</h2>
            <div class="metric">
                <span class="label">Documents</span>
                <span id="extracted-docs" class="value">0</span>
            </div>
            <div class="metric">
                <span class="label">Chunks</span>
                <span id="extracted-chunks" class="value">0</span>
            </div>
            <div class="metric">
                <span class="label">Rate</span>
                <span id="extract-rate" class="value rate">0 chunks/s</span>
            </div>
            <div class="metric">
                <span class="label">Batch Progress</span>
                <span id="extract-progress" class="value">-</span>
            </div>
        </div>

        <div class="card">
            <h2>Embedding</h2>
            <div class="metric">
                <span class="label">Documents</span>
                <span id="embedded-docs" class="value">0</span>
            </div>
            <div class="metric">
                <span class="label">Chunks</span>
                <span id="embedded-chunks" class="value">0</span>
            </div>
            <div class="metric">
                <span class="label">Rate</span>
                <span id="embed-rate" class="value rate">0 chunks/s</span>
            </div>
            <div class="metric">
                <span class="label">Queue Depth</span>
                <span id="queue-depth" class="value">0</span>
            </div>
        </div>

        <div class="card">
            <h2>Jobs (SQLite)</h2>
            <div class="metric">
                <span class="label">Pending</span>
                <span id="jobs-pending" class="value">-</span>
            </div>
            <div class="metric">
                <span class="label">Done</span>
                <span id="jobs-done" class="value">-</span>
            </div>
            <div class="metric">
                <span class="label">Failed</span>
                <span id="jobs-failed" class="value">-</span>
            </div>
            <div class="metric">
                <span class="label">Total</span>
                <span id="jobs-total" class="value">-</span>
            </div>
        </div>

        <div class="card" style="grid-column: span 2;">
            <h2>Recent Failures</h2>
            <div id="failures" class="failures">
                <div style="color: #666;">Loading...</div>
            </div>
        </div>
    </div>

    <div id="last-update">Last update: -</div>

    <script>
        // Track historical data for real-time rate calculation
        let history = [];
        const HISTORY_WINDOW = 10; // seconds to average over

        function formatElapsed(seconds) {
            if (seconds < 60) return seconds.toFixed(0) + 's';
            if (seconds < 3600) return (seconds / 60).toFixed(1) + 'm';
            return (seconds / 3600).toFixed(1) + 'h';
        }

        function calculateRealtimeRate(history, field) {
            if (history.length < 2) return null;
            const newest = history[history.length - 1];
            const oldest = history[0];
            const timeDelta = (newest.timestamp - oldest.timestamp) / 1000;
            if (timeDelta < 1) return null;
            const valueDelta = newest[field] - oldest[field];
            return valueDelta / timeDelta;
        }

        async function updateMetrics() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();

                // Track history for real-time rates
                const now = Date.now();
                history.push({
                    timestamp: now,
                    extracted_chunks: data.extracted_chunks,
                    embedded_chunks: data.embedded_chunks,
                    extracted_docs: data.extracted_docs,
                    embedded_docs: data.embedded_docs,
                });
                // Keep only last N seconds of history
                while (history.length > 1 && (now - history[0].timestamp) > HISTORY_WINDOW * 1000) {
                    history.shift();
                }

                // Calculate real-time rates
                const realtimeExtractRate = calculateRealtimeRate(history, 'extracted_chunks');
                const realtimeEmbedRate = calculateRealtimeRate(history, 'embedded_chunks');
                const realtimeDocRate = calculateRealtimeRate(history, 'embedded_docs');

                // Status
                const statusEl = document.getElementById('status');
                statusEl.textContent = data.status.replace(/_/g, ' ');
                statusEl.className = 'status ' + data.status;

                // Phase
                let phase = data.phase || 'idle';
                if (data.phase === 'downloading' && data.download_total > 0) {
                    phase = `downloading (${data.download_count}/${data.download_total})`;
                } else if (data.phase === 'extracting' && data.extract_total > 0) {
                    phase = `extracting (${data.extract_done}/${data.extract_total})`;
                }
                document.getElementById('phase').textContent = phase;

                // Elapsed
                document.getElementById('elapsed').textContent = formatElapsed(data.elapsed_seconds);

                // Current file(s)
                const files = data.current_files?.length > 0
                    ? data.current_files.join('\\n')
                    : (data.current_file || '-');
                document.getElementById('current-file').textContent = files;

                // Extraction
                document.getElementById('extracted-docs').textContent = data.extracted_docs.toLocaleString();
                document.getElementById('extracted-chunks').textContent = data.extracted_chunks.toLocaleString();
                // Show real-time rate if available, otherwise average
                const extractRateText = realtimeExtractRate !== null
                    ? `${realtimeExtractRate.toFixed(1)} chunks/s`
                    : `${data.extract_rate} chunks/s (avg)`;
                document.getElementById('extract-rate').textContent = extractRateText;
                document.getElementById('extract-progress').textContent =
                    data.extract_total > 0 ? `${data.extract_done}/${data.extract_total}` : '-';

                // Embedding
                document.getElementById('embedded-docs').textContent = data.embedded_docs.toLocaleString();
                document.getElementById('embedded-chunks').textContent = data.embedded_chunks.toLocaleString();
                // Show real-time rate + docs/min
                let embedRateText = realtimeEmbedRate !== null
                    ? `${realtimeEmbedRate.toFixed(1)} chunks/s`
                    : `${data.embed_rate} chunks/s (avg)`;
                if (realtimeDocRate !== null && realtimeDocRate > 0) {
                    embedRateText += ` (${(realtimeDocRate * 60).toFixed(1)} docs/min)`;
                }
                document.getElementById('embed-rate').textContent = embedRateText;
                document.getElementById('queue-depth').textContent = data.queue_depth.toLocaleString();

                // Update ETA
                updateETA(realtimeDocRate);

            } catch (e) {
                console.error('Failed to fetch metrics:', e);
            }
        }

        // Store jobs data globally for ETA calculation
        let jobsData = {};

        async function updateJobs() {
            try {
                const res = await fetch('/api/jobs');
                const data = await res.json();

                if (data.status === 'ok') {
                    jobsData = data;
                    document.getElementById('jobs-pending').textContent = (data.pending || 0).toLocaleString();
                    document.getElementById('jobs-done').textContent = (data.done || 0).toLocaleString();
                    document.getElementById('jobs-failed').textContent = (data.failed || 0).toLocaleString();
                    document.getElementById('jobs-total').textContent = (data.total || 0).toLocaleString();
                }
            } catch (e) {
                console.error('Failed to fetch jobs:', e);
            }
        }

        function updateETA(realtimeDocRate) {
            const etaEl = document.getElementById('eta');
            const pending = jobsData.pending || 0;
            const extracting = jobsData.extracting || 0;
            const extracted = jobsData.extracted || 0;
            const embedding = jobsData.embedding || 0;
            const remaining = pending + extracting + extracted + embedding;

            if (remaining === 0) {
                etaEl.textContent = 'Done!';
                return;
            }

            if (realtimeDocRate === null || realtimeDocRate <= 0) {
                etaEl.textContent = `${remaining} remaining`;
                return;
            }

            const secondsRemaining = remaining / realtimeDocRate;
            if (secondsRemaining < 60) {
                etaEl.textContent = `~${secondsRemaining.toFixed(0)}s (${remaining} left)`;
            } else if (secondsRemaining < 3600) {
                etaEl.textContent = `~${(secondsRemaining / 60).toFixed(0)}m (${remaining} left)`;
            } else {
                etaEl.textContent = `~${(secondsRemaining / 3600).toFixed(1)}h (${remaining} left)`;
            }
        }

        async function updateFailures() {
            try {
                const res = await fetch('/api/failures');
                const data = await res.json();

                const container = document.getElementById('failures');
                if (data.status === 'ok' && data.failures?.length > 0) {
                    container.innerHTML = data.failures.map(f => `
                        <div class="failure">
                            <div class="filename">${f.s3_key.split('/').pop()}</div>
                            <div>Stage: ${f.stage || 'unknown'} | Attempts: ${f.attempts}</div>
                            ${f.error ? `<div class="error">${f.error}</div>` : ''}
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<div style="color: #666;">No failures</div>';
                }
            } catch (e) {
                console.error('Failed to fetch failures:', e);
            }
        }

        function updateTimestamp() {
            document.getElementById('last-update').textContent =
                'Last update: ' + new Date().toLocaleTimeString();
        }

        // Initial load - get jobs first so ETA works
        updateJobs().then(() => {
            updateMetrics();
            updateFailures();
            updateTimestamp();
        });

        // Poll every second for metrics, every 3s for jobs, every 10s for failures
        setInterval(() => { updateMetrics(); updateTimestamp(); }, 1000);
        setInterval(updateJobs, 3000);
        setInterval(updateFailures, 10000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the dashboard HTML."""
    return DASHBOARD_HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
