import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Database,
  FileText,
  GitCompare,
  Image,
  Play,
  RefreshCw,
  Settings,
  ShieldAlert,
  Upload,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const emptyResults = {
  metrics: [],
  cross_validation: [],
  per_class: [],
  artifacts: [],
  methods: [],
  unsupervised: null,
  metadata: null,
  feature_metadata: null,
  report: "",
  assets: {},
};

function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? value.toString() : value.toFixed(digits);
  const n = Number(value);
  if (!Number.isNaN(n) && value !== "") return Number.isInteger(n) ? n.toString() : n.toFixed(digits);
  return String(value);
}

function assetUrl(path) {
  if (!path) return null;
  return `${API_BASE}${path}`;
}

function DataTable({ rows, columns }) {
  if (!rows?.length) return <div className="empty">No rows</div>;
  const keys = columns || Object.keys(rows[0]);
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            {keys.map((key) => (
              <th key={key}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {keys.map((key) => (
                <td key={key}>{fmt(row[key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function bestBy(rows, key) {
  if (!rows?.length) return null;
  return [...rows].sort((a, b) => Number(b[key]) - Number(a[key]))[0];
}

function MetricBars({ rows, metric, label }) {
  if (!rows?.length) return <div className="empty">No model metrics</div>;
  const sorted = [...rows].sort((a, b) => Number(b[metric]) - Number(a[metric]));
  return (
    <div className="bars">
      {sorted.map((row) => {
        const value = Number(row[metric] || 0);
        return (
          <div className="barRow" key={`${metric}-${row.model}`}>
            <div className="barMeta">
              <span>{row.model}</span>
              <strong>{fmt(value)}</strong>
            </div>
            <div className="barTrack" aria-label={`${row.model} ${label}`}>
              <div className="barFill" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function MethodCards({ methods, metrics }) {
  if (!methods?.length) return null;
  const scores = Object.fromEntries((metrics || []).map((row) => [row.model, row]));
  return (
    <div className="methodGrid">
      {methods.map((method) => (
        <article className="methodCard" key={method.name}>
          <header>
            <strong>{method.name}</strong>
            <span>{method.family}</span>
          </header>
          <p>{method.why}</p>
          {scores[method.name] && <small>Macro F1: {fmt(scores[method.name].macro_f1)}</small>}
        </article>
      ))}
    </div>
  );
}

function Stat({ label, value, icon: Icon }) {
  return (
    <section className="stat">
      <div className="statIcon">
        <Icon size={18} />
      </div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </section>
  );
}

function App() {
  const [results, setResults] = useState(emptyResults);
  const [activeTab, setActiveTab] = useState("detect");
  const [loading, setLoading] = useState(false);
  const [detectionLoading, setDetectionLoading] = useState(false);
  const [error, setError] = useState("");
  const [runLog, setRunLog] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [detection, setDetection] = useState(null);
  const [detectionError, setDetectionError] = useState("");
  const [settings, setSettings] = useState({
    max_windows: 8000,
    limit_rows_per_file: 50000,
    limitRows: true,
    reuse_processed: false,
    hyperparameter_search: false,
  });

  const best = useMemo(() => {
    return bestBy(results.metrics, "macro_f1");
  }, [results.metrics]);

  const bestCv = useMemo(() => {
    return bestBy(results.cross_validation, "cv_macro_f1_mean");
  }, [results.cross_validation]);

  async function loadResults() {
    setError("");
    const response = await fetch(`${API_BASE}/api/results`);
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    setResults(await response.json());
  }

  async function runExperiment() {
    setLoading(true);
    setError("");
    setRunLog(null);
    try {
      const body = {
        max_windows: Number(settings.max_windows),
        limit_rows_per_file: settings.limitRows ? Number(settings.limit_rows_per_file) : null,
        reuse_processed: settings.reuse_processed,
        hyperparameter_search: settings.hyperparameter_search,
        holdout_tail_rows: 0,
      };
      const response = await fetch(`${API_BASE}/api/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail || `API returned ${response.status}`);
      setRunLog(payload.run);
      setResults(payload.results);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  async function detectUpload() {
    if (!selectedFile) {
      setDetectionError("Choose a CAN CSV file first.");
      return;
    }

    setDetectionLoading(true);
    setDetectionError("");
    setDetection(null);
    try {
      const form = new FormData();
      form.append("file", selectedFile);
      const response = await fetch(`${API_BASE}/api/detect`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail || `API returned ${response.status}`);
      setDetection(payload);
    } catch (err) {
      setDetectionError(err.message || String(err));
    } finally {
      setDetectionLoading(false);
    }
  }

  useEffect(() => {
    loadResults().catch((err) => setError(err.message || String(err)));
  }, []);

  const tabs = [
    { id: "detect", label: "Detect", icon: ShieldAlert },
    { id: "models", label: "Models", icon: GitCompare },
    { id: "visuals", label: "Visuals", icon: Image },
    { id: "unsupervised", label: "Unsupervised", icon: BarChart3 },
    { id: "report", label: "Report", icon: FileText },
  ];

  return (
    <main className="appShell">
      <aside className="sidePanel">
        <div className="brand">
          <Activity size={22} />
          <div>
            <h1>CAN AI</h1>
            <p>Detection Lab</p>
          </div>
        </div>

        <section className="controlGroup">
          <h2>
            <Settings size={17} />
            Experiment
          </h2>

          <label>
            <span>Windows</span>
            <input
              type="range"
              min="2000"
              max="80000"
              step="2000"
              value={settings.max_windows}
              onChange={(event) => setSettings({ ...settings, max_windows: event.target.value })}
            />
            <output>{Number(settings.max_windows).toLocaleString()}</output>
          </label>

          <label className="checkRow">
            <input
              type="checkbox"
              checked={settings.limitRows}
              onChange={(event) => setSettings({ ...settings, limitRows: event.target.checked })}
            />
            <span>Limit rows</span>
          </label>

          <label>
            <span>Rows per file</span>
            <input
              type="number"
              min="1000"
              step="1000"
              disabled={!settings.limitRows}
              value={settings.limit_rows_per_file}
              onChange={(event) => setSettings({ ...settings, limit_rows_per_file: event.target.value })}
            />
          </label>

          <label className="checkRow">
            <input
              type="checkbox"
              checked={settings.reuse_processed}
              onChange={(event) => setSettings({ ...settings, reuse_processed: event.target.checked })}
            />
            <span>Reuse features</span>
          </label>

          <label className="checkRow">
            <input
              type="checkbox"
              checked={settings.hyperparameter_search}
              onChange={(event) => setSettings({ ...settings, hyperparameter_search: event.target.checked })}
            />
            <span>RandomizedSearchCV</span>
          </label>

          <button className="primaryButton" onClick={runExperiment} disabled={loading}>
            {loading ? <RefreshCw className="spin" size={18} /> : <Play size={18} />}
            {loading ? "Running" : "Run"}
          </button>

          <button className="secondaryButton" onClick={() => loadResults().catch((err) => setError(err.message))}>
            <RefreshCw size={17} />
            Refresh
          </button>
        </section>
      </aside>

      <section className="content">
        <header className="topBar">
          <div>
            <h2>CAN Traffic Anomaly Detection</h2>
            <p>Upload a CAN log, detect suspicious windows, and justify the detector with model comparison.</p>
          </div>
          <div className="statusPill">
            {error ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
            {error ? "Needs attention" : "Ready"}
          </div>
        </header>

        {error && <div className="errorBox">{error}</div>}

        <section className="statsGrid">
          <Stat label="Best Model" value={best?.model || "-"} icon={GitCompare} />
          <Stat label="Macro F1" value={best ? fmt(best.macro_f1) : "-"} icon={BarChart3} />
          <Stat label="Accuracy" value={best ? fmt(best.accuracy) : "-"} icon={CheckCircle2} />
          <Stat
            label="Windows"
            value={results.feature_metadata?.rows?.toLocaleString?.() || "-"}
            icon={Database}
          />
        </section>

        <nav className="tabs" aria-label="Result tabs">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={activeTab === id ? "tab active" : "tab"}
              onClick={() => setActiveTab(id)}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </nav>

        {activeTab === "detect" && (
          <div className="sectionStack">
            <section className="detectHero">
              <div>
                <h3>Analyze a CAN CSV</h3>
                <p>
                  Upload a CAN message log with timestamp, CAN ID, length, and DLC byte columns. The app uses the best
                  trained detector to classify each window as Normal, DoS, Fuzzy, or Impersonation.
                </p>
              </div>
              <label className="fileDrop">
                <Upload size={22} />
                <span>{selectedFile ? selectedFile.name : "Choose CAN CSV"}</span>
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
                />
              </label>
              <button className="primaryButton detectButton" onClick={detectUpload} disabled={detectionLoading}>
                {detectionLoading ? <RefreshCw className="spin" size={18} /> : <ShieldAlert size={18} />}
                {detectionLoading ? "Analyzing" : "Detect unusual traffic"}
              </button>
            </section>

            {detectionError && <div className="errorBox">{detectionError}</div>}

            {detection && (
              <>
                <section className={detection.detection.hacked ? "resultBanner suspicious" : "resultBanner clean"}>
                  <div>
                    <span>{detection.filename}</span>
                    <strong>{detection.detection.hacked ? "Suspicious traffic detected" : "No suspicious traffic detected"}</strong>
                    <p>{detection.detection.status}</p>
                  </div>
                  <div>
                    <span>Top suspicious type</span>
                    <strong>{detection.detection.top_attack || "Normal"}</strong>
                  </div>
                </section>

                <section className="statsGrid">
                  <Stat label="Messages" value={detection.detection.message_count.toLocaleString()} icon={Database} />
                  <Stat label="Windows" value={detection.detection.window_count.toLocaleString()} icon={BarChart3} />
                  <Stat
                    label="Suspicious"
                    value={detection.detection.suspicious_window_count.toLocaleString()}
                    icon={ShieldAlert}
                  />
                  <Stat
                    label="Suspicious Ratio"
                    value={`${(Number(detection.detection.suspicious_ratio) * 100).toFixed(2)}%`}
                    icon={AlertTriangle}
                  />
                </section>

                <section className="panel">
                  <h3>Detected Class Distribution</h3>
                  <DataTable
                    rows={Object.entries(detection.detection.class_counts).map(([label, count]) => ({
                      label,
                      count,
                      ratio: detection.detection.class_ratios[label],
                    }))}
                  />
                </section>

                <section className="panel">
                  <h3>Suspicious Segments</h3>
                  <DataTable rows={detection.detection.segments} />
                </section>

                <section className="panel">
                  <h3>Window Predictions Preview</h3>
                  <DataTable rows={detection.detection.predictions} />
                </section>
              </>
            )}
          </div>
        )}

        {activeTab === "models" && (
          <div className="sectionStack">
            <section className="infoStrip">
              <Database size={18} />
              <p>
                <strong>Why only four files in data/ai_processed?</strong> That folder stores shared processed data:
                cleaned messages, metadata, window features, and feature metadata. Each method uses the same feature
                table, then writes its own model and evaluation artifacts under outputs/ai.
              </p>
            </section>

            <section className="comparisonHero">
              <div>
                <span>Best test model</span>
                <strong>{best?.model || "-"}</strong>
                <small>Macro F1 {best ? fmt(best.macro_f1) : "-"}</small>
              </div>
              <div>
                <span>Best cross-validation model</span>
                <strong>{bestCv?.model || "-"}</strong>
                <small>CV Macro F1 {bestCv ? fmt(bestCv.cv_macro_f1_mean) : "-"}</small>
              </div>
              <div>
                <span>Compared methods</span>
                <strong>{results.metrics?.length || 0}</strong>
                <small>Same features and same split</small>
              </div>
            </section>

            <section className="panel">
              <h3>Applied AI Methods</h3>
              <MethodCards methods={results.methods} metrics={results.metrics} />
            </section>

            <section className="panel">
              <h3>Macro F1 Comparison</h3>
              <MetricBars rows={results.metrics} metric="macro_f1" label="macro F1" />
            </section>

            <section className="panel">
              <h3>Model Comparison</h3>
              <DataTable rows={results.metrics} />
            </section>
            <section className="panel">
              <h3>K-Fold Cross Validation</h3>
              <DataTable rows={results.cross_validation} />
            </section>
            <section className="panel">
              <h3>Per-Class Metrics</h3>
              <DataTable rows={results.per_class} />
            </section>

            <section className="panel">
              <h3>Generated Files Per Method</h3>
              <DataTable rows={results.artifacts} />
            </section>
          </div>
        )}

        {activeTab === "visuals" && (
          <div className="visualGrid">
            <section className="panel">
              <h3>Confusion Matrix</h3>
              {results.assets?.confusion_matrix ? (
                <img src={assetUrl(results.assets.confusion_matrix)} alt="Confusion matrix" />
              ) : (
                <div className="empty">No image</div>
              )}
            </section>
            <section className="panel">
              <h3>PCA / K-Means</h3>
              {results.assets?.pca_clusters ? (
                <img src={assetUrl(results.assets.pca_clusters)} alt="PCA and K-Means clusters" />
              ) : (
                <div className="empty">No image</div>
              )}
            </section>
          </div>
        )}

        {activeTab === "unsupervised" && (
          <section className="panel">
            <h3>PCA + K-Means Metrics</h3>
            {results.unsupervised ? (
              <DataTable rows={[results.unsupervised]} />
            ) : (
              <div className="empty">No metrics</div>
            )}
          </section>
        )}

        {activeTab === "report" && (
          <section className="panel reportPanel">
            <h3>Project Report</h3>
            {results.assets?.report_docx && (
              <a className="reportDownload" href={assetUrl(results.assets.report_docx)}>
                <FileText size={16} />
                Open DOCX report
              </a>
            )}
            {results.report ? <ReactMarkdown>{results.report}</ReactMarkdown> : <div className="empty">No report</div>}
          </section>
        )}

        {runLog && (
          <details className="runDetails">
            <summary>Run summary</summary>
            <pre>{JSON.stringify(runLog, null, 2)}</pre>
          </details>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
