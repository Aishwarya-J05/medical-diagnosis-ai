// frontend/src/App.jsx
import { useState, useEffect } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity, Upload, BarChart2, List,
  X, CheckCircle, AlertTriangle, Cpu, Database
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell
} from "recharts";
import { supabase } from "./supabaseClient";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "";

const CLASS_COLORS = { NORMAL: "#00b894", PNEUMONIA: "#ff6b6b" };

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("analyze");
  const [selectedHistory, setSelectedHistory] = useState(null);
  const [stats, setStats] = useState({ total: 0, pneumonia: 0, normal: 0, avgConfidence: 0, avgLatency: 0 });

  const fetchHistory = async () => {
    setHistoryLoading(true);
    try {
      const { data, error } = await supabase
        .from("diagnoses").select("*")
        .order("created_at", { ascending: false }).limit(50);
      if (error) throw error;
      setHistory(data || []);
      if (data && data.length > 0) {
        const pneumoniaCount = data.filter(d => d.prediction === "PNEUMONIA").length;
        setStats({
          total: data.length,
          pneumonia: pneumoniaCount,
          normal: data.length - pneumoniaCount,
          avgConfidence: data.reduce((s, d) => s + d.confidence, 0) / data.length,
          avgLatency: data.reduce((s, d) => s + (d.inference_time_ms || 0), 0) / data.length,
        });
      }
    } catch (err) {
      console.error(err);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    const loadHistory = async () => {
      await fetchHistory();
    };
    if (isMounted) loadHistory();
    return () => { isMounted = false; };
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f); setPreview(URL.createObjectURL(f));
    setResult(null); setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (!f) return;
    setFile(f); setPreview(URL.createObjectURL(f));
    setResult(null); setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await axios.post(`${API_URL}/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const data = response.data;
      setResult(data);
      await supabase.from("diagnoses").insert([{
        filename: file.name,
        prediction: data.prediction,
        confidence: data.confidence,
        normal_prob: data.probabilities.NORMAL,
        pneumonia_prob: data.probabilities.PNEUMONIA,
        heatmap_base64: data.heatmap_base64,
        inference_time_ms: data.inference_time_ms,
      }]);
      fetchHistory();
    } catch (err) {
      setError(err.response?.data?.detail || "Analysis failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  const navItems = [
    { id: "analyze", label: "Analyze", icon: Activity },
    { id: "history", label: "History", icon: List },
    { id: "stats", label: "Statistics", icon: BarChart2 },
  ];

  const chartData = [
    { name: "Normal", value: stats.normal, color: "#00b894" },
    { name: "Pneumonia", value: stats.pneumonia, color: "#ff6b6b" },
  ];

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon">
            <Cpu size={18} />
          </div>
          <div className="logo-text">
            <span className="logo-title">MedDiag AI</span>
            <span className="logo-sub">Radiology Suite</span>
          </div>
        </div>

        <span className="nav-section-label">Navigation</span>
        <nav className="sidebar-nav">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${activeTab === id ? "active" : ""}`}
              onClick={() => { setActiveTab(id); if (id === "history") fetchHistory(); }}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="api-status">
            <span className="status-dot" />
            API Connected
          </div>
        </div>
      </aside>

      {/* Content */}
      <div className="content">
        <AnimatePresence mode="wait">

          {/* ── Analyze ── */}
          {activeTab === "analyze" && (
            <motion.div className="page" key="analyze"
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.25 }}>

              <div className="page-header">
                <h1>Chest X-Ray Analysis</h1>
                <p>Upload an image to detect pathologies with explainable AI</p>
              </div>

              <div className="analyze-grid">
                {/* Upload */}
                <div className="card">
                  <p className="card-title">Image Input</p>
                  <div className="dropzone"
                    onDrop={handleDrop}
                    onDragOver={(e) => e.preventDefault()}
                    onClick={() => document.getElementById("fileInput").click()}>
                    {preview
                      ? <img src={preview} alt="preview" className="preview-img" />
                      : <div className="dropzone-placeholder">
                          <Upload size={30} />
                          <p>Drop X-Ray here or click to upload</p>
                          <p className="hint">Supports JPEG · PNG · DICOM</p>
                        </div>
                    }
                  </div>

                  <input id="fileInput" type="file"
                    accept="image/jpeg,image/png"
                    onChange={handleFileChange}
                    style={{ display: "none" }} />

                  {file && (
                    <div className="file-info">
                      <span>{file.name}</span>
                      <span>{(file.size / 1024).toFixed(1)} KB</span>
                    </div>
                  )}

                  <button className="analyze-btn" onClick={handleAnalyze} disabled={!file || loading}>
                    {loading
                      ? <span className="btn-loading"><span className="spinner" /> Processing...</span>
                      : "Run Diagnostic"}
                  </button>

                  {error && <div className="error-box">{error}</div>}
                </div>

                {/* Result */}
                <AnimatePresence>
                  {result && (
                    <motion.div className="card"
                      initial={{ opacity: 0, scale: 0.97 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.3 }}>
                      <p className="card-title">Diagnostic Report</p>

                      <div className="prediction-banner"
                        style={{ borderColor: CLASS_COLORS[result.prediction] + "66", color: CLASS_COLORS[result.prediction] }}>
                        <div className="pred-icon">
                          {result.prediction === "NORMAL"
                            ? <CheckCircle size={22} />
                            : <AlertTriangle size={22} />}
                        </div>
                        <div>
                          <div className="pred-label">{result.prediction}</div>
                          <div className="pred-conf">{(result.confidence * 100).toFixed(1)}% confidence</div>
                        </div>
                        <div className="pred-latency">{result.inference_time_ms.toFixed(0)}ms</div>
                      </div>

                      <div className="prob-section">
                        <h3>Class Distribution</h3>
                        {Object.entries(result.probabilities).map(([cls, prob]) => (
                          <div key={cls} className="prob-row">
                            <span className="prob-label">{cls}</span>
                            <div className="prob-bar-bg">
                              <motion.div className="prob-bar-fill"
                                initial={{ width: 0 }}
                                animate={{ width: `${prob * 100}%` }}
                                transition={{ duration: 0.8, ease: "easeOut" }}
                                style={{ backgroundColor: CLASS_COLORS[cls] }} />
                            </div>
                            <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>

                      <div className="divider" />

                      <div className="heatmap-section">
                        <h3>Grad-CAM Activation Map</h3>
                        <div className="heatmap-desc">
                          <div className="heatmap-legend">
                            <div className="legend-dot" style={{ background: "#ff4444" }} />
                            <span>High activation</span>
                          </div>
                          <div className="heatmap-legend">
                            <div className="legend-dot" style={{ background: "#4444ff" }} />
                            <span>Low activation</span>
                          </div>
                        </div>
                        <img src={`data:image/png;base64,${result.heatmap_base64}`}
                          alt="Grad-CAM" className="heatmap-img" />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}

          {/* ── History ── */}
          {activeTab === "history" && (
            <motion.div className="page" key="history"
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.25 }}>

              <div className="page-header">
                <h1>Diagnosis History</h1>
                <p>Complete audit trail of all analyses stored in Supabase</p>
              </div>

              {historyLoading ? (
                <div className="loading-state">Loading records...</div>
              ) : history.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state-icon"><Database size={22} /></div>
                  <p>No records found. Run a diagnostic to get started.</p>
                </div>
              ) : (
                <div className="card">
                  <div className="table-wrapper">
                    <table className="history-table">
                      <thead>
                        <tr>
                          <th>Filename</th>
                          <th>Result</th>
                          <th>Confidence</th>
                          <th>Normal</th>
                          <th>Pneumonia</th>
                          <th>Latency</th>
                          <th>Date</th>
                          <th></th>
                        </tr>
                      </thead>
                      <tbody>
                        {history.map((item) => (
                          <tr key={item.id}>
                            <td className="filename">{item.filename}</td>
                            <td>
                              <span className="badge" style={{
                                backgroundColor: CLASS_COLORS[item.prediction] + "18",
                                color: CLASS_COLORS[item.prediction],
                                border: `1px solid ${CLASS_COLORS[item.prediction]}44`,
                              }}>
                                {item.prediction}
                              </span>
                            </td>
                            <td>{(item.confidence * 100).toFixed(1)}%</td>
                            <td>{(item.normal_prob * 100).toFixed(1)}%</td>
                            <td>{(item.pneumonia_prob * 100).toFixed(1)}%</td>
                            <td className="date">{item.inference_time_ms?.toFixed(0)}ms</td>
                            <td className="date">{new Date(item.created_at).toLocaleDateString()}</td>
                            <td>
                              <button className="view-btn" onClick={() => setSelectedHistory(item)}>
                                View
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Modal */}
              <AnimatePresence>
                {selectedHistory && (
                  <motion.div className="modal-overlay"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    onClick={() => setSelectedHistory(null)}>
                    <motion.div className="modal"
                      initial={{ scale: 0.95, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0.95, opacity: 0 }}
                      onClick={(e) => e.stopPropagation()}>
                      <div className="modal-header">
                        <h3>{selectedHistory.filename}</h3>
                        <button className="modal-close" onClick={() => setSelectedHistory(null)}>
                          <X size={14} />
                        </button>
                      </div>
                      <div className="prediction-banner" style={{
                        borderColor: CLASS_COLORS[selectedHistory.prediction] + "66",
                        color: CLASS_COLORS[selectedHistory.prediction]
                      }}>
                        {selectedHistory.prediction === "NORMAL"
                          ? <CheckCircle size={18} />
                          : <AlertTriangle size={18} />}
                        <span className="pred-label">{selectedHistory.prediction}</span>
                        <span className="pred-conf" style={{ marginLeft: "0.5rem" }}>
                          {(selectedHistory.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <img
                        src={`data:image/png;base64,${selectedHistory.heatmap_base64}`}
                        alt="heatmap"
                        style={{ width: "100%", borderRadius: "10px", marginTop: "1rem" }}
                      />
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}

          {/* ── Stats ── */}
          {activeTab === "stats" && (
            <motion.div className="page" key="stats"
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.25 }}>

              <div className="page-header">
                <h1>Statistics</h1>
                <p>Aggregate performance metrics across all diagnoses</p>
              </div>

              <div className="stats-grid">
                {[
                  { label: "Total Analyses", value: stats.total, icon: Activity, color: "#00b894" },
                  { label: "Pneumonia Cases", value: stats.pneumonia, icon: AlertTriangle, color: "#ff6b6b" },
                  { label: "Normal Cases", value: stats.normal, icon: CheckCircle, color: "#00b894" },
                  { label: "Avg Confidence", value: `${(stats.avgConfidence * 100).toFixed(1)}%`, icon: BarChart2, color: "#00cec9" },
                ].map(({ label, value, icon: Icon, color }) => (
                  <motion.div key={label} className="stat-card"
                    whileHover={{ y: -3 }} transition={{ duration: 0.2 }}>
                    <div className="stat-icon"><Icon size={16} color={color} /></div>
                    <div className="stat-value" style={{ color }}>{value}</div>
                    <div className="stat-label">{label}</div>
                  </motion.div>
                ))}
              </div>

              {stats.total > 0 && (
                <div className="chart-card">
                  <p className="chart-title">Case Distribution</p>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={chartData} barSize={48}>
                      <XAxis dataKey="name" stroke="#3d6b63" tick={{ fill: "#7fa99e", fontSize: 12 }} />
                      <YAxis stroke="#3d6b63" tick={{ fill: "#7fa99e", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{
                          background: "#041214",
                          border: "1px solid rgba(0,184,148,0.2)",
                          borderRadius: "8px",
                          color: "#e8f5f3",
                          fontSize: "0.8rem"
                        }}
                      />
                      <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                        {chartData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} opacity={0.85} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>

                  <div className="divider" />

                  <div className="prob-row" style={{ marginTop: "0.5rem" }}>
                    <span className="prob-label">Pneumonia</span>
                    <div className="prob-bar-bg">
                      <motion.div className="prob-bar-fill"
                        initial={{ width: 0 }}
                        animate={{ width: stats.total > 0 ? `${(stats.pneumonia / stats.total) * 100}%` : "0%" }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        style={{ backgroundColor: "#ff6b6b" }} />
                    </div>
                    <span className="prob-value">
                      {stats.total > 0 ? ((stats.pneumonia / stats.total) * 100).toFixed(1) : 0}%
                    </span>
                  </div>
                  <div className="prob-row">
                    <span className="prob-label">Normal</span>
                    <div className="prob-bar-bg">
                      <motion.div className="prob-bar-fill"
                        initial={{ width: 0 }}
                        animate={{ width: stats.total > 0 ? `${(stats.normal / stats.total) * 100}%` : "0%" }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        style={{ backgroundColor: "#00b894" }} />
                    </div>
                    <span className="prob-value">
                      {stats.total > 0 ? ((stats.normal / stats.total) * 100).toFixed(1) : 0}%
                    </span>
                  </div>
                </div>
              )}
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  );
}