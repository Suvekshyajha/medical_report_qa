"use client";
import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import './RelevanceChart.css';

function scoreColor(score) {
  if (score > 0.8) return "#2ecc71";
  if (score > 0.6) return "#f39c12";
  return "#e74c3c";
}

function ScoreBadge({ score }) {
  const mod =
    score > 0.8 ? "score-badge--high"
    : score > 0.6 ? "score-badge--medium"
    : "score-badge--low";
  return (
    <span className={`score-badge ${mod}`}>
      {(score * 100).toFixed(0)}%
    </span>
  );
}

export default function RelevanceChart({ result }) {
  const [tab, setTab] = useState("chart");
  const [openIdx, setOpenIdx] = useState(null);

  if (!result) return null;

  const chartData = result.labels.map((label, i) => ({
    label,
    score: parseFloat(result.scores[i].toFixed(3)),
  }));

  return (
    <div className="relevance">

      <div className="relevance__tabs">
        {["chart", "sources"].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`relevance__tab${tab === t ? " relevance__tab--active" : " relevance__tab--inactive"}`}
          >
            {t === "chart" ? "📊 Relevance Scores" : "📄 Source Chunks"}
          </button>
        ))}
      </div>

      {tab === "chart" && (
        <div className="relevance__chart-panel">
          <p className="relevance__chart-label">Chunk Relevance Scores</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} margin={{ top: 0, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f4f8" />
              <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <Tooltip
                formatter={(v) => [`${(v * 100).toFixed(1)}%`, "Relevance"]}
                contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12 }}
              />
              <Bar dataKey="score" radius={[6, 6, 0, 0]}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={scoreColor(entry.score)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {tab === "sources" && (
        <div className="relevance__sources">
          {result.sources.map((src, i) => (
            <div key={i} className="relevance__chunk">
              <button
                onClick={() => setOpenIdx(openIdx === i ? null : i)}
                className="relevance__chunk-header"
              >
                <span className="relevance__chunk-title">
                  {result.labels[i]}
                  <ScoreBadge score={result.scores[i]} />
                </span>
                <span className="relevance__chunk-toggle">
                  {openIdx === i ? "−" : "+"}
                </span>
              </button>

              {openIdx === i && (
                <div className="relevance__chunk-body">
                  <pre className="relevance__chunk-text">{src}</pre>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}