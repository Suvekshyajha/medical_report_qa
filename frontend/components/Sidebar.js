"use client";
import './Sidebar.css';

export default function Sidebar({ sessions, activeId, onSelect, onNew, onDelete }) {
  return (
    <aside className="sidebar">

      <p className="sidebar__label">Navigation</p>

      <button onClick={onNew} className="sidebar__btn-new">
        + New Session
      </button>

      <p className="sidebar__history-label">Session History</p>

      <div className="sidebar__session-list">
        {sessions.length === 0 && (
          <p className="sidebar__empty">No sessions yet.</p>
        )}

        {sessions.map((s) => (
          <div key={s.id} className="sidebar__session-card">
            <button
              onClick={() => onSelect(s.id)}
              className={`sidebar__session-btn${activeId === s.id ? ' sidebar__session-btn--active' : ''}`}
            >
              <span className="sidebar__session-dot" />
              <span className="sidebar__session-name">{s.name}</span>
              <span className="sidebar__session-meta">
                {s.created} · {s.message_count} msgs
              </span>
            </button>

            <button
              onClick={() => onDelete(s.id)}
              title="Delete session"
              className="sidebar__btn-delete"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}