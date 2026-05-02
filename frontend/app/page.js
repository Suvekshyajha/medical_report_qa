"use client";
import { useEffect, useState } from "react";
import './page.css';
import Sidebar        from "../components/Sidebar";
import UploadZone     from "../components/UploadZone";
import ChatBox        from "../components/ChatBox";
import RelevanceChart from "../components/RelevanceChart";
import {
  getSessions,
  createSession,
  deleteSession,
  uploadPdf,
  sendMessage,
} from "../lib/api";

export default function Home() {
  const [sessions,    setSessions]    = useState([]);
  const [activeId,    setActiveId]    = useState(null);
  const [messages,    setMessages]    = useState([]);
  const [lastResult,  setLastResult]  = useState(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [fetchError,  setFetchError]  = useState("");

  const activeSession = sessions.find((s) => s.id === activeId) ?? null;

  useEffect(() => { loadSessions(); }, []);

  async function loadSessions() {
    try {
      const data = await getSessions();
      setSessions(data);
      if (data.length > 0 && !activeId) setActiveId(data[0].id);
    } catch {
      setFetchError("Cannot reach backend. Make sure the API server is running.");
    }
  }

  function handleSelectSession(id) {
    setActiveId(id);
    setMessages([]);
    setLastResult(null);
  }

  async function handleNewSession() {
    const session = await createSession("New Session");
    setSessions((prev) => [...prev, session]);
    handleSelectSession(session.id);
  }

  async function handleDeleteSession(id) {
    await deleteSession(id);
    const remaining = sessions.filter((s) => s.id !== id);
    setSessions(remaining);

    if (id === activeId) {
      setMessages([]);
      setLastResult(null);
      setActiveId(remaining.length > 0 ? remaining[0].id : null);
    }
  }

  async function handleUpload(file) {
    // Auto-create a session if none exists
    let sessionId = activeId;
    if (!sessionId) {
      const newSession = await createSession("New Session");
      setSessions((prev) => [...prev, newSession]);
      setActiveId(newSession.id);
      sessionId = newSession.id;
    }

    const result = await uploadPdf(sessionId, file);
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sessionId
          ? { ...s, name: result.filename, indexed_files: [...(s.indexed_files ?? []), result.filename] }
          : s
      )
    );
  }

  async function handleSend(text) {
    if (!activeId) return;
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setChatLoading(true);
    setLastResult(null);
    try {
      const data = await sendMessage(activeId, text);
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
      setLastResult(data);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeId
            ? { ...s, message_count: (s.message_count ?? 0) + 2 }
            : s
        )
      );
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "⚠️ Error: " + (e?.response?.data?.detail || "Failed to get answer.") },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <div className="page">

      <header className="page__header">
        <div className="page__header-icon">🩺</div>
        <div>
          <h1 className="page__header-title">Medical Report Intelligence</h1>
          <p className="page__header-subtitle">
            Advanced clinical analysis and document intelligence dashboard.
          </p>
        </div>
      </header>

      {fetchError && (
        <div className="page__error-banner">{fetchError}</div>
      )}

      <div className="page__body">
        <Sidebar
          sessions={sessions}
          activeId={activeId}
          onSelect={handleSelectSession}
          onNew={handleNewSession}
          onDelete={handleDeleteSession}
        />

        <main className="page__main">
          <UploadZone
            sessionName={activeSession?.name ?? ""}
            indexedFiles={activeSession?.indexed_files ?? []}
            onUpload={handleUpload}
            hasSession={!!activeSession}
          />
          <ChatBox
            messages={messages}
            onSend={handleSend}
            loading={chatLoading}
            disabled={!activeSession}
          />
          {lastResult && <RelevanceChart result={lastResult} />}
        </main>
      </div>
    </div>
  );
}