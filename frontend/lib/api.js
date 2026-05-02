import axios from "axios";

const BASE = "http://localhost:8000";

const http = axios.create({ baseURL: BASE });

// ── Sessions ──────────────────────────────────────────────────────────────────
export const getSessions   = ()           => http.get("/sessions").then(r => r.data);
export const createSession = (name)       => http.post("/sessions", { name }).then(r => r.data);
export const deleteSession = (id)         => http.delete(`/sessions/${id}`).then(r => r.data);
export const getSession    = (id)         => http.get(`/sessions/${id}`).then(r => r.data);

// ── PDF Upload ────────────────────────────────────────────────────────────────
export const uploadPdf = (sessionId, file) => {
  const form = new FormData();
  form.append("file", file);
  return http
    .post(`/sessions/${sessionId}/upload`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then(r => r.data);
};

// ── Chat ──────────────────────────────────────────────────────────────────────
export const sendMessage = (sessionId, message) =>
  http.post("/chat", { session_id: sessionId, message }).then(r => r.data);

// ── Last result ───────────────────────────────────────────────────────────────
export const getLastResult = (sessionId) =>
  http.get(`/sessions/${sessionId}/results`).then(r => r.data);
