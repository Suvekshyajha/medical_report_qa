"use client";
import { useEffect, useRef, useState } from "react";
import './ChatBox.css';

export default function ChatBox({ messages, onSend, loading }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    onSend(text);
  };

  return (
    <div>
      <p className="chat__label">Consultation Chat</p>

      <div className="chat__messages">
        {messages.length === 0 ? (
          <p className="chat__empty">Ready for analysis. Ask a clinical question.</p>
        ) : (
          messages.map((m, i) => (
            <div
              key={i}
              className={`chat__message${m.role === "user" ? " chat__message--user" : " chat__message--assistant"}`}
            >
              {m.role === "assistant" && (
                <span className="chat__assistant-tag">Assistant</span>
              )}
              {m.content}
            </div>
          ))
        )}

        {loading && (
          <div className="chat__typing">
            <span className="chat__typing-dot" />
            <span className="chat__typing-dot" />
            <span className="chat__typing-dot" />
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat__input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
          placeholder="Ask about symptoms, labs, or findings…"
          disabled={loading}
          className="chat__input"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="chat__btn-send"
        >
          Send
        </button>
      </div>
    </div>
  );
}