"use client";
import { useRef, useState } from "react";
import './UploadZone.css';

export default function UploadZone({ sessionName, indexedFiles, onUpload, hasSession }) {
  const fileRef = useRef();
  const [status, setStatus] = useState("idle");
  const [errMsg, setErrMsg] = useState("");

  const handleFile = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setErrMsg("Only PDF files are accepted.");
      setStatus("error");
      return;
    }
    setStatus("uploading");
    setErrMsg("");
    try {
      await onUpload(file);
      setStatus("done");
    } catch (e) {
      setErrMsg(e?.response?.data?.detail || "Upload failed.");
      setStatus("error");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  };

  const loaded = indexedFiles.length > 0;

  return (
    <div>
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="upload__drop-zone"
      >
        <div className="upload__icon">🧬</div>
        <p className="upload__title">
          {!hasSession
            ? 'Upload a report to begin'
            : loaded
              ? '✅ Report loaded'
              : `Workspace: ${sessionName}`}
        </p>
        <p className="upload__subtitle">
          {!hasSession
            ? 'A session will be created automatically when you upload a PDF.'
            : loaded
              ? indexedFiles.join(', ')
              : 'Drop a medical report PDF below to update the clinical context.'}
        </p>
      </div>

      <div className="upload__row">
        <button
          onClick={() => fileRef.current.click()}
          disabled={status === 'uploading'}
          className="upload__btn"
        >
          ⬆ {status === 'uploading' ? 'Uploading…' : 'Upload'}
        </button>
        <span className="upload__hint">200 MB per file · PDF</span>

        {status === 'error' && (
          <span className="upload__error">{errMsg}</span>
        )}

        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>
    </div>
  );
}