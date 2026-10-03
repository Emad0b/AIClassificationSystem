import { useState } from "react";

const MODALITIES = [
  {
    id: "image",
    title: "Image Classification",
    endpoint: "/predict/image",
    accept: "image/*",
    hint: "EuroSAT land-use imagery: forest, urban, or water",
  },
  {
    id: "audio",
    title: "Audio Classification",
    endpoint: "/predict/audio",
    accept: "audio/*",
    hint: "Bird calls, urban sounds, sirens, traffic",
  },
];

function formatConfidence(value) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function App() {
  const [activeModality, setActiveModality] = useState("image");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const modality = MODALITIES.find((item) => item.id === activeModality);

  function handleFileChange(event) {
    const selected = event.target.files?.[0];
    setFile(selected ?? null);
    setResult(null);
    setError("");

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (selected && activeModality === "image") {
      setPreviewUrl(URL.createObjectURL(selected));
    } else {
      setPreviewUrl("");
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file || !modality) {
      setError("Please choose a file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(modality.endpoint, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed");
      }

      setResult(payload);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">Real-Time Multimodal AI</p>
        <h1>AI Classification System</h1>
        <p className="subtitle">
          Upload an image or audio clip and get a live prediction with confidence scores
          from a production-style FastAPI backend.
        </p>
      </header>

      <main className="panel">
        <div className="tabs">
          {MODALITIES.map((item) => (
            <button
              key={item.id}
              type="button"
              className={item.id === activeModality ? "tab active" : "tab"}
              onClick={() => {
                setActiveModality(item.id);
                setFile(null);
                setResult(null);
                setError("");
                if (previewUrl) {
                  URL.revokeObjectURL(previewUrl);
                  setPreviewUrl("");
                }
              }}
            >
              {item.title}
            </button>
          ))}
        </div>

        <form className="upload-form" onSubmit={handleSubmit}>
          <label className="upload-box">
            <span>{modality?.title}</span>
            <small>{modality?.hint}</small>
            <input
              type="file"
              accept={modality?.accept}
              onChange={handleFileChange}
            />
          </label>

          {previewUrl ? (
            <img src={previewUrl} alt="Preview" className="preview" />
          ) : null}

          {file ? <p className="file-name">Selected: {file.name}</p> : null}

          <button type="submit" disabled={!file || loading}>
            {loading ? "Running inference..." : "Predict"}
          </button>
        </form>

        {error ? <p className="error">{error}</p> : null}

        {result ? (
          <section className="result-card">
            <p className="result-label">Prediction</p>
            <h2>{result.label}</h2>
            <p className="confidence">Confidence: {formatConfidence(result.confidence)}</p>
            <p className="meta">Modality: {result.modality}</p>
            <p className="meta">Model version: {result.model_version}</p>
          </section>
        ) : null}
      </main>
    </div>
  );
}
