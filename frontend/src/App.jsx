import { useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function App() {
  const [file, setFile] = useState(null)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [query, setQuery] = useState('')
  const [evaluating, setEvaluating] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      if (selectedFile.type !== 'application/pdf') {
        setError('Please select a PDF file')
        return
      }
      setFile(selectedFile)
      setError(null)
      // Don't clear uploadStatus here - only clear results
      setResults(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    setUploading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE_URL}/uploadpdf`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const data = await response.json()
      setUploadStatus(data)
      setError(null)
    } catch (err) {
      setError(err.message || 'Failed to upload PDF')
      setUploadStatus(null)
    } finally {
      setUploading(false)
    }
  }

  const handleEvaluate = async () => {
    if (!query.trim()) {
      setError('Please enter a query')
      return
    }

    if (!uploadStatus) {
      setError('Please upload a PDF first')
      return
    }

    setEvaluating(true)
    setError(null)

    try {
      const response = await fetch(
        `${API_BASE_URL}/evaluate?query=${encodeURIComponent(query)}`,
        {
          method: 'POST',
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Evaluation failed')
      }

      const data = await response.json()
      setResults(data)
      setError(null)
    } catch (err) {
      setError(err.message || 'Failed to evaluate query')
      setResults(null)
    } finally {
      setEvaluating(false)
    }
  }

  const getRiskColor = (risk) => {
    if (risk === 'High Trust') return '#4caf50'
    if (risk === 'Medium Trust') return '#ff9800'
    return '#f44336'
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>TrustAI - LLM Trustworthiness Evaluator</h1>
        <p>Upload a PDF document and evaluate LLM responses for trustworthiness</p>
      </header>

      <main className="app-main">
        {/* File Upload Section */}
        <section className="section">
          <h2>1. Upload PDF Document</h2>
          <div className="upload-section">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="file-input"
              id="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              {file ? file.name : 'Choose PDF File'}
            </label>
            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="primary-button"
            >
              {uploading ? 'Uploading...' : 'Upload PDF'}
            </button>
          </div>
          {uploadStatus && (
            <div className="success-message">
              ✓ {uploadStatus.message} - {uploadStatus.filename}
            </div>
          )}
        </section>

        {/* Query Input Section */}
        <section className="section">
          <h2>2. Enter Your Query</h2>
          <div className="query-section">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your question here..."
              className="query-input"
              rows="4"
            />
            <button
              onClick={handleEvaluate}
              disabled={!uploadStatus || evaluating || !query.trim()}
              className="primary-button"
            >
              {evaluating ? 'Evaluating...' : 'Evaluate Query'}
            </button>
            {(!uploadStatus || !query.trim()) && (
              <p className="help-text">
                {!uploadStatus && '⚠ Please upload a PDF first. '}
                {!query.trim() && '⚠ Please enter a query.'}
              </p>
            )}
          </div>
        </section>

        {/* Error Display */}
        {error && (
          <div className="error-message">
            ⚠ {error}
          </div>
        )}

        {/* Results Section */}
        {results && (
          <section className="section results-section">
            <h2>Evaluation Results</h2>

            {/* Query and Answer */}
            <div className="result-card">
              <h3>Query</h3>
              <p className="query-text">{results.query}</p>
            </div>

            <div className="result-card">
              <h3>LLM Answer</h3>
              <p className="answer-text">{results.llm_answer}</p>
            </div>

            {/* Risk Classification */}
            <div
              className="result-card risk-card"
              style={{ borderColor: getRiskColor(results.risk_classification) }}
            >
              <h3>Risk Classification</h3>
              <div
                className="risk-badge"
                style={{ backgroundColor: getRiskColor(results.risk_classification) }}
              >
                {results.risk_classification}
              </div>
              <p className="trust-score">
                Trust Score: <strong>{results.scores.trust_score.toFixed(2)}</strong>
              </p>
            </div>

            {/* Scores Breakdown */}
            <div className="result-card">
              <h3>Detailed Scores</h3>
              <div className="scores-grid">
                <div className="score-item">
                  <span className="score-label">Trust Score</span>
                  <span className="score-value">
                    {results.scores.trust_score.toFixed(2)}
                  </span>
                </div>
                <div className="score-item">
                  <span className="score-label">Semantic Similarity</span>
                  <span className="score-value">
                    {results.scores.semantic_similarity?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                <div className="score-item">
                  <span className="score-label">Retrieval Overlap</span>
                  <span className="score-value">
                    {results.scores.retrieval_overlap?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                <div className="score-item">
                  <span className="score-label">Contradiction Risk</span>
                  <span className="score-value">
                    {results.scores.nli?.contradiction_risk?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                <div className="score-item">
                  <span className="score-label">Evidence Count</span>
                  <span className="score-value">{results.evidence_count}</span>
                </div>
              </div>
            </div>

            {/* Explanation */}
            <div className="result-card">
              <h3>Explanation</h3>
              <p className="explanation-text">{results.explanation}</p>
            </div>

            {/* Evidence Chunks */}
            {results.evidence_chunks && results.evidence_chunks.length > 0 && (
              <div className="result-card">
                <h3>Retrieved Evidence (Top {results.evidence_chunks.length})</h3>
                <div className="evidence-list">
                  {results.evidence_chunks.map((chunk, index) => (
                    <div key={index} className="evidence-item">
                      <div className="evidence-header">
                        Evidence #{index + 1}
                        {chunk.metadata && chunk.metadata.page && (
                          <span className="evidence-page">
                            Page {chunk.metadata.page}
                          </span>
                        )}
                      </div>
                      <p className="evidence-content">{chunk.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
