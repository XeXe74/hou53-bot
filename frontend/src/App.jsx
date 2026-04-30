import { useState, useRef } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EXAMPLES = [
  '3 bedroom house built in 1998, 2 car garage, central air, good kitchen quality',
  'Large 4 bed 2 bath colonial, 2500 sqft, excellent overall quality, finished basement',
  'Small 2 bedroom bungalow, no garage, older construction from 1955, needs work',
]

function formatPrice(n) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n)
}

function ThemeToggle() {
  const [dark, setDark] = useState(() =>
    window.matchMedia('(prefers-color-scheme: dark)').matches
  )
  function toggle() {
    const next = !dark
    setDark(next)
    document.documentElement.setAttribute('data-theme', next ? 'dark' : 'light')
  }
  return (
    <button className="theme-toggle" onClick={toggle} aria-label="Toggle theme">
      {dark
        ? <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
        : <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      }
    </button>
  )
}

function FeatureBar({ name, importance, maxImportance }) {
  const pct = maxImportance > 0 ? (importance / maxImportance) * 100 : 0
  return (
    <div className="feature-row">
      <div className="feature-header">
        <span className="feature-name">{name}</span>
        <span className="feature-value">{importance.toFixed(4)}</span>
      </div>
      <div className="feature-bar-track">
        <div className="feature-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function ResultView({ result, onReset }) {
  const maxImportance = Math.max(...result.top_features.map(f => f.importance))
  const extractedEntries = Object.entries(result.extracted_features)

  return (
    <div className="result">
      <div className="price-card">
        <span className="price-label">Estimated Market Value</span>
        <span className="price-value">{formatPrice(result.predicted_price)}</span>
        <span className="price-range">
          Range: <strong>{formatPrice(result.price_range_low)}</strong> — <strong>{formatPrice(result.price_range_high)}</strong>
        </span>
        <p className="summary-text">{result.summary}</p>
      </div>

      <div className="features-card">
        <div className="features-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          Top factors influencing this estimate
        </div>
        {result.top_features.map(f => (
          <FeatureBar key={f.feature} name={f.description} importance={f.importance} maxImportance={maxImportance} />
        ))}
        <span className="model-badge">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
          Model: {result.model_used}
        </span>
      </div>

      {extractedEntries.length > 0 && (
        <div className="extracted-card">
          <div className="extracted-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
            Features extracted from your description
          </div>
          <div className="extracted-grid">
            {extractedEntries.map(([k, v]) => (
              <div className="extracted-item" key={k}>
                <span className="extracted-key">{k}</span>
                <span className="extracted-val">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <button className="reset-btn" onClick={onReset}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4"/></svg>
        Estimate another house
      </button>
    </div>
  )
}

export default function App() {
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const textareaRef = useRef(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!description.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: description.trim() }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Server error ${res.status}`)
      }
      setResult(await res.json())
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function handleExample(text) {
    setDescription(text)
    setResult(null)
    setError(null)
    textareaRef.current?.focus()
  }

  function handleReset() {
    setResult(null)
    setError(null)
    setDescription('')
    textareaRef.current?.focus()
  }

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-logo">
          <img
            src="/logo.png"
            alt="HOU53-bot mascot"
            width="40"
            height="40"
            className="header-avatar"
          />
          <div>
            <div className="header-logo-text">HOU53<span>-bot</span></div>
            <div className="header-tagline">AI House Valuation</div>
          </div>
        </div>
        <ThemeToggle />
      </header>

      <main className="main">
        {!result ? (
          <>
            {/* ── Hero ── */}
            <div className="hero">
              <div className="hero-content">
                <h1 className="hero-title">What is your house <span>worth?</span></h1>
                <p className="hero-subtitle">
                  Describe a property in plain language and get an instant price estimate powered by machine learning.
                </p>
              </div>
              <div className="hero-image-wrap">
                <img
                  src="/cover.png"
                  alt="HOU53-bot real estate agent"
                  className="hero-image"
                  width="480"
                  height="270"
                  loading="eager"
                />
              </div>
            </div>

            {/* ── Input card ── */}
            <form className="input-card" onSubmit={handleSubmit}>
              <label className="input-label" htmlFor="description">Describe the house</label>
              <textarea
                id="description"
                ref={textareaRef}
                className="input-textarea"
                value={description}
                onChange={e => setDescription(e.target.value)}
                placeholder="e.g. A 4 bedroom house built in 2005 with a finished basement, 2-car attached garage, central air, and excellent kitchen quality..."
                rows={5}
                maxLength={2000}
              />
              <div className="input-footer">
                <span className="input-hint">{description.length} / 2000 characters</span>
                <button className="submit-btn" type="submit" disabled={loading || description.trim().length < 10}>
                  {loading
                    ? <><span className="spinner" /> Analyzing…</>
                    : <><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg> Estimate price</>
                  }
                </button>
              </div>
            </form>

            {/* ── Examples ── */}
            <div className="examples">
              <span className="examples-label">Try an example:</span>
              {EXAMPLES.map((ex, i) => (
                <button key={i} className="example-chip" onClick={() => handleExample(ex)}>
                  {ex.length > 55 ? ex.slice(0, 55) + '…' : ex}
                </button>
              ))}
            </div>

            {error && (
              <div className="error-banner" role="alert">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                {error}
              </div>
            )}
          </>
        ) : (
          <ResultView result={result} onReset={handleReset} />
        )}
      </main>

      <footer className="footer">
        HOU53-bot · Trained on the Ames Housing Dataset · For educational purposes only
      </footer>
    </div>
  )
}