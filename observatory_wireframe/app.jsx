/* global React, ReactDOM, Home, DecisionBrief, Scorecard, DeepSearch, History */
const { useState: useStateApp, useContext, createContext } = React;

/* ============================================================
   Global "show technical IDs" mode.
   When off (default), the UI shows friendly names like
   "2026 baseline" instead of method/config/run codes.
   Toggle lives in the top bar.
   ============================================================ */
const IDModeCtx = createContext(false);
window.useIDMode = function useIDMode() { return useContext(IDModeCtx); };

const NAV = [
  { id: "home",      label: "Decision console", icon: "◎" },
  { id: "decision",  label: "Decision brief",   icon: "◆", badge: "Draft" },
  { id: "scorecard", label: "Scorecard",        icon: "▦" },
  { id: "search",    label: "Deep search",      icon: "↻", badge: "Live" },
  { id: "history",   label: "History",          icon: "⌖" },
];

function App() {
  const [route, setRoute] = useStateApp("home");
  const [countyDrawer, setCountyDrawer] = useStateApp(null);
  const [showIDs, setShowIDs] = useStateApp(false);
  const data = window.OBS_DATA;

  const goto = (id) => { setRoute(id); window.scrollTo(0, 0); };

  return (
    <IDModeCtx.Provider value={showIDs}>
    <div className="app">
      {/* Top bar */}
      <div className="app__topbar">
        <span className="brand">
          <span className="glyph">ND</span>
          State Data Center
        </span>
        <span style={{ width: 1, height: 22, background: "rgba(255,255,255,0.18)" }} />
        <span className="product">Projection Observatory</span>
        <span className="spacer" />
        <span className="meta">
          <button onClick={() => setShowIDs(s => !s)} style={{
            background: showIDs ? "rgba(250,162,27,0.15)" : "rgba(255,255,255,0.06)",
            border: "1px solid " + (showIDs ? "rgba(250,162,27,0.4)" : "rgba(255,255,255,0.18)"),
            color: showIDs ? "var(--nd-harvest-orange)" : "rgba(255,255,255,0.85)",
            padding: "2px 8px", borderRadius: 999, fontSize: 10.5,
            letterSpacing: "0.04em", textTransform: "uppercase", cursor: "pointer",
            fontFamily: "inherit", fontWeight: 600,
          }} title="Toggle technical IDs (method codes, run IDs, file paths)">
            IDs · {showIDs ? "on" : "off"}
          </button>
          <span className="pill-live">Deep search · running</span>
          <span><span style={{ color: "rgba(255,255,255,0.55)" }}>vintage</span> <b>2026</b></span>
          <span><span style={{ color: "rgba(255,255,255,0.55)" }}>refreshed</span> <b>3 min ago</b></span>
        </span>
      </div>

      {/* Sidebar */}
      <nav className="app__nav">
        <div className="nav-eyebrow">Workflow</div>
        {NAV.map(n => (
          <button key={n.id}
            className={`nav-item ${route === n.id ? "active" : ""}`}
            onClick={() => goto(n.id)}>
            <span className="icon">{n.icon}</span>
            <span>{n.label}</span>
            {n.badge && <span className="badge">{n.badge}</span>}
          </button>
        ))}
        <div className="nav-foot">
          <div style={{ fontWeight: 600, color: "rgba(255,255,255,0.85)" }}>Wireframe v0.1</div>
          <div>53 counties · 30 yr horizon</div>
          <div>SOP-003 promotion workflow</div>
          <div style={{ marginTop: 8 }}>
            <span style={{
              fontSize: 9.5, letterSpacing: "0.12em", textTransform: "uppercase",
              background: "rgba(250,162,27,0.12)", color: "var(--nd-harvest-orange)",
              padding: "2px 6px", borderRadius: 2, border: "1px solid rgba(250,162,27,0.3)",
            }}>Internal · one-operator</span>
          </div>
        </div>
      </nav>

      {/* Main */}
      <main className="app__main" data-screen-label={NAV.find(n => n.id === route)?.label}>
        {route === "home"      && <Home data={data} goto={goto} openCounty={setCountyDrawer} />}
        {route === "decision"  && <DecisionBrief data={data} goto={goto} />}
        {route === "scorecard" && <Scorecard data={data} goto={goto} />}
        {route === "search"    && <DeepSearch data={data} goto={goto} />}
        {route === "history"   && <History data={data} goto={goto} />}
      </main>

      {/* County drawer (optional progressive disclosure from map) */}
      {countyDrawer && (
        <Drawer title={`${countyDrawer.name} County`}
                onClose={() => setCountyDrawer(null)}
                footer={
                  <>
                    <button className="btn primary" onClick={() => { setCountyDrawer(null); goto("scorecard"); }}>Open in scorecard ›</button>
                    <button className="btn ghost" onClick={() => setCountyDrawer(null)}>Close</button>
                  </>
                }>
          <CountyDetail c={countyDrawer} data={data} />
        </Drawer>
      )}
    </div>
    </IDModeCtx.Provider>
  );
}

function CountyDetail({ c, data }) {
  const showIDs = window.useIDMode();
  // Synthesize a per-county trajectory for current + candidate
  const years = Array.from({ length: 30 }, (_, i) => 2025 + i);
  const champ = years.map((y, i) => c.pop * Math.pow(1 + (0.002 + (i - 15) * 0.0006), i));
  const chall = champ.map((v, i) => v * (1 - c.delta_mape / 100 * (i / 30) * 0.4));
  const ratio = chall.map((v, i) => (v / champ[i] - 1) * 100);
  return (
    <div>
      <div style={{ display: "flex", gap: 10, alignItems: "baseline" }}>
        <Pill kind="champion">{c.group.replace("_"," / ")}</Pill>
        <span style={{ fontSize: 12, color: "var(--ink-4)" }}>
          {showIDs && <>FIPS {c.fips} · </>}
          2024 pop {c.pop.toLocaleString()}
        </span>
      </div>

      <div className="kpi-row" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 10 }}>
        <KPI k="Current error" v={`${c.champion_mape.toFixed(2)} %`} d="in production" />
        <KPI k="Candidate error" v={`${c.challenger_mape.toFixed(2)} %`} d="college-fix candidate"
             delta={`${c.delta_mape > 0 ? "+" : ""}${c.delta_mape.toFixed(2)} pp`}
             deltaKind={c.delta_mape < 0 ? "good" : "bad"} />
      </div>

      <Panel title="Projection curves (candidate vs current)" padding="dense">
        <svg viewBox="0 0 460 160" style={{ width: "100%", height: 160 }}>
          {[20, 60, 100, 140].map(y => (
            <line key={y} x1="36" x2="450" y1={y} y2={y} stroke="var(--line-2)" />
          ))}
          {[2025, 2035, 2045, 2055].map((y, i) => (
            <text key={y} x={36 + i * 138} y={155} fontSize="9" fill="var(--ink-4)" textAnchor="middle">{y}</text>
          ))}
          {(() => {
            const minP = Math.min(...champ, ...chall) * 0.95;
            const maxP = Math.max(...champ, ...chall) * 1.05;
            const range = maxP - minP || 1;
            const px = (i) => 36 + (i / 29) * 414;
            const py = (v) => 20 + (1 - (v - minP) / range) * 120;
            const cPath = champ.map((v, i) => `${i === 0 ? "M" : "L"} ${px(i)} ${py(v)}`).join(" ");
            const chPath = chall.map((v, i) => `${i === 0 ? "M" : "L"} ${px(i)} ${py(v)}`).join(" ");
            return (
              <>
                <path d={cPath} fill="none" stroke="var(--primary)" strokeWidth="1.6" />
                <path d={chPath} fill="none" stroke="var(--accent)" strokeWidth="1.6" strokeDasharray="3 3" />
              </>
            );
          })()}
        </svg>
        <div style={{ display: "flex", gap: 12, fontSize: 11 }}>
          <span className="legend-key"><span className="sw" style={{ background: "var(--primary)" }} /> Current</span>
          <span className="legend-key"><span className="sw" style={{ background: "var(--accent)" }} /> Candidate</span>
        </div>
      </Panel>

      <Panel title="Horizon bias (Δ pop %)" padding="dense" style={{ marginTop: 10 }}>
        <Sparkline values={ratio} color="var(--accent)" fill="var(--accent)" height={48} width={460} dots />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: "var(--ink-4)" }}>
          <span>2025</span><span>2040</span><span>2055</span>
        </div>
      </Panel>

      <Panel title="Latest residual (2024 estimate vs projection)" padding="dense" style={{ marginTop: 10 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 12 }}>
          <div><span style={{ color: "var(--ink-4)" }}>Residual: </span><strong>{c.last_residual_2024.toFixed(2)} %</strong></div>
          <div><span style={{ color: "var(--ink-4)" }}>Bias (candidate): </span><strong>{c.signed_bias_challenger.toFixed(2)}</strong></div>
        </div>
      </Panel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
