/* global React, Pill, Tag, Eyebrow, KPI, SectionHead, Panel, Sparkline,
   NDMap, NDMapGIS, NDMapLegend, Toggle, Drawer, fmtDelta, fmtNum, useIDMode */
const { useState: useStateHome, useMemo: useMemoHome } = React;

function Home({ data, goto, openCounty }) {
  const D = data;
  const v = D.scorecard.verdict;
  const sc = D.scorecard;
  const L = D.labels;
  const showIDs = window.useIDMode();

  const [mapMetric, setMapMetric] = useStateHome("delta_mape");
  const [selectedCounty, setSelectedCounty] = useStateHome(null);
  const [scrolled, setScrolled] = useStateHome(false);

  // Watch scroll to enable sticky compressed verdict
  React.useEffect(() => {
    const main = document.querySelector(".app__main");
    if (!main) return;
    const handler = () => setScrolled(main.scrollTop > 220);
    main.addEventListener("scroll", handler, { passive: true });
    handler();
    return () => main.removeEventListener("scroll", handler);
  }, []);

  // Group rollups for the map summary
  const groupRollups = useMemoHome(() => {
    const buckets = {};
    D.counties.forEach(c => {
      buckets[c.group] = buckets[c.group] || { sum: 0, count: 0, worst: c, best: c };
      buckets[c.group].sum += c.delta_mape;
      buckets[c.group].count++;
      if (c.delta_mape > buckets[c.group].worst.delta_mape) buckets[c.group].worst = c;
      if (c.delta_mape < buckets[c.group].best.delta_mape) buckets[c.group].best = c;
    });
    return Object.entries(buckets).map(([k, v]) => ({
      group: k, mean: v.sum / v.count, count: v.count, worst: v.worst, best: v.best
    }));
  }, [D.counties]);

  // Compute max abs delta for in-row bar scaling on primary metrics
  const primaryMetrics = sc.metrics.filter(m => m.primary);
  const maxAbsDelta = Math.max(...primaryMetrics.map(m => Math.abs(m.challenger - m.champion)), 0.01);

  return (
    <>
      {/* ============== STICKY COMPRESSED VERDICT (slides in when scrolled) ============== */}
      <div className={`verdict-sticky ${scrolled ? "is-on" : ""}`}>
        <span className="verdict__status" style={{ margin: 0 }}>● {v.user_status_label}</span>
        <span>
          <span className="crumb" style={{ marginRight: 8 }}>{L.benchmark.title}</span>
          <strong>{L.champion.title}</strong>
          <span style={{ color: "var(--ink-4)", margin: "0 6px" }}>vs.</span>
          <strong>{L.challenger.title}</strong>
        </span>
        <button className="btn primary sm" onClick={() => goto("decision")}>
          Open decision brief →
        </button>
      </div>

      {/* ============== VERDICT STRIP ============== */}
      <div className={`verdict ${v.state === "ready_for_review" ? "ready" : v.state}`}>
        <div className="verdict__left">
          <div className="verdict__crumbs">
            <span>{L.benchmark.title}</span>
            <span className="sep">·</span>
            <span>{L.benchmark.sub}</span>
            {showIDs && <><span className="sep">·</span><span className="code">{L.benchmark.id}</span></>}
          </div>
          <div>
            <span className="verdict__status">● {v.user_status_label}</span>
            <span style={{ fontSize: 11, color: "var(--ink-4)", letterSpacing: "0.04em", textTransform: "uppercase", fontWeight: 600 }}>
              {v.confidence} · {v.operational_label}
            </span>
          </div>
          <div className="verdict__headline">{v.headline}</div>
          <div className="verdict__reason">{v.main_reason}</div>
          <div className="verdict__meta">
            <div>
              <div className="k">{L.champion.role}</div>
              <div className="v">{L.champion.title}</div>
              {showIDs && <div style={{ fontSize: 10, color: "var(--ink-4)", fontFamily: "var(--font-mono), Consolas, monospace" }}>{L.champion.method_id} · {L.champion.config_id}</div>}
            </div>
            <div>
              <div className="k">{L.challenger.role}</div>
              <div className="v">{L.challenger.title}</div>
              {showIDs && <div style={{ fontSize: 10, color: "var(--ink-4)", fontFamily: "var(--font-mono), Consolas, monospace" }}>{L.challenger.method_id} · {L.challenger.config_id}</div>}
            </div>
            <div>
              <div className="k">Record status</div>
              <div className="v">{v.status}</div>
            </div>
          </div>
        </div>
        <div className="verdict__right">
          <div className="right-label">Recommended action</div>
          <div style={{ fontFamily: "var(--font-display, Jost), sans-serif", fontSize: 15, fontWeight: 500, color: "var(--ink-1)", lineHeight: 1.3 }}>
            Promote candidate to production
          </div>
          <div style={{ fontSize: 11.5, color: "var(--ink-3)", lineHeight: 1.4 }}>
            Sign-off persists to the review folder; production alias updates on commit.
          </div>
          <div className="right-bar">
            <button className="btn primary" onClick={() => goto("decision")}>
              Open decision brief & sign off →
            </button>
            <button className="btn ghost sm" onClick={() => goto("scorecard")}>Inspect scorecard ›</button>
          </div>
        </div>
      </div>

      {/* ============== KPI ROW ============== */}
      <div className="kpi-row">
        <KPI k="County error"
             v="8.85 %"
             d="current 8.87 %"
             delta="−0.02 pp" deltaKind="good" />
        <KPI k="Urban / college error"
             v="11.68 %"
             d="current 12.86 %"
             delta="−1.18 pp" deltaKind="good" />
        <KPI k="Bakken error"
             v="19.61 %"
             d="current 19.09 %"
             delta="+0.52 pp" deltaKind="bad" />
        <KPI k="Recent state error"
             v="1.64 %"
             d="current 2.56 %"
             delta="−0.93 pp" deltaKind="good" />
      </div>

      {/* ============== DEEP-SEARCH STRIP ============== */}
      <div className="search-strip">
        <div>
          <div className="k">Deep search</div>
          <div className="v">
            <span className="pill-live" style={{ marginRight: 6 }}>● live</span>
            Convergence-window sweep
          </div>
          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.55)", marginTop: 2 }}>
            started {D.deep_search.started_at} · {D.deep_search.eta}
            {showIDs && <> · pack <code style={{ color: "rgba(255,255,255,0.7)" }}>{D.deep_search.pack}</code></>}
          </div>
        </div>
        <div>
          <div className="k">Progress</div>
          <div className="v">{D.deep_search.progress.completed}/{D.deep_search.progress.planned} <span className="sub">runs</span></div>
          <div className="progress-bar"><span style={{ width: `${(D.deep_search.progress.completed/D.deep_search.progress.planned)*100}%` }} /></div>
        </div>
        <div>
          <div className="k">Candidates found</div>
          <div className="v">{D.deep_search.progress.candidates_found} <span className="sub">of {D.deep_search.progress.completed} runs</span></div>
        </div>
        <div>
          <div className="k">CPU allocation</div>
          <div className="v">{D.deep_search.cores.parallel_runs} × {D.deep_search.cores.workers_per_run} <span className="sub">= {D.deep_search.cores.allocated} cores</span></div>
        </div>
        <div style={{ textAlign: "right" }}>
          <a href="#" className="deep-link" onClick={(e) => { e.preventDefault(); goto("search"); }}>
            Open deep search →
          </a>
        </div>
      </div>

      {/* ============== THREE-COLUMN: HEAD-TO-HEAD + WHAT CHANGED + HEALTH ============== */}
      <div className="row-3">
        {/* Head-to-head primary metrics with in-row delta bars */}
        <Panel title="Primary metrics — current vs. candidate"
               crumbs="decision evidence"
               sub="lower is better unless noted"
               footer={<>Full scorecard with segment lenses lives in <a href="#" onClick={(e)=>{e.preventDefault(); goto("scorecard");}}>Scorecard ›</a></>}>
          <table className="tbl">
            <thead>
              <tr>
                <th>Metric</th>
                <th className="num">Current</th>
                <th className="num">Candidate</th>
                <th className="num" style={{ width: 50 }}>Δ</th>
                <th style={{ width: 90 }}></th>
              </tr>
            </thead>
            <tbody>
              {primaryMetrics.map(m => {
                const delta = m.challenger - m.champion;
                const lower = m.lower_is_better;
                const good = lower ? delta < 0 : delta > 0;
                const targetZero = m.target_zero;
                const zeroGood = targetZero ? Math.abs(m.challenger) < Math.abs(m.champion) : null;
                const isGood = targetZero ? zeroGood : good;
                const pct = Math.min(100, (Math.abs(delta) / maxAbsDelta) * 100);
                return (
                  <tr key={m.key} className="primary-row">
                    <td>{m.label}</td>
                    <td className="num">{m.champion.toFixed(2)}</td>
                    <td className="num">{m.challenger.toFixed(2)}</td>
                    <td className={`num ${Math.abs(delta) < 0.005 ? "muted" : (isGood ? "pos" : "neg")}`}>
                      {delta > 0 ? "+" : ""}{delta.toFixed(2)}
                    </td>
                    <td>
                      <div style={{
                        position: "relative", height: 8, background: "var(--paper-2)",
                        borderRadius: 1, overflow: "hidden",
                      }}>
                        <div style={{
                          position: "absolute", left: "50%", top: 0, bottom: 0,
                          width: 1, background: "var(--ink-5)",
                        }} />
                        <div style={{
                          position: "absolute", top: 0, bottom: 0,
                          left: delta < 0 ? `${50 - pct/2}%` : "50%",
                          width: `${pct/2}%`,
                          background: isGood ? "var(--accent)" : "var(--bad)",
                          opacity: 0.85,
                        }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
            <Tag kind="good">All hard gates clean</Tag>
            <Tag kind="good">Operational artifacts complete</Tag>
            <Tag>Runtime 31m / 45m budget</Tag>
          </div>
        </Panel>

        {/* What changed feed */}
        <Panel title="What changed since you last looked"
               sub="last 72 hours">
          <div className="activity">
            {D.activity.map((a, i) => (
              <div key={i} className="row">
                <div className={`icon ${a.icon}`}>
                  {a.icon === "decision" && "◆"}
                  {a.icon === "search" && "↻"}
                  {a.icon === "alert" && "!"}
                  {a.icon === "data" && "⌬"}
                  {a.icon === "promotion" && "★"}
                </div>
                <div className="when">{a.when}</div>
                <div className="text">
                  {a.text} <a href="#" onClick={(e) => { e.preventDefault(); goto(a.tab); }}>open ›</a>
                </div>
              </div>
            ))}
          </div>
        </Panel>

        {/* Production health */}
        <Panel title="Production health"
               sub="what's currently live"
               footer={<>{D.production_health.days_in_production} days since last promotion · revalidation clean</>}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div>
              <Eyebrow>In production</Eyebrow>
              <div style={{ fontWeight: 600, marginTop: 2, fontSize: 13 }}>
                {D.production_health.champion_label}
              </div>
              {showIDs && <div style={{ fontSize: 10.5, color: "var(--ink-4)", fontFamily: "var(--font-mono), Consolas, monospace" }}>{D.production_health.champion_method} · {D.production_health.champion_config}</div>}
              <div style={{ fontSize: 11, color: "var(--ink-3)" }}>set {D.production_health.alias_set_on}</div>
            </div>
            <div>
              <Eyebrow>Drift indicators</Eyebrow>
              <div className="drift-list" style={{ marginTop: 4 }}>
                {D.production_health.drift_indicators.map((d, i) => (
                  <div key={i} className="drift-row" style={{ gridColumn: "span 2" }}>
                    <span className="k">{d.label}</span>
                    <span className="v">{d.value}</span>
                    <span className={`arrow ${d.trend}`}>
                      {d.trend === "stable" ? "─" : d.trend === "drifting" ? "↗" : "↑"}
                    </span>
                    <span style={{ gridColumn: "1 / -1", fontSize: 11, color: "var(--ink-4)", marginTop: -2 }}>{d.detail}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <Eyebrow>Last 5 vintages (county MAPE)</Eyebrow>
              <Sparkline values={D.production_health.last_5_vintages.map(v => v.county_mape)} color="var(--primary)" fill="var(--primary)" dots height={36} width={240} />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: "var(--ink-4)", marginTop: -2 }}>
                {D.production_health.last_5_vintages.map(v => <span key={v.vintage}>{v.vintage.slice(2)}</span>)}
              </div>
            </div>
          </div>
        </Panel>
      </div>

      {/* ============== ND COUNTY CHOROPLETH (GIS) ============== */}
      <div className="row-2-3">
        <Panel title="County-level signal" crumbs="53 ND counties"
               action={
                 <Toggle
                   value={mapMetric}
                   onChange={setMapMetric}
                   options={[
                     { value: "delta_mape",      label: "Δ (candidate − current)" },
                     { value: "champion_mape",   label: "Current error" },
                     { value: "challenger_mape", label: "Candidate error" },
                     { value: "bias",            label: "Signed bias" },
                   ]}/>
               }
               footer={<><NDMapLegend metric={mapMetric} /></>}>
          <NDMapGIS counties={D.counties} metric={mapMetric}
                    selected={selectedCounty}
                    onSelect={(c) => { setSelectedCounty(c); openCounty && openCounty(c); }}
                    height={380}
                    fallback={<NDMap counties={D.counties} metric={mapMetric}
                                     selected={selectedCounty}
                                     onSelect={(c) => { setSelectedCounty(c); openCounty && openCounty(c); }} />} />
          {selectedCounty && (
            <div className="county-detail">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <div>
                  <span className="nm">{selectedCounty.name} County</span>
                  <span className="grp" style={{ marginLeft: 8 }}>{selectedCounty.group.replace("_", " / ")}</span>
                </div>
                <span style={{ fontSize: 11, color: "var(--ink-4)" }}>
                  {showIDs && <>FIPS {selectedCounty.fips} · </>}
                  pop {selectedCounty.pop.toLocaleString()}
                </span>
              </div>
              <div className="gridkv">
                <div><span className="k">Current error: </span><span className="v">{selectedCounty.champion_mape.toFixed(2)} %</span></div>
                <div><span className="k">Candidate error: </span><span className="v">{selectedCounty.challenger_mape.toFixed(2)} %</span></div>
                <div><span className="k">Δ: </span><span className="v" style={{ color: selectedCounty.delta_mape < 0 ? "#054550" : "#5b1a1c" }}>
                  {selectedCounty.delta_mape > 0 ? "+" : ""}{selectedCounty.delta_mape.toFixed(2)} pp
                </span></div>
                <div><span className="k">Signed bias (candidate): </span><span className="v">{selectedCounty.signed_bias_challenger.toFixed(2)}</span></div>
              </div>
            </div>
          )}
        </Panel>

        <Panel title="Group rollups" crumbs="county segments"
               sub="ND has 53 counties split across these lenses">
          <table className="tbl">
            <thead>
              <tr>
                <th>Group</th>
                <th className="num">N</th>
                <th className="num">Mean Δ</th>
                <th>Worst county</th>
                <th>Best county</th>
              </tr>
            </thead>
            <tbody>
              {groupRollups.map(g => (
                <tr key={g.group}>
                  <td>
                    <strong style={{ textTransform: "capitalize" }}>{g.group.replace("_", " / ")}</strong>
                  </td>
                  <td className="num">{g.count}</td>
                  <td className={`num ${g.mean < 0 ? "pos" : "neg"}`}>
                    {g.mean > 0 ? "+" : ""}{g.mean.toFixed(2)}
                  </td>
                  <td>
                    {g.worst.name}
                    <div className="small">{g.worst.delta_mape > 0 ? "+" : ""}{g.worst.delta_mape.toFixed(2)} pp</div>
                  </td>
                  <td>
                    {g.best.name}
                    <div className="small">{g.best.delta_mape > 0 ? "+" : ""}{g.best.delta_mape.toFixed(2)} pp</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <hr className="rule" />
          <Eyebrow>Long horizon (2025–2055)</Eyebrow>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 4 }}>
            <div>
              <div className="kv"><span className="k">State drift</span><span style={{ marginLeft: "auto", fontWeight: 600 }}>2.90 %</span></div>
              <Sparkline values={[1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9]} color="var(--primary)" fill="var(--primary)" />
              <div className="small" style={{ color: "var(--ink-4)" }}>current 3.22 % · improving 0.31 pp</div>
            </div>
            <div>
              <div className="kv"><span className="k">Negative-pop violations</span><span style={{ marginLeft: "auto", fontWeight: 600 }}>0 / 0</span></div>
              <div className="wf-placeholder" style={{ height: 38, minHeight: 0 }}>0 violations across all 53 counties × 30y</div>
            </div>
          </div>
        </Panel>
      </div>
    </>
  );
}

window.Home = Home;
