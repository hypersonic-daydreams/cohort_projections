/* global React, Pill, Tag, KPI, SectionHead, Panel, Eyebrow, Sparkline,
   Toggle, NDMap, NDMapLegend, Drawer, fmtDelta, fmtNum */

/* ============================================================
   Scorecard — dense head-to-head table with segment lenses
   ============================================================ */
function Scorecard({ data, goto }) {
  const sc = data.scorecard;
  const L = data.labels;
  const showIDs = window.useIDMode();
  const [lens, setLens] = React.useState("all");

  // Compose extended metric list with segment lenses
  const lenses = [
    { key: "all", label: "All metrics" },
    { key: "state", label: "State backtest" },
    { key: "county", label: "County segments" },
    { key: "horizon", label: "Long horizon" },
    { key: "operational", label: "Operational" },
  ];

  const allRows = [
    { group: "state",   ...sc.metrics.find(m => m.key === "state_ape_short") },
    { group: "state",   ...sc.metrics.find(m => m.key === "state_ape_medium") },
    { group: "state",   ...sc.metrics.find(m => m.key === "signed_bias") },
    { group: "county",  ...sc.metrics.find(m => m.key === "county_mape_overall") },
    { group: "county",  ...sc.metrics.find(m => m.key === "county_mape_urban") },
    { group: "county",  ...sc.metrics.find(m => m.key === "county_mape_rural") },
    { group: "county",  ...sc.metrics.find(m => m.key === "county_mape_bakken") },
    { group: "county",  ...sc.metrics.find(m => m.key === "county_mape_res") },
    { group: "county",  ...sc.metrics.find(m => m.key === "county_mape_smallest") },
    { group: "horizon", ...sc.metrics.find(m => m.key === "horizon_long_drift") },
  ];

  const segmentColor = (delta) => {
    if (Math.abs(delta) < 0.05) return "neutral";
    return delta < 0 ? "good" : "bad";
  };

  // Worst / best per group from county data
  const worstCounties = [...data.counties].sort((a, b) => b.delta_mape - a.delta_mape).slice(0, 8);
  const bestCounties  = [...data.counties].sort((a, b) => a.delta_mape - b.delta_mape).slice(0, 8);

  return (
    <>
      <SectionHead eyebrow="Decision evidence" title="Scorecard" variant="decision"
        sub={L.benchmark.title + (showIDs ? " · " + L.benchmark.id : "")}>
        <button className="btn ghost sm" onClick={() => goto("decision")}>Back to decision brief ›</button>
      </SectionHead>

      {/* Lens row */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <Eyebrow>Lens</Eyebrow>
        <Toggle value={lens} onChange={setLens}
                options={lenses.map(l => ({ value: l.key, label: l.label }))} />
        <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          <Tag kind="good">All hard gates clean</Tag>
          <Tag>Runtime 31m</Tag>
          <Tag>Δ recorded in pp</Tag>
        </div>
      </div>

      {/* Head-to-head table */}
      <Panel title="Head-to-head"
             crumbs="current · candidate · delta · verdict"
             sub={L.champion.title + " vs. " + L.challenger.title}>
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: "32%" }}>Metric</th>
              <th>Segment</th>
              <th className="num">Current</th>
              <th className="num">Candidate</th>
              <th className="num">Δ</th>
              <th className="num">Δ %</th>
              <th>Verdict</th>
              <th>Worst county</th>
            </tr>
          </thead>
          <tbody>
            {[ "state", "county", "horizon" ].filter(g => lens === "all" || lens === g).map(group => (
              <React.Fragment key={group}>
                <tr className="divider">
                  <td colSpan="8">{group === "state" ? "State backtest (recent origin)" :
                                  group === "county" ? "County segments" :
                                  "Long horizon"}</td>
                </tr>
                {allRows.filter(r => r.group === group).map(m => {
                  const delta = m.challenger - m.champion;
                  const lower = m.lower_is_better;
                  const targetZero = m.target_zero;
                  const good = targetZero ? Math.abs(m.challenger) < Math.abs(m.champion)
                                          : (lower ? delta < 0 : delta > 0);
                  const pctDelta = m.champion !== 0 ? (delta / m.champion) * 100 : 0;
                  // pick a relevant worst-county summary for county_* rows
                  let worst = null;
                  if (m.key === "county_mape_urban") {
                    worst = data.counties.filter(c => c.group === "urban_college").sort((a,b)=>b.delta_mape - a.delta_mape)[0];
                  } else if (m.key === "county_mape_bakken") {
                    worst = data.counties.filter(c => c.group === "bakken").sort((a,b)=>b.delta_mape - a.delta_mape)[0];
                  } else if (m.key === "county_mape_rural") {
                    worst = data.counties.filter(c => c.group === "rural").sort((a,b)=>b.delta_mape - a.delta_mape)[0];
                  } else if (m.key === "county_mape_res") {
                    worst = data.counties.filter(c => c.group === "reservation").sort((a,b)=>b.delta_mape - a.delta_mape)[0];
                  }
                  return (
                    <tr key={m.key}>
                      <td>{m.label}</td>
                      <td>{m.key.replace(/^county_mape_/, "")
                              .replace(/^state_ape_/, "state ")
                              .replace(/^horizon_/, "")}
                          {targetZero && <Tag style={{ marginLeft: 4 }}>target 0</Tag>}
                          {m.primary && <Tag kind="good" style={{ marginLeft: 4 }}>primary</Tag>}</td>
                      <td className="num">{m.champion.toFixed(3)}</td>
                      <td className="num">{m.challenger.toFixed(3)}</td>
                      <td className={`num ${Math.abs(delta) < 0.005 ? "muted" : good ? "pos" : "neg"}`}>
                        {delta > 0 ? "+" : ""}{delta.toFixed(3)}
                      </td>
                      <td className={`num ${Math.abs(pctDelta) < 0.1 ? "muted" : good ? "pos" : "neg"}`}>
                        {pctDelta > 0 ? "+" : ""}{pctDelta.toFixed(1)} %
                      </td>
                      <td>
                        {Math.abs(delta) < 0.005 ? <Pill kind="inconclusive">no change</Pill> :
                         good                    ? <Pill kind="passed_all_gates">improved</Pill> :
                                                   <Pill kind="needs_human_review">worsened</Pill>}
                      </td>
                      <td className="small">
                        {worst ? <>{worst.name}<br /><span className="muted">{worst.delta_mape > 0 ? "+" : ""}{worst.delta_mape.toFixed(2)} pp</span></> : <span className="muted">—</span>}
                      </td>
                    </tr>
                  );
                })}
              </React.Fragment>
            ))}
            {(lens === "all" || lens === "operational") && (
              <>
                <tr className="divider"><td colSpan="8">Operational gates</td></tr>
                {sc.hard_constraints.map((h, i) => (
                  <tr key={i}>
                    <td>{h.name}</td>
                    <td><Tag kind={h.gate === "hard" ? "bad" : ""}>{h.gate}</Tag></td>
                    <td className="num">{typeof h.champion === "boolean" ? (h.champion ? "yes" : "no") : String(h.champion)}</td>
                    <td className="num">{typeof h.challenger === "boolean" ? (h.challenger ? "yes" : "no") : String(h.challenger)}</td>
                    <td className="num muted">—</td>
                    <td className="num muted">—</td>
                    <td><Pill kind="passed_all_gates">clean</Pill></td>
                    <td className="muted small">—</td>
                  </tr>
                ))}
              </>
            )}
          </tbody>
        </table>
      </Panel>

      {/* County leaderboard split */}
      <div className="row-2">
        <Panel title="Counties where the candidate does better"
               crumbs="top 8 by Δ"
               sub="negative Δ = candidate wins">
          <table className="tbl">
            <thead>
              <tr>
                <th>County</th>
                <th>Group</th>
                <th className="num">Champion</th>
                <th className="num">Challenger</th>
                <th className="num">Δ</th>
              </tr>
            </thead>
            <tbody>
              {bestCounties.map(c => (
                <tr key={c.fips}>
                  <td><strong>{c.name}</strong></td>
                  <td><Tag>{c.group.replace("_", "/")}</Tag></td>
                  <td className="num">{c.champion_mape.toFixed(2)}</td>
                  <td className="num">{c.challenger_mape.toFixed(2)}</td>
                  <td className="num pos">{c.delta_mape.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel title="Counties where the candidate does worse"
               crumbs="top 8 by Δ"
               sub="positive Δ = candidate loses">
          <table className="tbl">
            <thead>
              <tr>
                <th>County</th>
                <th>Group</th>
                <th className="num">Champion</th>
                <th className="num">Challenger</th>
                <th className="num">Δ</th>
              </tr>
            </thead>
            <tbody>
              {worstCounties.map(c => (
                <tr key={c.fips}>
                  <td><strong>{c.name}</strong></td>
                  <td><Tag>{c.group.replace("_", "/")}</Tag></td>
                  <td className="num">{c.champion_mape.toFixed(2)}</td>
                  <td className="num">{c.challenger_mape.toFixed(2)}</td>
                  <td className="num neg">+{c.delta_mape.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>

      {/* Cross-segment heatmap (sparklines + per-segment) */}
      <Panel title="Cross-segment heat" crumbs="county MAPE distribution"
             sub="how segment medians shifted vs champion">
        <table className="tbl">
          <thead>
            <tr>
              <th>Segment</th>
              <th className="num">N counties</th>
              <th className="num">Champion median</th>
              <th className="num">Challenger median</th>
              <th className="num">Mean Δ</th>
              <th>Distribution shift (Δ histogram)</th>
            </tr>
          </thead>
          <tbody>
            {["bakken", "urban_college", "rural", "reservation"].map(g => {
              const cs = data.counties.filter(c => c.group === g);
              const champMed = cs.map(c => c.champion_mape).sort((a,b)=>a-b)[Math.floor(cs.length/2)];
              const challMed = cs.map(c => c.challenger_mape).sort((a,b)=>a-b)[Math.floor(cs.length/2)];
              const meanD = cs.reduce((s,c)=>s+c.delta_mape, 0) / cs.length;
              // delta distribution bucketed -2..+2
              const bins = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];
              const counts = new Array(bins.length).fill(0);
              cs.forEach(c => {
                const b = bins.findIndex(b => c.delta_mape <= b);
                counts[b === -1 ? bins.length - 1 : b]++;
              });
              const max = Math.max(...counts) || 1;
              return (
                <tr key={g}>
                  <td><strong style={{ textTransform: "capitalize" }}>{g.replace("_", " / ")}</strong></td>
                  <td className="num">{cs.length}</td>
                  <td className="num">{champMed.toFixed(2)}</td>
                  <td className="num">{challMed.toFixed(2)}</td>
                  <td className={`num ${meanD < 0 ? "pos" : "neg"}`}>{meanD > 0 ? "+" : ""}{meanD.toFixed(2)}</td>
                  <td>
                    <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 28 }}>
                      {counts.map((c, i) => (
                        <div key={i} title={`${bins[i]} pp: ${c} counties`}
                          style={{
                            width: 12,
                            height: `${(c / max) * 100}%`,
                            background: bins[i] < 0 ? "var(--accent)" : bins[i] > 0 ? "var(--bad)" : "var(--ink-5)",
                            opacity: c === 0 ? 0.1 : 0.85,
                            minHeight: 1,
                          }} />
                      ))}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>
    </>
  );
}

/* ============================================================
   Deep Search — runs, candidates, journal
   ============================================================ */
function DeepSearch({ data, goto }) {
  const ds = data.deep_search;
  const showIDs = window.useIDMode();
  return (
    <>
      <SectionHead eyebrow="Background process" title="Deep search"
        sub={"convergence-window sweep · " + ds.cores.allocated + " cores · " + ds.eta + (showIDs ? " · pack " + ds.pack : "")}>
        <div style={{ display: "flex", gap: 6 }}>
          <button className="btn ghost sm">Pause</button>
          <button className="btn danger sm">Stop</button>
        </div>
      </SectionHead>

      <div className="kpi-row">
        <KPI k="Runs completed" v={`${ds.progress.completed} / ${ds.progress.planned}`} d="bounded budget · 22 plans" />
        <KPI k="Candidates found" v={ds.progress.candidates_found} d="passing gate-clean policy" />
        <KPI k="Pending review" v={ds.progress.candidates_pending_review} d="needs human eyes" />
        <KPI k="Best Δ overall" v="−0.34 pp" d={showIDs ? "cf001-recent-window-7" : "recent-window sweep, slot 7"}
             delta="−1.42 pp urban" deltaKind="good" />
      </div>

      <div className="row-2-3">
        <Panel title="Frontier candidates" crumbs="ordered by Δ county_mape_overall"
               sub="gate-clean only">
          <table className="tbl">
            <thead>
              <tr>
                <th>Candidate</th>
                <th className="num">Δ overall</th>
                <th className="num">Δ urban</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {ds.leaders.map((l, i) => (
                <tr key={l.id}>
                  <td>
                    {showIDs
                      ? <code>{l.id}</code>
                      : <>{["Recent-window sweep, slot 7",
                            "Mortality-factor sweep, slot 3",
                            "Recent-window × college-fix interaction, slot 2"][i]}</>}
                  </td>
                  <td className="num pos">{l.delta_overall.toFixed(2)}</td>
                  <td className="num pos">{l.delta_urban.toFixed(2)}</td>
                  <td><Pill kind={l.status}>{l.status.replace(/_/g, " ")}</Pill></td>
                  <td><button className="btn ghost sm">Inspect ›</button></td>
                </tr>
              ))}
            </tbody>
          </table>
          <hr className="rule" />
          <Eyebrow>Plateau detection</Eyebrow>
          <table className="tbl" style={{ marginTop: 4 }}>
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Slope</th>
                <th>Recommendation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{showIDs ? <code>recent_window_years</code> : "Recent-origin window length"}</td>
                <td className="num">−0.18</td>
                <td><Pill kind="needs_human_review">explore further</Pill></td>
              </tr>
              <tr>
                <td>{showIDs ? <code>mortality_factor</code> : "Mortality improvement factor"}</td>
                <td className="num">−0.04</td>
                <td><Pill kind="inconclusive">near plateau</Pill></td>
              </tr>
              <tr>
                <td>{showIDs ? <code>convergence_window</code> : "Cohort convergence window"}</td>
                <td className="num">+0.01</td>
                <td><Pill kind="inconclusive">deprioritise</Pill></td>
              </tr>
            </tbody>
          </table>
        </Panel>

        <Panel title="Search journal" crumbs="streaming" sub="latest events first">
          <div style={{ fontFamily: "var(--font-mono, ui-monospace), monospace", fontSize: 11.5 }}>
            {ds.journal.map((j, i) => {
              const friendly = [
                "Sandbox 2 / run 11 completed — passed all gates",
                "Spawned a new recent-window variant (sandbox 3)",
                "Frontier updated: 3 candidates above plateau threshold",
                "Sandbox 1 / run 9 inconclusive — operational warning (runtime 47m)",
                "Convergence-window boundary scan complete; tier-2 sweeps queued",
              ][i] || j.msg;
              return (
                <div key={i} style={{
                  display: "grid", gridTemplateColumns: "44px 60px 1fr", gap: 8,
                  padding: "4px 0", borderBottom: "1px dashed var(--line-2)",
                }}>
                  <span style={{ color: "var(--ink-4)" }}>{j.t}</span>
                  <Pill kind={j.level === "warning" ? "needs_human_review" : "passed_all_gates"}>{j.level}</Pill>
                  <span style={{ color: "var(--ink-2)" }}>{showIDs ? j.msg : friendly}</span>
                </div>
              );
            })}
          </div>
          <hr className="rule" />
          <Eyebrow>Active sandboxes</Eyebrow>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginTop: 6 }}>
            {[1, 2, 3].map(i => {
              const variant = [
                "recent-window 9",
                "mortality-factor 4",
                "interaction 12",
              ][i-1];
              const techId = ["cf001-recent-window-9","cf001-mortality-factor-4","cf001-interaction-12"][i-1];
              return (
                <div key={i} style={{
                  border: "1px solid var(--line)", borderRadius: 3, padding: 8,
                  background: "#fff", fontSize: 11.5,
                }}>
                  <div style={{ fontWeight: 600, fontSize: 12 }}>Sandbox {i}</div>
                  <div style={{ color: "var(--ink-3)" }}>4 workers · {showIDs ? techId : variant}</div>
                  <Pill kind={i === 1 ? "needs_human_review" : "running"}>{i === 1 ? "review" : "running"}</Pill>
                </div>
              );
            })}
          </div>
        </Panel>
      </div>

      <Panel title="Search policy" crumbs={showIDs ? "config/observatory_search_policy.yaml" : "how the search keeps itself safe"} sub="gate-clean statuses only">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, fontSize: 12 }}>
          <div>
            <Eyebrow>Plateau threshold</Eyebrow>
            <div style={{ fontWeight: 600 }}>0.02 pp / step</div>
            <div className="small" style={{ color: "var(--ink-4)" }}>parameters below this are deprioritised</div>
          </div>
          <div>
            <Eyebrow>Hard gate ceilings</Eyebrow>
            <div style={{ fontWeight: 600 }}>per-metric (policy file)</div>
            <div className="small" style={{ color: "var(--ink-4)" }}>bounded by evaluation_policy.yaml</div>
          </div>
          <div>
            <Eyebrow>Max parallel runs</Eyebrow>
            <div style={{ fontWeight: 600 }}>3 (12-core budget)</div>
            <div className="small" style={{ color: "var(--ink-4)" }}>4 workers per run</div>
          </div>
          <div>
            <Eyebrow>AI synthesis</Eyebrow>
            <div style={{ fontWeight: 600 }}>off · advisory only</div>
            <div className="small" style={{ color: "var(--ink-4)" }}>cannot set decision state</div>
          </div>
        </div>
      </Panel>
    </>
  );
}

/* ============================================================
   History — promotions / decision timeline
   ============================================================ */
function History({ data, goto }) {
  const showIDs = window.useIDMode();
  return (
    <>
      <SectionHead eyebrow="Promotion ledger" title="History"
        sub="every production change, decision, and revalidation in order" />

      <Panel title="Production timeline" sub="newest first">
        <table className="tbl">
          <thead>
            <tr>
              <th>Date</th>
              <th>Version</th>
              <th>Action</th>
              <th>Note</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {data.history.map((h, i) => (
              <tr key={i}>
                <td><strong>{h.date}</strong></td>
                <td>
                  <strong>{h.title}</strong>
                  {showIDs && <div style={{ fontSize: 10.5, color: "var(--ink-4)", fontFamily: "var(--font-mono), Consolas, monospace" }}>{h.method}</div>}
                </td>
                <td><Pill kind={h.action}>{h.action.replace(/_/g, " ")}</Pill></td>
                <td>{h.note}</td>
                <td><button className="btn ghost sm">Open record ›</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <div className="row-2">
        <Panel title="Promotion cadence" crumbs="last 12 months">
          <Sparkline values={[3, 2, 4, 3, 2, 5, 3, 4, 2, 1, 4, 3]} color="var(--primary)" fill="var(--primary)" height={48} width={460} />
          <div style={{ fontSize: 11, color: "var(--ink-4)", marginTop: 4 }}>
            Benchmarks per month · monthly average 3.0
          </div>
        </Panel>
        <Panel title="Approval ratio" crumbs="last 12 months">
          <div style={{ display: "flex", gap: 14 }}>
            <div>
              <Eyebrow>Promoted</Eyebrow>
              <div style={{ fontSize: 24, fontWeight: 600 }}>11 <span style={{ fontSize: 12, color: "var(--ink-4)" }}>/ 18</span></div>
            </div>
            <div>
              <Eyebrow>Rejected</Eyebrow>
              <div style={{ fontSize: 24, fontWeight: 600 }}>3</div>
            </div>
            <div>
              <Eyebrow>More runs</Eyebrow>
              <div style={{ fontSize: 24, fontWeight: 600 }}>4</div>
            </div>
          </div>
          <hr className="rule" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4, height: 22 }}>
            <div style={{ background: "var(--primary)", color: "#fff", fontSize: 10, padding: "0 4px", display: "grid", placeItems: "center" }}>61% promoted</div>
            <div style={{ background: "var(--warn)", color: "#000", fontSize: 10, padding: "0 4px", display: "grid", placeItems: "center" }}>22% more runs</div>
            <div style={{ background: "var(--bad)", color: "#fff", fontSize: 10, padding: "0 4px", display: "grid", placeItems: "center" }}>17% rejected</div>
          </div>
        </Panel>
      </div>
    </>
  );
}

Object.assign(window, { Scorecard, DeepSearch, History });
