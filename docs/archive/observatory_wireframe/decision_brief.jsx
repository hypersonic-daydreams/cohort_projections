/* global React, Pill, Tag, KPI, SectionHead, Panel, Eyebrow, Sparkline,
   Toggle, NDMap, NDMapLegend, Drawer, fmtDelta, fmtNum */

const DECISION_OPTIONS = [
  { value: "promote", label: "Promote candidate to production",
    desc: "Update the alias; trigger revalidation.",
    btnKind: "primary" },
  { value: "accept_but_do_not_promote", label: "Accept, don't promote yet",
    desc: "Record the win, hold current production; useful when waiting for cohort vintage.",
    btnKind: "accent" },
  { value: "retain_champion", label: "Keep current production",
    desc: "Acknowledge the comparison but leave production in place.",
    btnKind: "ghost" },
  { value: "needs_more_segmentation", label: "Request more runs",
    desc: "Specific segments need additional evidence before sign-off.",
    btnKind: "ghost" },
  { value: "reject", label: "Reject candidate",
    desc: "Candidate fails decision criteria. Record reason; remove from queue.",
    btnKind: "danger" },
];

function DecisionBrief({ data, goto, onCommit }) {
  const sc = data.scorecard;
  const L  = data.labels;
  const v  = sc.verdict;
  const showIDs = window.useIDMode();
  const [choice, setChoice]   = React.useState(v.recommended_action);
  const [notes,  setNotes]    = React.useState("");
  const [reviewer, setReviewer] = React.useState("Kevin Iverson (State Demographer)");
  const [submitted, setSubmitted] = React.useState(false);
  const [showMore, setShowMore] = React.useState(false);

  // Check-list items (real review questions)
  const checklist = [
      { label: "Hard gates clean (negatives, aggregation, registration)", status: "yes",
      detail: "All 53 counties × 30y projection — 0 negative-pop violations; aggregation matches state rollup." },
    { label: "Targeted weakness improved", status: "yes",
      detail: "Urban / college county error 12.86 → 11.68 (−1.18 pp). Matches the ADR-061 college-fix intent." },
    { label: "Recent-origin state error improved", status: "yes",
      detail: "Short window −0.20 pp, medium window −0.93 pp. Long-horizon drift also improves 0.31 pp." },
    { label: "No segment regresses beyond tolerance", status: "no",
      detail: "Bakken +0.52 pp (worst single segment). Within historical noise band for that group, but watch." },
    { label: "Operationally clean (runtime, repro, completeness)", status: "yes",
      detail: "Runtime 31m of 45m budget; reproducibility log present; all 5 artifacts written." },
    { label: "Reviewer sign-off persists to decision record", status: "pending",
      detail: showIDs
        ? "Will write to docs/reviews/benchmark_decisions/" + sc.decision_id + ".md on commit."
        : "Will write to the review folder on commit." },
  ];

  return (
    <>
      <SectionHead eyebrow="Decision evidence" title="Decision brief" variant="decision"
        sub={L.benchmark.title + " · " + L.benchmark.sub} />

      {/* Verdict strip (reuse home pattern, but compressed) */}
      <div className="verdict ready" style={{ gridTemplateColumns: "1fr 280px" }}>
        <div className="verdict__left">
          <div className="verdict__crumbs">
            <span>{L.champion.title}</span><span className="sep">vs</span>
            <span>{L.challenger.title}</span>
            {showIDs && <><span className="sep">·</span><span className="code">{sc.benchmark_run_id}</span></>}
          </div>
          <span className="verdict__status">● {v.user_status_label}</span>
          <div className="verdict__headline">{v.headline}</div>
          <div className="verdict__reason">{v.main_reason}</div>
          <div className="verdict__meta">
            <div><div className="k">Confidence</div><div className="v">{v.confidence}</div></div>
            <div><div className="k">Operational</div><div className="v">{v.operational_label}</div></div>
            <div><div className="k">Record status</div><div className="v">{v.status}</div></div>
          </div>
        </div>
        <div className="verdict__right">
          <div className="right-label">Workflow position</div>
          <div className="signoff-progress">
            <ProgressSteps step={1} steps={["Drafted", "Review", "Sign off", "Promote alias", "Revalidate"]} />
          </div>
          <div style={{ fontSize: 11.5, color: "var(--ink-3)" }}>
            Reviewer hasn't signed off yet. This page commits your decision and writes the .md record.
          </div>
        </div>
      </div>

      {/* What matters most: gain / tradeoff / risk */}
      <div className="row-3" style={{ gridTemplateColumns: "1.4fr 1.4fr 1fr" }}>
        <Panel title="Main gain" crumbs="why you'd say yes">
          <div style={{ fontSize: 14, lineHeight: 1.5 }}>{v.main_gain}</div>
          <hr className="rule" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 6, fontSize: 12 }}>
            <span>Urban / college county MAPE</span>
            <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600 }}>12.86 → <span style={{ color: "#054550" }}>11.68</span></span>
            <span>Counties improved (Δ &lt; 0)</span>
            <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600 }}>
              {data.counties.filter(c => c.delta_mape < -0.05).length} of 53
            </span>
            <span>Recent-origin state APE (med)</span>
            <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600 }}>2.56 → <span style={{ color: "#054550" }}>1.64</span></span>
          </div>
        </Panel>
        <Panel title="Main tradeoff" crumbs="why you might hesitate">
          <div style={{ fontSize: 14, lineHeight: 1.5 }}>{v.main_tradeoff}</div>
          <hr className="rule" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 6, fontSize: 12 }}>
            <span>Bakken MAPE</span>
            <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600 }}>19.09 → <span style={{ color: "#7E1E22" }}>19.61</span></span>
            <span>Counties worsened (Δ &gt; 0.5 pp)</span>
            <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600 }}>
              {data.counties.filter(c => c.delta_mape > 0.5).length} of 53
            </span>
            <span>Worst single county</span>
            <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600 }}>
              {(() => {
                const w = [...data.counties].sort((a,b)=>b.delta_mape - a.delta_mape)[0];
                return `${w.name} (+${w.delta_mape.toFixed(2)})`;
              })()}
            </span>
          </div>
        </Panel>
        <Panel title="Blockers & risks">
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div><Pill kind="passed_all_gates">●</Pill> No hard-gate failures</div>
            <div><Pill kind="passed_all_gates">●</Pill> No operational warnings</div>
            <div><Pill kind="needs_human_review">●</Pill> Bakken delta exceeds noise band — flag for follow-up</div>
            <div><Pill kind="champion">●</Pill> ADR-061 traceability present</div>
          </div>
        </Panel>
      </div>

      {/* Review checklist */}
      <Panel title="Review checklist" sub="answer each before signing off">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {checklist.map((c, i) => (
            <div key={i} style={{
              display: "grid",
              gridTemplateColumns: "22px 1fr",
              gap: 8,
              padding: 8,
              border: "1px solid var(--line)",
              borderRadius: 3,
              background: "#fff",
              alignItems: "start",
            }}>
              <span style={{
                width: 22, height: 22, borderRadius: 3, display: "grid", placeItems: "center",
                background: c.status === "yes" ? "var(--nd-earthy-teal-20)" :
                            c.status === "no"  ? "#F1DCDD" : "var(--paper-2)",
                color:      c.status === "yes" ? "#054550" :
                            c.status === "no"  ? "#5b1a1c" : "var(--ink-4)",
                fontWeight: 700, fontSize: 12,
              }}>
                {c.status === "yes" ? "✓" : c.status === "no" ? "✕" : "?"}
              </span>
              <div>
                <div style={{ fontWeight: 600, fontSize: 12.5 }}>{c.label}</div>
                <div style={{ fontSize: 11.5, color: "var(--ink-3)", marginTop: 2 }}>{c.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* Hard constraints + operational table */}
      <Panel title="Hard constraints & operational gates"
             crumbs="must pass for promotion"
             sub={<button className="btn ghost sm" onClick={() => setShowMore(s => !s)}>{showMore ? "Hide" : "Show"} full constraint set</button>}>
        <table className="tbl">
          <thead>
            <tr>
              <th>Gate</th><th>Type</th><th className="num">Champion</th><th className="num">Challenger</th><th>Result</th>
            </tr>
          </thead>
          <tbody>
            {sc.hard_constraints.slice(0, showMore ? undefined : 3).map((h, i) => (
              <tr key={i}>
                <td>{h.name}</td>
                <td><Tag kind={h.gate === "hard" ? "bad" : ""}>{h.gate}</Tag></td>
                <td className="num">{typeof h.champion === "boolean" ? (h.champion ? "yes" : "no") : String(h.champion)}</td>
                <td className="num">{typeof h.challenger === "boolean" ? (h.challenger ? "yes" : "no") : String(h.challenger)}</td>
                <td><Pill kind="passed_all_gates">clean</Pill></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      {/* Sign-off form */}
      <div style={{ marginTop: 4 }}>
        <SectionHead eyebrow="Persisted decision" title="Sign off"
          sub={showIDs ? ("writes to " + L.decision.file) : "writes to the review folder"} />
      </div>
      <Panel>
        <div className="signoff">
          <div>
            <Eyebrow>Decision</Eyebrow>
            <div className="choice-row" style={{ marginTop: 6 }}>
              {DECISION_OPTIONS.map(opt => (
                <button key={opt.value}
                  className={`choice ${choice === opt.value ? "selected" : ""} ${opt.value === v.recommended_action ? "recommend" : ""}`}
                  onClick={() => setChoice(opt.value)}>
                  {opt.value === v.recommended_action && (
                    <span className="recommend-pill">Recommended</span>
                  )}
                  <span className="label">{opt.label}</span>
                  <span className="desc">{opt.desc}</span>
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 12 }}>
            <div>
              <Eyebrow>Decision rationale</Eyebrow>
              <textarea
                placeholder={
                  choice === "promote"
                    ? "e.g. Challenger fixes the targeted college-county miss without breaking hard constraints. Bakken regression is within group noise band; will track in next benchmark."
                    : choice === "reject"
                    ? "e.g. Bakken regression exceeds tolerance for production alias."
                    : "Why this decision? This text is appended to the .md record verbatim."
                }
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
              />
            </div>
            <div>
              <Eyebrow>Reviewer</Eyebrow>
              <input
                style={{
                  width: "100%", padding: "8px 10px", fontSize: 12.5,
                  border: "1px solid var(--line)", borderRadius: 3,
                  fontFamily: "inherit",
                }}
                value={reviewer}
                onChange={(e) => setReviewer(e.target.value)} />
              <Eyebrow>Reversion plan</Eyebrow>
              <div style={{
                padding: "8px 10px", border: "1px solid var(--line)",
                borderRadius: 3, background: "var(--paper-2)",
                fontSize: 11.5, color: "var(--ink-2)",
              }}>
                Restore production to <strong>{L.champion.title}</strong> if candidate is later reverted. Auto-included in record.
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4 }}>
            <button className="btn primary" disabled={!choice || submitted}
                    onClick={() => setSubmitted(true)}>
              {submitted ? "✓ Decision recorded · alias update queued" : `Commit decision (${DECISION_OPTIONS.find(o => o.value === choice).label})`}
            </button>
            <button className="btn ghost" onClick={() => goto("scorecard")}>Inspect scorecard first ›</button>
            <div style={{ marginLeft: "auto", fontSize: 11, color: "var(--ink-4)" }}>
              {showIDs ? <>Decision ID: <span className="code">{sc.decision_id}</span></>
                       : <>{L.decision.title}</>}
            </div>
          </div>
          {submitted && (
            <div style={{
              marginTop: 4, padding: "10px 12px", border: "1px solid var(--accent)",
              background: "var(--nd-earthy-teal-20)", color: "#054550", borderRadius: 3,
              fontSize: 12,
            }}>
              <strong>Decision recorded.</strong> The {L.decision.title.toLowerCase()} now reads <span className="code">{choice}</span> by <span className="code">{reviewer}</span>.
              {showIDs && <> Saved to <span className="code">{L.decision.id}.md</span>.</>}
              {choice === "promote" && " Production alias update is queued."}
            </div>
          )}
        </div>
      </Panel>
    </>
  );
}

/* Mini progress steps used in the verdict strip */
function ProgressSteps({ step, steps }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${steps.length}, 1fr)`, gap: 2, marginTop: 4 }}>
      {steps.map((s, i) => (
        <div key={s} style={{
          padding: "4px 6px",
          background: i <= step ? "var(--primary)" : "#fff",
          color:      i <= step ? "#fff" : "var(--ink-3)",
          border:     "1px solid var(--line)",
          fontSize: 10.5,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          fontWeight: 600,
          textAlign: "center",
        }}>{s}</div>
      ))}
    </div>
  );
}

window.DecisionBrief = DecisionBrief;
