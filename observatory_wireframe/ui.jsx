/* global React */
const { useState, useMemo, useEffect, useRef } = React;

/* ============================================================
   Shared small components
   ============================================================ */

function Pill({ children, kind = "untested" }) {
  return <span className={`pill ${kind}`}>{children}</span>;
}

function Tag({ children, kind = "" }) {
  return <span className={`tag ${kind}`}>{children}</span>;
}

function Eyebrow({ children }) { return <div className="eyebrow">{children}</div>; }

function KPI({ k, v, d, delta, deltaKind }) {
  return (
    <div className="kpi">
      <div className="k">{k}</div>
      <div className="v">
        <span style={{ whiteSpace: "nowrap" }}>{v}</span>
        {delta !== undefined && <span className={`delta ${deltaKind || "neutral"}`}>{delta}</span>}
      </div>
      {d && <div className="d">{d}</div>}
    </div>
  );
}

function SectionHead({ eyebrow, title, sub, variant = "", children }) {
  return (
    <div className={`section__head ${variant}`}>
      {eyebrow && <div className="eyebrow">{eyebrow}</div>}
      <h2>{title}</h2>
      {sub && <div className="sub">{sub}</div>}
      {children && <div style={{ marginLeft: "auto" }}>{children}</div>}
    </div>
  );
}

function Panel({ title, sub, crumbs, action, footer, padding = "normal", children, style }) {
  return (
    <div className="panel" style={style}>
      {(title || sub || crumbs || action) && (
        <div className="panel__hd">
          {crumbs && <span className="crumbs">{crumbs}</span>}
          {title && <span className="title">{title}</span>}
          {sub && <span className="sub">{sub}</span>}
          {action && <span style={{ marginLeft: "auto" }}>{action}</span>}
        </div>
      )}
      <div className={`panel__bd ${padding === "dense" ? "dense" : ""}`}>{children}</div>
      {footer && <div className="panel__ft">{footer}</div>}
    </div>
  );
}

/* ============================================================
   Sparkline (SVG)
   ============================================================ */
function Sparkline({ values, color = "var(--ink-3)", fill = "none", height = 28, width = 120, dots = false }) {
  if (!values || values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = width / (values.length - 1);
  const pts = values.map((v, i) => [i * step, height - 4 - ((v - min) / range) * (height - 8)]);
  const path = "M " + pts.map(p => p.join(",")).join(" L ");
  const area = fill !== "none"
    ? `M 0,${height} L ${pts.map(p => p.join(",")).join(" L ")} L ${width},${height} Z`
    : null;
  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      {area && <path d={area} fill={fill} opacity="0.18" />}
      <path d={path} fill="none" stroke={color} strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />
      {dots && pts.map((p, i) => (
        <circle key={i} cx={p[0]} cy={p[1]} r="1.8" fill={color} />
      ))}
    </svg>
  );
}

/* ============================================================
   Diverging color scale for choropleth: delta in pp,
   negative = challenger better = teal, positive = challenger worse = brown.
   ============================================================ */
function deltaColor(d, scale = 1.5) {
  // d in pp. Outside ±scale clipped to the strongest band.
  const stops = [
    [-Infinity, -1.5,  "var(--map-good-3)", true ],
    [-1.5, -0.6,       "var(--map-good-2)", true ],
    [-0.6, -0.1,       "var(--map-good-1)", false],
    [-0.1,  0.1,       "var(--map-mid)",    false],
    [ 0.1,  0.6,       "var(--map-bad-1)",  false],
    [ 0.6,  1.5,       "var(--map-bad-2)",  true ],
    [ 1.5,  Infinity,  "var(--map-bad-3)",  true ],
  ];
  for (const [lo, hi, color, dark] of stops) {
    if (d >= lo && d < hi) return { color, dark };
  }
  return { color: "var(--map-mid)", dark: false };
}

/* ============================================================
   ND county cartogram
   ============================================================ */
function NDMap({ counties, metric = "delta_mape", selected, onSelect, height }) {
  // Build an 11×8 grid of cells by integer (col, row); support fractional positions
  // by snapping nearest cell or rendering as a sub-grid.
  const cellMap = {};
  counties.forEach(c => {
    const col = Math.round(c.col);
    const row = Math.round(c.row);
    const key = `${col}-${row}`;
    if (!cellMap[key]) cellMap[key] = [];
    cellMap[key].push(c);
  });

  // For simple cartogram: render each county at its (col,row) cell, dividing
  // the cell vertically if multiple counties share it.
  const cells = [];
  for (let r = 0; r < 7; r++) {
    for (let cc = 0; cc < 11; cc++) {
      const list = cellMap[`${cc}-${r}`] || [];
      cells.push({ row: r, col: cc, list });
    }
  }

  return (
    <div className="ndmap" style={height ? { aspectRatio: "auto", height } : {}}>
      {cells.map(cell => {
        const list = cell.list;
        if (!list.length) {
          return <div key={`${cell.row}-${cell.col}`} className="cell" style={{ background: "transparent", border: "1px dashed transparent" }} />;
        }
        if (list.length === 1) {
          const c = list[0];
          let value, label;
          if (metric === "delta_mape") {
            value = c.delta_mape;
            label = (value >= 0 ? "+" : "") + value.toFixed(1);
          } else if (metric === "champion_mape") {
            value = c.champion_mape;
            label = value.toFixed(1);
          } else if (metric === "challenger_mape") {
            value = c.challenger_mape;
            label = value.toFixed(1);
          } else if (metric === "bias") {
            value = c.signed_bias_challenger;
            label = (value >= 0 ? "+" : "") + value.toFixed(1);
          }
          // For absolute MAPE metrics, use a different scale (higher = darker)
          let style = {};
          let dark = false;
          if (metric === "delta_mape") {
            const { color, dark: d } = deltaColor(value);
            style.background = color; dark = d;
          } else if (metric === "bias") {
            // -2..+2 diverging
            const sval = Math.max(-2, Math.min(2, value));
            const { color, dark: d } = deltaColor(sval / 1.3);
            style.background = color; dark = d;
          } else {
            // absolute MAPE: 5%→light, 25%→dark teal
            const t = Math.max(0, Math.min(1, (value - 5) / 20));
            const palette = ["#EFEBE0","#CFE0E2","#9EC4CA","#6CAAB4","#3A8F9B","#085A65"];
            style.background = palette[Math.floor(t * (palette.length - 1))];
            dark = t > 0.55;
          }
          const isSelected = selected && selected.fips === c.fips;
          return (
            <div key={c.fips}
              className={`cell ${dark ? "dark" : ""} ${isSelected ? "is-selected" : ""}`}
              style={style}
              onClick={() => onSelect && onSelect(c)}
              title={`${c.name} — ${label}`}>
              <div className="lbl">
                <div>{c.name.length > 8 ? c.name.slice(0, 7) + "…" : c.name}</div>
                <div style={{ opacity: 0.85, fontWeight: 700 }}>{label}</div>
              </div>
            </div>
          );
        }
        // Multiple counties share a cell: stack them vertically
        return (
          <div key={`${cell.row}-${cell.col}`} style={{ display: "grid", gridTemplateRows: `repeat(${list.length}, 1fr)`, gap: 2 }}>
            {list.map(c => {
              const value = c.delta_mape;
              const { color, dark } = deltaColor(value);
              const label = (value >= 0 ? "+" : "") + value.toFixed(1);
              return (
                <div key={c.fips}
                  className={`cell ${dark ? "dark" : ""} ${selected && selected.fips === c.fips ? "is-selected" : ""}`}
                  style={{ background: color }}
                  onClick={() => onSelect && onSelect(c)}>
                  <div className="lbl" style={{ fontSize: 8 }}>
                    <div>{c.name}</div>
                    <div style={{ fontWeight: 700 }}>{label}</div>
                  </div>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

function NDMapLegend({ metric }) {
  if (metric === "delta_mape" || metric === "bias") {
    return (
      <div className="ndmap-legend">
        <span>Challenger better</span>
        <div className="swatches">
          <span style={{ background: "var(--map-good-3)" }} />
          <span style={{ background: "var(--map-good-2)" }} />
          <span style={{ background: "var(--map-good-1)" }} />
          <span style={{ background: "var(--map-mid)" }} />
          <span style={{ background: "var(--map-bad-1)" }} />
          <span style={{ background: "var(--map-bad-2)" }} />
          <span style={{ background: "var(--map-bad-3)" }} />
        </div>
        <span>Challenger worse</span>
        <span style={{ marginLeft: 8, color: "var(--ink-4)" }}>−1.5 pp to +1.5 pp</span>
      </div>
    );
  }
  return (
    <div className="ndmap-legend">
      <span>Low MAPE</span>
      <div className="swatches">
        <span style={{ background: "#EFEBE0" }} />
        <span style={{ background: "#CFE0E2" }} />
        <span style={{ background: "#9EC4CA" }} />
        <span style={{ background: "#6CAAB4" }} />
        <span style={{ background: "#3A8F9B" }} />
        <span style={{ background: "#085A65" }} />
      </div>
      <span>High MAPE</span>
    </div>
  );
}

/* ============================================================
   Toggle row
   ============================================================ */
function Toggle({ options, value, onChange }) {
  return (
    <div className="toggle-row">
      {options.map(opt => (
        <button key={opt.value}
          className={value === opt.value ? "active" : ""}
          onClick={() => onChange(opt.value)}>
          {opt.label}
        </button>
      ))}
    </div>
  );
}

/* ============================================================
   Drawer
   ============================================================ */
function Drawer({ title, onClose, children, footer, width }) {
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="drawer" style={width ? { width } : {}}>
        <div className="drawer__hd">
          <strong>{title}</strong>
          <button className="btn ghost sm x" onClick={onClose}>Close ✕</button>
        </div>
        <div className="drawer__bd">{children}</div>
        {footer && <div className="drawer__ft">{footer}</div>}
      </div>
    </>
  );
}

/* ============================================================
   Pretty delta cell
   ============================================================ */
function fmtDelta(v, suffix = " pp", lowerBetter = true) {
  if (v === null || v === undefined) return <span className="muted">—</span>;
  const sign = v > 0 ? "+" : "";
  const good = lowerBetter ? v < 0 : v > 0;
  const cls = Math.abs(v) < 0.005 ? "muted" : (good ? "pos" : "neg");
  return <span className={cls}>{sign}{v.toFixed(3)}{suffix}</span>;
}

function fmtNum(v, digits = 3) {
  if (v === null || v === undefined || isNaN(v)) return <span className="muted">—</span>;
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v !== "number") return String(v);
  return v.toFixed(digits);
}

/* expose everything used by other files */
Object.assign(window, {
  Pill, Tag, Eyebrow, KPI, SectionHead, Panel,
  Sparkline, NDMap, NDMapLegend, Toggle, Drawer,
  deltaColor, fmtDelta, fmtNum,
});
