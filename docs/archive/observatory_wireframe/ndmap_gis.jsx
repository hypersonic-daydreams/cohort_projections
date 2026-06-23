/* global React */
/* ============================================================
   ND county GIS map.
   Loads us-atlas counties-10m.json from unpkg at runtime,
   projects ND through d3-geo, renders real SVG polygons.
   Falls back to the cartogram if the fetch fails (offline).
   ============================================================ */

const NDMapGIS = (function () {
  const { useState, useEffect, useRef, useMemo } = React;

  // Cache the parsed GeoJSON across remounts so panel switches stay snappy.
  let _ndFeatures = null;
  let _ndPromise = null;

  async function loadNDCounties() {
    if (_ndFeatures) return _ndFeatures;
    if (_ndPromise) return _ndPromise;
    _ndPromise = (async () => {
      // topojson-client (~10kb) + us-atlas counties (~1MB) from unpkg
      if (!window.topojson) {
        await new Promise((res, rej) => {
          const s = document.createElement("script");
          s.src = "https://unpkg.com/topojson-client@3/dist/topojson-client.min.js";
          s.onload = res;
          s.onerror = rej;
          document.head.appendChild(s);
        });
      }
      const resp = await fetch("https://unpkg.com/us-atlas@3.0.1/counties-10m.json");
      if (!resp.ok) throw new Error("us-atlas fetch failed: " + resp.status);
      const topo = await resp.json();
      const fc = window.topojson.feature(topo, topo.objects.counties);
      // Filter to ND (FIPS 38xxx)
      _ndFeatures = fc.features.filter(f => String(f.id).startsWith("38"));
      return _ndFeatures;
    })();
    return _ndPromise;
  }

  // Project lon/lat to SVG (simple equirectangular bounded to ND extent).
  // Avoids dragging in d3-geo for a single-state map.
  function buildProjection(features, width, height, padding = 6) {
    // Compute bounding box from all coordinates.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    const walk = (coords, depth = 0) => {
      if (typeof coords[0] === "number") {
        const [x, y] = coords;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      } else {
        coords.forEach(c => walk(c, depth + 1));
      }
    };
    features.forEach(f => walk(f.geometry.coordinates));

    // Adjust for latitude distortion (cosine of mean lat).
    const meanLat = (minY + maxY) / 2;
    const aspect = Math.cos(meanLat * Math.PI / 180);
    const lonRange = (maxX - minX);
    const latRange = (maxY - minY);
    const dataAspect = (lonRange * aspect) / latRange;
    const targetAspect = (width - padding * 2) / (height - padding * 2);

    let sx, sy, ox, oy;
    if (dataAspect > targetAspect) {
      // Width-bound
      sx = (width - padding * 2) / (lonRange * aspect);
      sy = sx * aspect;
      const projH = latRange * sy;
      ox = padding;
      oy = padding + (height - padding * 2 - projH) / 2;
    } else {
      // Height-bound
      sy = (height - padding * 2) / latRange;
      sx = sy / aspect;
      const projW = lonRange * sx * aspect;
      ox = padding + (width - padding * 2 - projW) / 2;
      oy = padding;
    }

    return (lon, lat) => [
      ox + (lon - minX) * sx * aspect,
      oy + (maxY - lat) * sy,
    ];
  }

  function geometryToPath(geometry, proj) {
    const draw = (ring) =>
      ring.map(([lon, lat], i) => {
        const [x, y] = proj(lon, lat);
        return (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
      }).join(" ") + " Z";

    const coords = geometry.coordinates;
    if (geometry.type === "Polygon") {
      return coords.map(draw).join(" ");
    }
    if (geometry.type === "MultiPolygon") {
      return coords.flatMap(poly => poly.map(draw)).join(" ");
    }
    return "";
  }

  function centroid(geometry, proj) {
    let cx = 0, cy = 0, n = 0;
    const walk = (coords) => {
      if (typeof coords[0] === "number") {
        const [x, y] = proj(coords[0], coords[1]);
        cx += x; cy += y; n++;
      } else {
        coords.forEach(walk);
      }
    };
    walk(geometry.coordinates);
    return [cx / n, cy / n];
  }

  // Map FIPS to county data row from OBS_DATA
  function NDMapGIS({ counties, metric, selected, onSelect, width = 720, height = 380, showLabels = false, fallback }) {
    const [features, setFeatures] = useState(_ndFeatures);
    const [error, setError] = useState(null);
    const svgRef = useRef(null);

    useEffect(() => {
      let cancelled = false;
      loadNDCounties()
        .then(fs => { if (!cancelled) setFeatures(fs); })
        .catch(e => { if (!cancelled) setError(e); });
      return () => { cancelled = true; };
    }, []);

    const byFips = useMemo(() => {
      const m = {};
      counties.forEach(c => { m[c.fips] = c; });
      return m;
    }, [counties]);

    const projection = useMemo(() => {
      if (!features) return null;
      return buildProjection(features, width, height);
    }, [features, width, height]);

    if (error && fallback) return fallback;
    if (!features || !projection) {
      return (
        <div style={{ display: "grid", placeItems: "center", height,
                      border: "1px dashed var(--line)", borderRadius: 3,
                      background: "var(--paper-2)", color: "var(--ink-4)",
                      fontSize: 12 }}>
          {error
            ? "Map data unavailable — showing fallback below"
            : "Loading ND county boundaries…"}
        </div>
      );
    }

    // Choose fill per feature
    const fillFor = (c) => {
      if (!c) return "#E6E2D7";
      if (metric === "delta_mape") {
        return colorDelta(c.delta_mape);
      } else if (metric === "bias") {
        return colorDelta(c.signed_bias_challenger / 1.3);
      } else if (metric === "champion_mape") {
        return colorMape(c.champion_mape);
      } else if (metric === "challenger_mape") {
        return colorMape(c.challenger_mape);
      }
      return "#E6E2D7";
    };

    return (
      <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`}
           style={{ width: "100%", height: "auto", display: "block", background: "#FCFBF7" }}>
        {features.map(f => {
          const c = byFips[f.id];
          const path = geometryToPath(f.geometry, projection);
          const isSelected = selected && selected.fips === f.id;
          return (
            <path key={f.id}
              d={path}
              fill={fillFor(c)}
              stroke={isSelected ? "#0F1217" : "rgba(0,0,0,0.4)"}
              strokeWidth={isSelected ? 1.6 : 0.5}
              style={{ cursor: "pointer" }}
              onClick={() => c && onSelect && onSelect(c)}
              onMouseEnter={(e) => e.currentTarget.setAttribute("stroke-width", "1.4")}
              onMouseLeave={(e) => e.currentTarget.setAttribute("stroke-width", isSelected ? "1.6" : "0.5")}>
              <title>{c ? `${c.name} — ${labelForMetric(c, metric)}` : f.properties.name}</title>
            </path>
          );
        })}

        {/* Always-on labels for the biggest / most important counties */}
        {features.map(f => {
          const c = byFips[f.id];
          if (!c) return null;
          const [cx, cy] = centroid(f.geometry, projection);
          const isLarge = c.pop > 25000;
          const isSelected = selected && selected.fips === f.id;
          if (!showLabels && !isLarge && !isSelected) return null;
          // Color: black on light, white on dark
          const fill = fillFor(c);
          const dark = isDarkFill(fill);
          return (
            <g key={"l-" + f.id} pointerEvents="none">
              <text x={cx} y={cy - 1}
                    textAnchor="middle"
                    fontSize="8"
                    fontFamily="var(--font-sans, Segoe UI), Arial, sans-serif"
                    fontWeight="600"
                    fill={dark ? "rgba(255,255,255,0.95)" : "rgba(0,0,0,0.75)"}>
                {c.name}
              </text>
              <text x={cx} y={cy + 8}
                    textAnchor="middle"
                    fontSize="8"
                    fontFamily="var(--font-mono, ui-monospace), Consolas, monospace"
                    fontWeight="700"
                    fill={dark ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.65)"}>
                {labelForMetric(c, metric)}
              </text>
            </g>
          );
        })}
      </svg>
    );
  }

  // Helpers ----------------------------------------------------------------
  function labelForMetric(c, metric) {
    if (metric === "delta_mape") return (c.delta_mape >= 0 ? "+" : "") + c.delta_mape.toFixed(2);
    if (metric === "bias")       return (c.signed_bias_challenger >= 0 ? "+" : "") + c.signed_bias_challenger.toFixed(2);
    if (metric === "champion_mape") return c.champion_mape.toFixed(1);
    if (metric === "challenger_mape") return c.challenger_mape.toFixed(1);
    return "";
  }

  function colorDelta(d) {
    if (d < -1.5)  return "#054550";
    if (d < -0.6)  return "#3A8F9B";
    if (d < -0.1)  return "#9EC4CA";
    if (d <  0.1)  return "#EFEBE0";
    if (d <  0.6)  return "#D6989B";
    if (d <  1.5)  return "#A8353A";
    return "#7E1E22";
  }
  function colorMape(v) {
    const t = Math.max(0, Math.min(1, (v - 5) / 20));
    const palette = ["#EFEBE0", "#CFE0E2", "#9EC4CA", "#6CAAB4", "#3A8F9B", "#085A65"];
    return palette[Math.floor(t * (palette.length - 1))];
  }
  function isDarkFill(hex) {
    const m = hex.match(/^#([0-9a-f]{6})$/i);
    if (!m) return false;
    const n = parseInt(m[1], 16);
    const r = (n >> 16) & 0xff, g = (n >> 8) & 0xff, b = n & 0xff;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    return lum < 128;
  }

  return NDMapGIS;
})();

window.NDMapGIS = NDMapGIS;
