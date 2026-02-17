import React, { useState, useMemo, useCallback } from "react";

const W = 700, H = 400, PAD = { t: 40, r: 30, b: 50, l: 60 };
const PW = W - PAD.l - PAD.r, PH = H - PAD.t - PAD.b;

const sigmoid = (scale, rate, z) => scale * (1 - Math.exp(-rate * z));
const mu = (base, scale, rate, z) => base + sigmoid(scale, rate, z);

// Find alternative params that match truth at two z-points within a band
function findAlternatives(trueBase, trueScale, trueRate, zCenter, zWidth, count = 5) {
  const zLo = zCenter - zWidth / 2;
  const zHi = zCenter + zWidth / 2;
  const yLo = mu(trueBase, trueScale, trueRate, Math.max(zLo, 0.01));
  const yHi = mu(trueBase, trueScale, trueRate, zHi);
  
  const alts = [];
  const baseOffsets = [-0.3, -0.15, 0.15, 0.3, -0.45, 0.45, -0.6, 0.6];
  
  for (let k = 0; k < baseOffsets.length && alts.length < count; k++) {
    const db = baseOffsets[k];
    const newBase = trueBase + db;
    const sLo = sigmoid(1, trueRate, Math.max(zLo, 0.01));
    const sHi = sigmoid(1, trueRate, zHi);
    
    // Try different rates
    for (let rMul = 0.3; rMul <= 3.0; rMul += 0.1) {
      const newRate = trueRate * rMul;
      const s1 = 1 - Math.exp(-newRate * Math.max(zLo, 0.01));
      const s2 = 1 - Math.exp(-newRate * zHi);
      
      if (Math.abs(s2 - s1) < 1e-6) continue;
      
      // Solve: newBase + newScale * s1 = yLo, newBase + newScale * s2 = yHi
      const newScale = (yHi - yLo) / (s2 - s1);
      const checkBase = yLo - newScale * s1;
      
      if (newScale < 0 || newScale > 5 || newRate < 0.1 || newRate > 10) continue;
      if (Math.abs(checkBase - newBase) > 0.3) continue;
      
      // Use checkBase for exact fit at endpoints
      const errLo = Math.abs(mu(checkBase, newScale, newRate, Math.max(zLo, 0.01)) - yLo);
      const errHi = Math.abs(mu(checkBase, newScale, newRate, zHi) - yHi);
      
      if (errLo < 0.01 && errHi < 0.01) {
        // Check it's actually different
        const paramDist = Math.abs(newScale - trueScale) + Math.abs(newRate - trueRate);
        if (paramDist > 0.1) {
          alts.push({ base: checkBase, scale: newScale, rate: newRate });
          break;
        }
      }
    }
  }
  return alts;
}

// Compute max divergence outside band
function maxDivergence(trueB, trueS, trueR, altB, altS, altR, zMin, zMax, zBandLo, zBandHi) {
  let maxDiv = 0;
  for (let z = zMin; z <= zMax; z += 0.02) {
    if (z >= zBandLo && z <= zBandHi) continue;
    const diff = Math.abs(mu(altB, altS, altR, z) - mu(trueB, trueS, trueR, z));
    if (diff > maxDiv) maxDiv = diff;
  }
  return maxDiv;
}

const COLORS = {
  bg: "#0a0e17",
  panel: "#111827",
  grid: "#1e293b",
  text: "#94a3b8",
  textBright: "#e2e8f0",
  accent: "#22d3ee",
  truth: "#22d3ee",
  band: "rgba(251, 191, 36, 0.12)",
  bandBorder: "rgba(251, 191, 36, 0.5)",
  alts: ["#f472b6", "#a78bfa", "#fb923c", "#4ade80", "#f87171"],
};

export default function RidgeViz() {
  const [zCenter, setZCenter] = useState(1.5);
  const [zWidth, setZWidth] = useState(0.5);
  const [showAlts, setShowAlts] = useState(true);
  
  const trueBase = -1.0, trueScale = 0.5, trueRate = 1.0;
  const zMin = 0, zMax = 3.0;
  
  const zBandLo = Math.max(zCenter - zWidth / 2, 0.01);
  const zBandHi = zCenter + zWidth / 2;
  
  const alts = useMemo(() => 
    findAlternatives(trueBase, trueScale, trueRate, zCenter, zWidth, 4),
    [zCenter, zWidth]
  );
  
  // Compute y range
  const yMin = -1.8, yMax = 0.2;
  const yRange = yMax - yMin;
  
  const toX = useCallback((z) => PAD.l + (z - zMin) / (zMax - zMin) * PW, []);
  const toY = useCallback((y) => PAD.t + (1 - (y - yMin) / yRange) * PH, []);
  
  // Generate curve paths
  const makePath = useCallback((base, scale, rate) => {
    const pts = [];
    for (let z = zMin; z <= zMax; z += 0.02) {
      pts.push(`${z === zMin ? 'M' : 'L'}${toX(z).toFixed(1)},${toY(mu(base, scale, rate, z)).toFixed(1)}`);
    }
    return pts.join(' ');
  }, [toX, toY]);
  
  const truthPath = useMemo(() => makePath(trueBase, trueScale, trueRate), [makePath]);
  const altPaths = useMemo(() => alts.map(a => makePath(a.base, a.scale, a.rate)), [alts, makePath]);
  
  // Band error: max difference within band between truth and alts
  const bandErrors = useMemo(() => alts.map(a => {
    let maxErr = 0;
    for (let z = zBandLo; z <= zBandHi; z += 0.01) {
      const diff = Math.abs(mu(a.base, a.scale, a.rate, z) - mu(trueBase, trueScale, trueRate, z));
      if (diff > maxErr) maxErr = diff;
    }
    return maxErr;
  }), [alts, zBandLo, zBandHi]);
  
  const outsideErrors = useMemo(() => alts.map(a => 
    maxDivergence(trueBase, trueScale, trueRate, a.base, a.scale, a.rate, zMin, zMax, zBandLo, zBandHi)
  ), [alts, zBandLo, zBandHi]);
  
  // Grid lines
  const yTicks = [-1.5, -1.0, -0.5, 0.0];
  const zTicks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

  return (
    <div style={{
      background: COLORS.bg,
      minHeight: "100vh",
      padding: "24px",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
      color: COLORS.text,
    }}>
      <div style={{ maxWidth: 760, margin: "0 auto" }}>
        {/* Title */}
        <div style={{ marginBottom: 20 }}>
          <h1 style={{
            fontSize: 18, fontWeight: 700, color: COLORS.textBright,
            margin: 0, letterSpacing: "-0.02em"
          }}>
            μ(z) Ridge Problem
          </h1>
          <p style={{ fontSize: 12, margin: "6px 0 0", color: COLORS.text, lineHeight: 1.5 }}>
            Within a narrow z-band (yellow), many different (μ_base, μ_scale, μ_rate) 
            produce <span style={{color: COLORS.bandBorder}}>indistinguishable</span> curves.
            The likelihood ridge makes these parameters unidentifiable from a single window.
          </p>
        </div>

        {/* Main plot */}
        <svg width={W} height={H} style={{ background: COLORS.panel, borderRadius: 8, display: "block" }}>
          {/* Grid */}
          {yTicks.map(y => (
            <g key={y}>
              <line x1={PAD.l} y1={toY(y)} x2={W - PAD.r} y2={toY(y)} 
                    stroke={COLORS.grid} strokeWidth={1} />
              <text x={PAD.l - 8} y={toY(y) + 4} textAnchor="end" 
                    fill={COLORS.text} fontSize={10}>{y.toFixed(1)}</text>
            </g>
          ))}
          {zTicks.map(z => (
            <g key={z}>
              <line x1={toX(z)} y1={PAD.t} x2={toX(z)} y2={H - PAD.b} 
                    stroke={COLORS.grid} strokeWidth={1} />
              <text x={toX(z)} y={H - PAD.b + 16} textAnchor="middle" 
                    fill={COLORS.text} fontSize={10}>{z.toFixed(1)}</text>
            </g>
          ))}
          
          {/* Axis labels */}
          <text x={W / 2} y={H - 6} textAnchor="middle" fill={COLORS.text} fontSize={11}>
            z (latent volatility state)
          </text>
          <text x={14} y={H / 2} textAnchor="middle" fill={COLORS.text} fontSize={11}
                transform={`rotate(-90, 14, ${H / 2})`}>
            μ(z)
          </text>
          
          {/* Z-band highlight */}
          <rect x={toX(zBandLo)} y={PAD.t} 
                width={toX(zBandHi) - toX(zBandLo)} height={PH}
                fill={COLORS.band} />
          <line x1={toX(zBandLo)} y1={PAD.t} x2={toX(zBandLo)} y2={H - PAD.b}
                stroke={COLORS.bandBorder} strokeWidth={1.5} strokeDasharray="4,3" />
          <line x1={toX(zBandHi)} y1={PAD.t} x2={toX(zBandHi)} y2={H - PAD.b}
                stroke={COLORS.bandBorder} strokeWidth={1.5} strokeDasharray="4,3" />
          <text x={(toX(zBandLo) + toX(zBandHi)) / 2} y={PAD.t + 14}
                textAnchor="middle" fill={COLORS.bandBorder} fontSize={9} fontWeight={600}>
            WINDOW z-RANGE
          </text>
          
          {/* Alternative curves */}
          {showAlts && altPaths.map((path, i) => (
            <path key={i} d={path} fill="none" stroke={COLORS.alts[i]} 
                  strokeWidth={1.8} opacity={0.7} strokeDasharray="6,4" />
          ))}
          
          {/* Truth curve (on top) */}
          <path d={truthPath} fill="none" stroke={COLORS.truth} strokeWidth={2.5} />
          
          {/* Legend */}
          <g transform={`translate(${W - PAD.r - 145}, ${PAD.t + 8})`}>
            <rect x={-6} y={-4} width={148} height={showAlts ? 18 + alts.length * 16 : 22} 
                  rx={4} fill="rgba(0,0,0,0.5)" />
            <line x1={0} y1={6} x2={20} y2={6} stroke={COLORS.truth} strokeWidth={2.5} />
            <text x={26} y={10} fill={COLORS.textBright} fontSize={10}>
              Truth (−1.0 + 0.5·sig)
            </text>
            {showAlts && alts.map((a, i) => (
              <g key={i} transform={`translate(0, ${16 + i * 16})`}>
                <line x1={0} y1={6} x2={20} y2={6} stroke={COLORS.alts[i]} 
                      strokeWidth={1.8} strokeDasharray="6,4" opacity={0.7} />
                <text x={26} y={10} fill={COLORS.alts[i]} fontSize={9} opacity={0.85}>
                  ({a.base.toFixed(2)}, {a.scale.toFixed(2)}, {a.rate.toFixed(2)})
                </text>
              </g>
            ))}
          </g>
        </svg>

        {/* Controls */}
        <div style={{
          display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16,
          marginTop: 16, padding: 16,
          background: COLORS.panel, borderRadius: 8,
        }}>
          <div>
            <label style={{ fontSize: 11, color: COLORS.text, display: "block", marginBottom: 6 }}>
              Z-Band Center: <span style={{ color: COLORS.textBright }}>{zCenter.toFixed(2)}</span>
            </label>
            <input type="range" min={0.3} max={2.7} step={0.05} value={zCenter}
                   onChange={e => setZCenter(+e.target.value)}
                   style={{ width: "100%", accentColor: COLORS.accent }} />
          </div>
          <div>
            <label style={{ fontSize: 11, color: COLORS.text, display: "block", marginBottom: 6 }}>
              Z-Band Width: <span style={{ color: COLORS.textBright }}>{zWidth.toFixed(2)}</span>
              <span style={{ color: COLORS.text, fontSize: 10 }}> (ρ=0.95 window ≈ 0.5)</span>
            </label>
            <input type="range" min={0.15} max={2.5} step={0.05} value={zWidth}
                   onChange={e => setZWidth(+e.target.value)}
                   style={{ width: "100%", accentColor: COLORS.accent }} />
          </div>
        </div>

        {/* Error table */}
        {showAlts && alts.length > 0 && (
          <div style={{
            marginTop: 16, padding: 16,
            background: COLORS.panel, borderRadius: 8,
            fontSize: 11,
          }}>
            <div style={{ 
              display: "grid", gridTemplateColumns: "24px 1fr 1fr 1fr 90px 90px", 
              gap: "4px 12px", alignItems: "center" 
            }}>
              <div style={{ color: COLORS.text, fontWeight: 600 }}>#</div>
              <div style={{ color: COLORS.text, fontWeight: 600 }}>μ_base</div>
              <div style={{ color: COLORS.text, fontWeight: 600 }}>μ_scale</div>
              <div style={{ color: COLORS.text, fontWeight: 600 }}>μ_rate</div>
              <div style={{ color: COLORS.text, fontWeight: 600, textAlign: "right" }}>In-band err</div>
              <div style={{ color: COLORS.text, fontWeight: 600, textAlign: "right" }}>Outside err</div>
              
              {/* Truth row */}
              <div style={{ color: COLORS.truth }}>●</div>
              <div style={{ color: COLORS.truth }}>{trueBase.toFixed(3)}</div>
              <div style={{ color: COLORS.truth }}>{trueScale.toFixed(3)}</div>
              <div style={{ color: COLORS.truth }}>{trueRate.toFixed(3)}</div>
              <div style={{ color: COLORS.truth, textAlign: "right" }}>—</div>
              <div style={{ color: COLORS.truth, textAlign: "right" }}>—</div>
              
              {alts.map((a, i) => (
                <React.Fragment key={i}>
                  <div style={{ color: COLORS.alts[i] }}>●</div>
                  <div style={{ color: COLORS.alts[i] }}>{a.base.toFixed(3)}</div>
                  <div style={{ color: COLORS.alts[i] }}>{a.scale.toFixed(3)}</div>
                  <div style={{ color: COLORS.alts[i] }}>{a.rate.toFixed(3)}</div>
                  <div style={{ color: bandErrors[i] < 0.02 ? "#4ade80" : "#fb923c", textAlign: "right" }}>
                    {bandErrors[i].toFixed(4)}
                  </div>
                  <div style={{ color: outsideErrors[i] > 0.2 ? "#f87171" : "#4ade80", textAlign: "right" }}>
                    {outsideErrors[i].toFixed(3)}
                  </div>
                </React.Fragment>
              ))}
            </div>
            
            <div style={{ marginTop: 12, fontSize: 10, color: COLORS.text, lineHeight: 1.6 }}>
              <span style={{ color: "#4ade80" }}>Green</span> in-band error = curves are indistinguishable within the window.{" "}
              <span style={{ color: "#f87171" }}>Red</span> outside error = curves diverge massively outside.{" "}
              <strong style={{ color: COLORS.textBright }}>Widen the band</strong> to see the ridge dissolve — 
              at Δz &gt; 1.5, alternatives can no longer hide.
            </div>
          </div>
        )}
        
        {/* Toggle */}
        <div style={{ marginTop: 12, textAlign: "center" }}>
          <button onClick={() => setShowAlts(!showAlts)} style={{
            background: "transparent", border: `1px solid ${COLORS.grid}`,
            color: COLORS.text, padding: "6px 16px", borderRadius: 4,
            cursor: "pointer", fontSize: 11,
          }}>
            {showAlts ? "Hide" : "Show"} alternative curves
          </button>
        </div>
      </div>
    </div>
  );
}
