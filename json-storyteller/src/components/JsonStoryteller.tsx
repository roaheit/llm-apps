import React, { useState } from "react";
import { useJsonStoryteller } from "../hooks/useJsonStoryteller";
import type { JsonStorytellerProps, StoryTone } from "../types";

const TONES: { id: StoryTone; label: string; icon: string; desc: string }[] = [
  { id: "narrative", label: "Narrative", icon: "📖", desc: "Story-like prose" },
  { id: "analyst",   label: "Analyst",   icon: "📊", desc: "Data-driven report" },
  { id: "casual",    label: "Casual",    icon: "💬", desc: "Like texting a friend" },
  { id: "poetic",    label: "Poetic",    icon: "✨", desc: "Lyrical and vivid" },
];

const DEFAULT_PLACEHOLDER = JSON.stringify(
  { user: { name: "Ada Lovelace", role: "engineer", joined: "2021-04-01", logins: 312 } },
  null, 2
);

export function JsonStoryteller({
  llm,
  data,
  tone: toneProp = "narrative",
  onStoryGenerated,
  onError,
  className,
  style,
  headless = false,
  placeholder = DEFAULT_PLACEHOLDER,
  theme = "dark",
}: JsonStorytellerProps) {
  const isDark = theme === "dark";

  const [tone, setTone]           = useState<StoryTone>(toneProp);
  const [jsonInput, setJsonInput] = useState<string>(
    data ? (typeof data === "string" ? data : JSON.stringify(data, null, 2)) : ""
  );
  const [jsonError, setJsonError] = useState("");
  const [copied, setCopied]       = useState(false);

  const { narrate, story, loading, error } = useJsonStoryteller({ llm, tone, onStoryGenerated, onError });

  const validateAndSet = (val: string) => {
    setJsonInput(val);
    if (!val.trim()) { setJsonError(""); return; }
    try { JSON.parse(val); setJsonError(""); }
    catch (e) { setJsonError((e as Error).message); }
  };

  const handleGenerate = () => {
    if (jsonError || !jsonInput.trim()) return;
    try { narrate(JSON.parse(jsonInput)); }
    catch { narrate(jsonInput); }
  };

  const copyStory = () => {
    navigator.clipboard.writeText(story);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const C = {
    bg:         isDark ? "#080c14" : "#f8f9fb",
    surface:    isDark ? "#0d1424" : "#ffffff",
    surfaceAlt: isDark ? "#0a1020" : "#f1f3f7",
    border:     isDark ? "#ffffff0d" : "#e2e8f0",
    text:       isDark ? "#e2e8f0"  : "#1e293b",
    muted:      "#64748b",
    lineNum:    isDark ? "#ffffff18" : "#00000018",
    error:      "#f87171",
  };

  if (headless) {
    return (
      <div className={className} style={style}>
        {story && <p style={{ fontFamily:"'Georgia',serif", fontSize:"18px", lineHeight:1.85, color:C.text, margin:0 }}>{story}</p>}
        {error && <p style={{ color:C.error, fontSize:"14px" }}>{error.message}</p>}
      </div>
    );
  }

  return (
    <div className={className} style={{ background:C.bg, fontFamily:"'DM Sans',system-ui,sans-serif", color:C.text, borderRadius:"16px", padding:"32px", maxWidth:"800px", width:"100%", display:"grid", gap:"20px", border:`1px solid ${C.border}`, ...style }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Fraunces:ital,wght@0,300;0,600;1,300&family=JetBrains+Mono:wght@400;500&display=swap');
        *{box-sizing:border-box}
        .jst-ta{width:100%;background:transparent;border:none;outline:none;resize:none;font-family:'JetBrains Mono',monospace;font-size:13px;line-height:1.7;caret-color:#00ffe7;padding:0;min-height:160px;color:${C.text}}
        .jst-tone{cursor:pointer;border:1px solid ${C.border};background:${C.surface};border-radius:10px;padding:9px 13px;display:flex;align-items:center;gap:8px;transition:all 0.18s;flex:1;min-width:105px;}
        .jst-tone:hover{border-color:#00ffe740;}
        .jst-tone.on{border-color:#00ffe7;background:#00ffe710;box-shadow:0 0 14px #00ffe718;}
        .jst-btn{cursor:pointer;border:none;background:linear-gradient(135deg,#00ffe7,#a78bfa);color:#080c14;font-family:'DM Sans',sans-serif;font-size:14px;font-weight:600;padding:11px 26px;border-radius:10px;transition:all 0.18s;}
        .jst-btn:hover:not(:disabled){opacity:.88;transform:translateY(-1px);box-shadow:0 8px 24px #00ffe728;}
        .jst-btn:disabled{opacity:.38;cursor:not-allowed;}
        .jst-cp{cursor:pointer;border:1px solid ${C.border};background:transparent;color:${C.muted};font-size:12px;padding:5px 12px;border-radius:6px;transition:all .18s;font-family:inherit;}
        .jst-cp:hover{color:${C.text};}
        @keyframes jst-sh{0%{background-position:-200% center}100%{background-position:200% center}}
        .jst-bar{height:2px;background:linear-gradient(90deg,transparent,#00ffe7,#a78bfa,transparent);background-size:200% auto;animation:jst-sh 1.4s linear infinite;border-radius:1px;}
        @keyframes jst-fi{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        .jst-out{animation:jst-fi .45s ease forwards;}
      `}</style>

      {/* Header */}
      <div>
        <h2 style={{ fontFamily:"'Fraunces',serif", fontSize:"26px", fontWeight:600, margin:"0 0 3px", background:`linear-gradient(135deg,${C.text} 40%,#a78bfa)`, WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>
          JSON Storyteller
        </h2>
        <p style={{ color:C.muted, fontSize:"13px", margin:0 }}>Paste JSON and select a tone to generate a narrative.</p>
      </div>

      {/* JSON Editor */}
      <div style={{ background:C.surface, border:`1px solid ${jsonError?"#f8717130":C.border}`, borderRadius:"12px", overflow:"hidden" }}>
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"9px 14px", background:C.surfaceAlt, borderBottom:`1px solid ${C.border}` }}>
          <div style={{ display:"flex", gap:"5px" }}>
            {["#ff5f57","#febc2e","#28c840"].map(c=><div key={c} style={{ width:"9px",height:"9px",borderRadius:"50%",background:c,opacity:.65 }}/>)}
          </div>
          <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:"11px", color:C.muted }}>input.json</span>
          <button onClick={()=>{ setJsonInput(placeholder); setJsonError(""); }} style={{ background:"none",border:"none",color:C.muted,cursor:"pointer",fontSize:"11px",fontFamily:"inherit" }}>load example</button>
        </div>
        <div style={{ display:"flex", padding:"14px 14px 14px 0" }}>
          <div style={{ color:C.lineNum, fontFamily:"'JetBrains Mono',monospace", fontSize:"13px", lineHeight:"1.7", paddingRight:"12px", paddingLeft:"14px", textAlign:"right", minWidth:"42px", userSelect:"none", borderRight:`1px solid ${C.border}` }}>
            {(jsonInput||placeholder).split("\n").map((_,i)=><div key={i}>{i+1}</div>)}
          </div>
          <textarea className="jst-ta" value={jsonInput} onChange={e=>validateAndSet(e.target.value)} placeholder={placeholder} style={{ paddingLeft:"13px" }} spellCheck={false}/>
        </div>
        {jsonError && <div style={{ padding:"5px 14px 10px", color:C.error, fontSize:"12px", fontFamily:"'JetBrains Mono',monospace" }}>✗ {jsonError}</div>}
      </div>

      {/* Tone selector */}
      <div>
        <p style={{ fontSize:"11px", color:C.muted, letterSpacing:".1em", textTransform:"uppercase", margin:"0 0 9px", fontFamily:"'JetBrains Mono',monospace" }}>Tone</p>
        <div style={{ display:"flex", gap:"8px", flexWrap:"wrap" }}>
          {TONES.map(t=>(
            <button key={t.id} className={`jst-tone ${tone===t.id?"on":""}`} onClick={()=>setTone(t.id)}>
              <span style={{ fontSize:"14px" }}>{t.icon}</span>
              <div>
                <div style={{ fontSize:"12px", fontWeight:500, color:tone===t.id?"#00ffe7":C.text }}>{t.label}</div>
                <div style={{ fontSize:"11px", color:C.muted }}>{t.desc}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Generate button */}
      <div style={{ display:"flex", alignItems:"center", gap:"14px" }}>
        <button className="jst-btn" onClick={handleGenerate} disabled={loading||!!jsonError||!jsonInput.trim()}>
          {loading?"Narrating…":"Generate Story →"}
        </button>
        {loading && <div style={{ flex:1 }}><div className="jst-bar"/></div>}
      </div>

      {error && <div style={{ background:"#f8717110", border:"1px solid #f8717130", borderRadius:"10px", padding:"11px 15px", color:C.error, fontSize:"14px" }}>{error.message}</div>}

      {/* Story output */}
      {story && (
        <div className="jst-out" style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:"12px", overflow:"hidden" }}>
          <div style={{ padding:"9px 14px", background:C.surfaceAlt, borderBottom:`1px solid ${C.border}`, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
            <span style={{ fontSize:"11px", color:"#00ffe7", fontFamily:"'JetBrains Mono',monospace", letterSpacing:".1em", textTransform:"uppercase" }}>
              {TONES.find(t=>t.id===tone)?.icon} {tone} output
            </span>
            <button className="jst-cp" onClick={copyStory}>{copied?"✓ Copied":"Copy"}</button>
          </div>
          <div style={{ padding:"22px" }}>
            <p style={{ fontFamily:"'Fraunces',serif", fontSize:"17px", lineHeight:1.85, color:C.text, margin:0, fontWeight:300, fontStyle:tone==="poetic"?"italic":"normal" }}>
              {story}
            </p>
          </div>
          <div style={{ height:"2px", background:"linear-gradient(90deg,#00ffe730,#a78bfa30,transparent)" }}/>
        </div>
      )}
    </div>
  );
}
