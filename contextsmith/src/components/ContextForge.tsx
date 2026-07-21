import React, { useState, useRef } from "react";
import { useRAG } from "../hooks/useRAG";
import type { ContextForgeProps, DocumentSource, QueryResult, RetrievedChunk } from "../types";

// ─── Sub-components ──────────────────────────────────────────────────────────

function StatusBadge({ status, isDark }: { status: string; isDark: boolean }) {
  const color =
    status === "done"  ? "#00ffe7" :
    status === "error" ? "#f87171" :
    status === "idle"  ? "#ffffff30" : "#a78bfa";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: "5px",
      background: color + "18", border: `1px solid ${color}40`,
      borderRadius: "5px", padding: "2px 8px",
      fontSize: "11px", color, fontFamily: "monospace",
    }}>
      {["embedding","chunking","loading","storing","retrieving","generating"].includes(status) && (
        <span style={{ display:"inline-block", width:"6px", height:"6px", borderRadius:"50%", background:color, animation:"rag-pulse 1s ease-in-out infinite" }}/>
      )}
      {status}
    </span>
  );
}

function SourceChip({ source, onRemove, isDark }: {
  source: { id: string; sourceName: string; sourceType: string };
  onRemove: () => void;
  isDark: boolean;
}) {
  const icons: Record<string, string> = { file: "📄", url: "🔗", text: "📝" };
  const C = { surface: isDark?"#0d1424":"#fff", border: isDark?"#ffffff12":"#e2e8f0", text: isDark?"#e2e8f0":"#1e293b", muted:"#64748b" };
  return (
    <div style={{
      display:"flex", alignItems:"center", gap:"8px",
      padding:"7px 12px", background:C.surface, border:`1px solid ${C.border}`,
      borderRadius:"8px", fontSize:"12px", color:C.text,
    }}>
      <span>{icons[source.sourceType] ?? "📄"}</span>
      <span style={{ maxWidth:"160px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
        {source.sourceName}
      </span>
      <button onClick={onRemove} style={{ background:"none", border:"none", color:C.muted, cursor:"pointer", fontSize:"14px", padding:"0 2px", lineHeight:1 }}>×</button>
    </div>
  );
}

function SourceCard({ chunk, rank, isDark }: { chunk: RetrievedChunk; rank: number; isDark: boolean }) {
  const [open, setOpen] = useState(false);
  const C = { surface: isDark?"#0d1424":"#fff", surfaceAlt: isDark?"#0a1020":"#f1f3f7", border: isDark?"#ffffff0d":"#e2e8f0", text: isDark?"#e2e8f0":"#1e293b", muted:"#64748b" };
  const score = typeof chunk.score === "number" ? chunk.score : 0;
  const pct = Math.round(score * 100);
  return (
    <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:"10px", overflow:"hidden" }}>
      <div onClick={()=>setOpen(o=>!o)} style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"10px 14px", cursor:"pointer", background:C.surfaceAlt }}>
        <div style={{ display:"flex", alignItems:"center", gap:"10px" }}>
          <span style={{ fontSize:"11px", fontFamily:"monospace", color:"#a78bfa", minWidth:"20px" }}>#{rank}</span>
          <span style={{ fontSize:"12px", color:C.text, maxWidth:"220px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{chunk.sourceName}</span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:"8px" }}>
          <div style={{ width:"48px", height:"4px", background: isDark?"#ffffff10":"#00000010", borderRadius:"2px" }}>
            <div style={{ width:`${pct}%`, height:"100%", background:`hsl(${pct * 1.2},80%,55%)`, borderRadius:"2px" }}/>
          </div>
          <span style={{ fontSize:"11px", fontFamily:"monospace", color:C.muted }}>{pct}%</span>
          <span style={{ fontSize:"11px", color:C.muted }}>{open?"▲":"▼"}</span>
        </div>
      </div>
      {open && (
        <div style={{ padding:"12px 14px", borderTop:`1px solid ${C.border}` }}>
          <p style={{ margin:0, fontSize:"13px", lineHeight:1.65, color:C.text, fontFamily:"monospace", whiteSpace:"pre-wrap" }}>{chunk.text}</p>
        </div>
      )}
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export function ContextForge({
  config,
  onQueried,
  onError,
  theme = "dark",
  className,
  style,
  placeholder = "Ask a question about your documents…",
}: ContextForgeProps) {
  const isDark = theme === "dark";
  const [question, setQuestion]         = useState("");
  const [urlInput, setUrlInput]         = useState("");
  const [textInput, setTextInput]       = useState("");
  const [activeTab, setActiveTab]       = useState<"file"|"url"|"text">("file");
  const [copied, setCopied]             = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const {
    index, query, removeDocument, clearIndex,
    indexedDocuments, totalChunks,
    lastResult, indexingProgress, queryStatus, error,
  } = useRAG({ config, onQueried, onError });

  const C = {
    bg:         isDark ? "#080c14" : "#f8f9fb",
    surface:    isDark ? "#0d1424" : "#ffffff",
    surfaceAlt: isDark ? "#0a1020" : "#f1f3f7",
    border:     isDark ? "#ffffff0d" : "#e2e8f0",
    text:       isDark ? "#e2e8f0"  : "#1e293b",
    muted:      "#64748b",
    accent:     "#00ffe7",
    purple:     "#a78bfa",
    error:      "#f87171",
  };

  const handleFiles = (files: FileList | null) => {
    if (!files?.length) return;
    const sources: DocumentSource[] = Array.from(files).map(f => ({ type: "file", content: f }));
    index(sources);
  };

  const handleUrl = () => {
    if (!urlInput.trim()) return;
    index([{ type: "url", content: urlInput.trim() }]);
    setUrlInput("");
  };

  const handleText = () => {
    if (!textInput.trim()) return;
    index([{ type: "text", content: textInput.trim() }]);
    setTextInput("");
  };

  const handleQuery = () => {
    if (!question.trim() || totalChunks === 0) return;
    query(question);
  };

  const copyAnswer = () => {
    if (!lastResult) return;
    navigator.clipboard.writeText(lastResult.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const indexing = ["loading","chunking","embedding","storing"].includes(indexingProgress.status);
  const querying = ["embedding","retrieving","generating"].includes(queryStatus);

  return (
    <div className={className} style={{
      background: C.bg, fontFamily: "'DM Sans',system-ui,sans-serif",
      color: C.text, borderRadius: "16px", padding: "32px",
      maxWidth: "860px", width: "100%",
      display: "grid", gap: "24px",
      border: `1px solid ${C.border}`, ...style,
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        *{box-sizing:border-box}
        .rag-ta{width:100%;background:transparent;border:none;outline:none;resize:none;font-family:'DM Sans',sans-serif;font-size:14px;line-height:1.6;color:${C.text};caret-color:#00ffe7;padding:0;}
        .rag-input{width:100%;background:transparent;border:none;outline:none;font-family:'DM Sans',sans-serif;font-size:14px;color:${C.text};caret-color:#00ffe7;padding:0;}
        .rag-btn{cursor:pointer;border:none;background:linear-gradient(135deg,#00ffe7,#a78bfa);color:#080c14;font-family:'DM Sans',sans-serif;font-size:13px;font-weight:600;padding:10px 22px;border-radius:9px;transition:all 0.18s;white-space:nowrap;}
        .rag-btn:hover:not(:disabled){opacity:.88;transform:translateY(-1px);box-shadow:0 6px 20px #00ffe728;}
        .rag-btn:disabled{opacity:.35;cursor:not-allowed;}
        .rag-ghost{cursor:pointer;border:1px solid ${C.border};background:transparent;color:${C.muted};font-family:'DM Sans',sans-serif;font-size:12px;padding:7px 14px;border-radius:8px;transition:all .18s;}
        .rag-ghost:hover{border-color:#00ffe740;color:${C.text};}
        .rag-tab{cursor:pointer;border:none;background:transparent;color:${C.muted};font-family:'DM Sans',sans-serif;font-size:13px;padding:8px 16px;border-radius:8px;transition:all .18s;}
        .rag-tab.on{background:${C.surface};color:${C.text};border:1px solid ${C.border};}
        .rag-tab:not(.on):hover{color:${C.text};}
        .rag-drop{border:2px dashed ${C.border};border-radius:12px;padding:32px;text-align:center;cursor:pointer;transition:border-color .2s;}
        .rag-drop:hover,.rag-drop.over{border-color:#00ffe760;}
        @keyframes rag-sh{0%{background-position:-200% center}100%{background-position:200% center}}
        .rag-bar{height:2px;background:linear-gradient(90deg,transparent,#00ffe7,#a78bfa,transparent);background-size:200% auto;animation:rag-sh 1.4s linear infinite;border-radius:1px;}
        @keyframes rag-pulse{0%,100%{opacity:1}50%{opacity:.4}}
        @keyframes rag-fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
        .rag-fi{animation:rag-fi .4s ease forwards;}
      `}</style>

      {/* Header */}
      <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", flexWrap:"wrap", gap:"12px" }}>
        <div>
          <div style={{ display:"flex", alignItems:"center", gap:"10px", marginBottom:"6px" }}>
            <h2 style={{ fontFamily:"system-ui", fontSize:"20px", fontWeight:700, margin:0, letterSpacing:"-0.02em" }}>RAG Pipeline</h2>
            <StatusBadge status={indexingProgress.status === "idle" && queryStatus !== "idle" ? queryStatus : indexingProgress.status} isDark={isDark}/>
          </div>
          <p style={{ color:C.muted, fontSize:"13px", margin:0 }}>
            {totalChunks > 0
              ? `${indexedDocuments.length} document${indexedDocuments.length !== 1 ? "s" : ""} · ${totalChunks} chunks indexed`
              : "Index documents, then query them"}
          </p>
        </div>
        {indexedDocuments.length > 0 && (
          <button className="rag-ghost" onClick={clearIndex} disabled={indexing}>Clear index</button>
        )}
      </div>

      {/* Indexing section */}
      <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:"14px", overflow:"hidden" }}>
        <div style={{ padding:"12px 16px", background:C.surfaceAlt, borderBottom:`1px solid ${C.border}`, display:"flex", alignItems:"center", gap:"4px" }}>
          {(["file","url","text"] as const).map(t => (
            <button key={t} className={`rag-tab ${activeTab===t?"on":""}`} onClick={()=>setActiveTab(t)}>
              {t==="file"?"📄 File":t==="url"?"🔗 URL":"📝 Text"}
            </button>
          ))}
        </div>

        <div style={{ padding:"20px" }}>
          {activeTab === "file" && (
            <div
              className="rag-drop"
              onClick={() => fileRef.current?.click()}
              onDragOver={e => { e.preventDefault(); e.currentTarget.classList.add("over"); }}
              onDragLeave={e => e.currentTarget.classList.remove("over")}
              onDrop={e => { e.preventDefault(); e.currentTarget.classList.remove("over"); handleFiles(e.dataTransfer.files); }}
            >
              <div style={{ fontSize:"28px", marginBottom:"8px" }}>📂</div>
              <div style={{ fontSize:"14px", color:C.text, marginBottom:"4px" }}>Drop files here or click to browse</div>
              <div style={{ fontSize:"12px", color:C.muted }}>TXT, MD, CSV, JSON, HTML supported</div>
              <input ref={fileRef} type="file" multiple accept=".txt,.md,.csv,.json,.html" style={{ display:"none" }} onChange={e=>handleFiles(e.target.files)}/>
            </div>
          )}

          {activeTab === "url" && (
            <div style={{ display:"flex", gap:"10px" }}>
              <div style={{ flex:1, background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:"10px", padding:"10px 14px" }}>
                <input className="rag-input" value={urlInput} onChange={e=>setUrlInput(e.target.value)} placeholder="https://example.com/article" onKeyDown={e=>e.key==="Enter"&&handleUrl()}/>
              </div>
              <button className="rag-btn" onClick={handleUrl} disabled={!urlInput.trim()||indexing}>Fetch</button>
            </div>
          )}

          {activeTab === "text" && (
            <div>
              <div style={{ background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:"10px", padding:"12px 14px", marginBottom:"10px" }}>
                <textarea className="rag-ta" rows={5} value={textInput} onChange={e=>setTextInput(e.target.value)} placeholder="Paste any text to index…"/>
              </div>
              <button className="rag-btn" onClick={handleText} disabled={!textInput.trim()||indexing}>Index Text</button>
            </div>
          )}
        </div>

        {/* Progress bar */}
        {indexing && (
          <div style={{ padding:"0 20px 16px" }}>
            <div className="rag-bar"/>
            <div style={{ fontSize:"12px", color:C.muted, marginTop:"8px", fontFamily:"monospace" }}>
              {indexingProgress.status}
              {indexingProgress.documentName && ` · ${indexingProgress.documentName.slice(0,50)}`}
              {indexingProgress.totalChunks && ` · ${indexingProgress.chunksProcessed}/${indexingProgress.totalChunks} chunks`}
            </div>
          </div>
        )}
      </div>

      {/* Indexed documents */}
      {indexedDocuments.length > 0 && (
        <div>
          <p style={{ fontSize:"11px", color:C.muted, letterSpacing:".1em", textTransform:"uppercase", margin:"0 0 10px", fontFamily:"monospace" }}>Indexed</p>
          <div style={{ display:"flex", gap:"8px", flexWrap:"wrap" }}>
            {indexedDocuments.map(doc => (
              <SourceChip key={doc.id} source={doc} onRemove={()=>removeDocument(doc.id)} isDark={isDark}/>
            ))}
          </div>
        </div>
      )}

      {/* Query section */}
      <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:"14px", overflow:"hidden" }}>
        <div style={{ padding:"14px 16px 0" }}>
          <textarea
            className="rag-ta"
            rows={3}
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder={totalChunks === 0 ? "Index documents above before querying…" : placeholder}
            disabled={querying}
            onKeyDown={e => { if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); handleQuery(); } }}
          />
        </div>
        <div style={{ padding:"10px 16px 12px", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
          <span style={{ fontSize:"11px", color:C.muted, fontFamily:"monospace" }}>
            {totalChunks > 0 ? `${totalChunks} chunks · top ${config.retrieval?.topK ?? 5} retrieved` : "No documents indexed"}
          </span>
          <button className="rag-btn" onClick={handleQuery} disabled={querying || !question.trim() || totalChunks === 0}>
            {querying ? `${queryStatus}…` : "Query →"}
          </button>
        </div>
        {querying && <div style={{ margin:"0 16px 14px" }}><div className="rag-bar"/></div>}
      </div>

      {/* Error */}
      {error && (
        <div style={{ background:"#f8717110", border:"1px solid #f8717130", borderRadius:"10px", padding:"11px 15px", color:C.error, fontSize:"13px", fontFamily:"monospace" }}>
          ✗ {error.message}
        </div>
      )}

      {/* Results */}
      {lastResult && (
        <div className="rag-fi">
          {/* Answer */}
          <div style={{ background:C.surface, border:`1px solid #00ffe720`, borderRadius:"14px", overflow:"hidden", marginBottom:"16px", boxShadow:"0 0 32px #00ffe710" }}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"11px 16px", background:isDark?"#091a14":"#f0fdf8", borderBottom:"1px solid #00ffe720" }}>
              <div style={{ display:"flex", alignItems:"center", gap:"8px" }}>
                <span style={{ fontSize:"14px" }}>💡</span>
                <span style={{ fontSize:"12px", color:C.accent, fontFamily:"monospace", letterSpacing:".08em", textTransform:"uppercase" }}>Answer</span>
                <span style={{ fontSize:"11px", color:C.muted }}>{lastResult.durationMs}ms</span>
              </div>
              <button className="rag-ghost" onClick={copyAnswer}>{copied?"✓ Copied":"Copy"}</button>
            </div>
            <div style={{ padding:"20px" }}>
              <p style={{ fontFamily:"'Georgia',serif", fontSize:"16px", lineHeight:1.8, color:C.text, margin:0, whiteSpace:"pre-wrap" }}>
                {lastResult.answer}
              </p>
            </div>
            <div style={{ height:"2px", background:"linear-gradient(90deg,#00ffe740,transparent)" }}/>
          </div>

          {/* Sources */}
          {lastResult.sources.length > 0 && (
            <div>
              <p style={{ fontSize:"11px", color:C.muted, letterSpacing:".1em", textTransform:"uppercase", margin:"0 0 10px", fontFamily:"monospace" }}>
                Sources · {lastResult.sources.length} chunks retrieved
              </p>
              <div style={{ display:"grid", gap:"8px" }}>
                {lastResult.sources.map((src, i) => (
                  <SourceCard key={src.id} chunk={src} rank={i + 1} isDark={isDark}/>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
