"""FastAPI route definitions and recruiter-friendly browser UI."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response

from app.graph import GraphBackedSHLAgent
from app.interfaces import RecommenderAgent
from app.schemas import ChatRequest, ChatResponse, HealthResponse


def create_router(agent: RecommenderAgent | None = None) -> APIRouter:
    """Create the required API plus a thin UI that calls the stateless API."""

    router = APIRouter()
    runtime_agent: RecommenderAgent = agent or GraphBackedSHLAgent()

    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def home() -> HTMLResponse:
        return HTMLResponse(_CHAT_UI)

    @router.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        return await runtime_agent.chat(request.messages)

    return router


_CHAT_UI = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SHL Assessment Recommender</title>
  <style>
    :root {
      --bg: #f5f7fb; --card: #ffffff; --ink: #172033; --muted: #647086;
      --line: #dfe5f0; --brand: #2447d8; --soft: #eef2ff; --ok: #0f766e;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    .layout { min-height: 100vh; display: grid; grid-template-columns: 340px 1fr; }
    aside { background: var(--card); border-right: 1px solid var(--line); padding: 28px; }
    main { height: 100vh; display: flex; flex-direction: column; }
    h1 { margin: 0 0 12px; font-size: 27px; line-height: 1.08; letter-spacing: -0.03em; }
    p { color: var(--muted); line-height: 1.5; }
    .badge { display: inline-block; color: var(--ok); background: #ecfdf5; border: 1px solid #b7efe0; border-radius: 999px; padding: 7px 11px; font-size: 13px; margin-bottom: 18px; }
    .examples { display: grid; gap: 10px; margin-top: 22px; }
    .example, .secondary { border: 1px solid var(--line); background: #fbfcff; border-radius: 13px; padding: 11px 13px; cursor: pointer; text-align: left; color: var(--ink); }
    .example:hover, .secondary:hover { border-color: var(--brand); }
    .chat { flex: 1; overflow-y: auto; padding: 28px; }
    .message { max-width: 980px; margin: 0 auto 16px; display: flex; gap: 12px; align-items: flex-start; }
    .avatar { width: 38px; height: 38px; flex: 0 0 38px; border-radius: 14px; display: grid; place-items: center; font-weight: 800; font-size: 12px; }
    .user .avatar { background: #172033; color: white; }
    .assistant .avatar { background: var(--soft); color: var(--brand); }
    .bubble { width: 100%; background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 15px; box-shadow: 0 10px 25px rgba(23,32,51,.05); overflow-x: auto; }
    .reply-text { line-height: 1.5; white-space: pre-wrap; }
    .comparison-table { width: 100%; border-collapse: collapse; margin-top: 4px; min-width: 720px; font-size: 14px; }
    .comparison-table th { background: var(--soft); color: var(--ink); text-align: left; }
    .comparison-table th, .comparison-table td { border: 1px solid var(--line); padding: 10px 12px; vertical-align: top; line-height: 1.45; }
    .comparison-table td:first-child { font-weight: 750; width: 160px; color: var(--ink); }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; margin-top: 12px; }
    .rec { border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: #fbfcff; }
    .rec a { color: var(--brand); font-weight: 750; text-decoration: none; }
    .pill { display: inline-block; margin-top: 8px; background: var(--soft); color: var(--brand); border-radius: 999px; padding: 4px 9px; font-size: 12px; text-transform: capitalize; }
    form { border-top: 1px solid var(--line); padding: 18px 28px; background: rgba(255,255,255,.9); }
    .composer { max-width: 980px; margin: 0 auto; display: flex; gap: 10px; }
    textarea { flex: 1; resize: none; border: 1px solid var(--line); border-radius: 16px; padding: 14px; font: inherit; min-height: 54px; outline: none; }
    textarea:focus { border-color: var(--brand); }
    button.primary { border: 0; background: var(--brand); color: white; border-radius: 16px; padding: 0 20px; font-weight: 800; cursor: pointer; }
    button.primary:disabled { opacity: .65; cursor: wait; }
    .actions { display: flex; gap: 9px; flex-wrap: wrap; margin-top: 18px; }
    .note { font-size: 13px; color: var(--muted); margin-top: 16px; }
    @media (max-width: 850px) { .layout { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid var(--line); } main { height: auto; min-height: 70vh; } .composer { flex-direction: column; } button.primary { min-height: 48px; } }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <div class="badge">Catalog-grounded SHL only</div>
      <h1>SHL Assessment Recommender</h1>
      <p>Chat with the agent. It clarifies vague needs, recommends SHL catalog assessments, refines when requirements change, and compares assessments using catalog data only.</p>
      <div class="examples">
        <button class="example">I need an assessment</button>
        <button class="example">Hiring Java developer with 4 years experience</button>
        <button class="example">Actually, add personality tests</button>
        <button class="example">What is the difference between OPQ and GSA?</button>
      </div>
      <div class="actions">
        <button class="secondary" id="reset">New chat</button>
        <a class="secondary" href="/docs" style="text-decoration:none">API docs</a>
      </div>
      <p class="note">The browser sends the full conversation history to POST /chat each turn. The server stores no chat memory.</p>
    </aside>
    <main>
      <section class="chat" id="chat"></section>
      <form id="form">
        <div class="composer">
          <textarea id="input" rows="2" placeholder="Describe the role, job description, or assessment need..."></textarea>
          <button class="primary" id="send" type="submit">Send</button>
        </div>
      </form>
    </main>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const form = document.getElementById("form");
    const input = document.getElementById("input");
    const send = document.getElementById("send");
    const reset = document.getElementById("reset");
    const messages = [];
    function esc(s) { return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    function isSeparatorRow(line) {
      return /^\\|\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?$/.test(line.trim());
    }
    function splitTableRow(line) {
      return line.trim().replace(/^\\|/, "").replace(/\\|$/, "").split("|").map(cell => cell.trim());
    }
    function renderMarkdownTable(lines) {
      const rows = lines.filter(line => line.trim().startsWith("|") && !isSeparatorRow(line)).map(splitTableRow);
      if (!rows.length) return "";
      const header = rows[0];
      const body = rows.slice(1);
      return `
        <table class="comparison-table">
          <thead><tr>${header.map(cell => `<th>${esc(cell)}</th>`).join("")}</tr></thead>
          <tbody>${body.map(row => `<tr>${row.map(cell => `<td>${esc(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
      `;
    }
    function renderReply(content) {
      const lines = content.split("\\n");
      const tableStart = lines.findIndex(line => line.trim().startsWith("|"));
      if (tableStart === -1) {
        return `<div class="reply-text">${esc(content)}</div>`;
      }
      const intro = lines.slice(0, tableStart).join("\\n").trim();
      const tableLines = lines.slice(tableStart).filter(line => line.trim().startsWith("|"));
      const introHtml = intro ? `<div class="reply-text">${esc(intro)}</div>` : "";
      return introHtml + renderMarkdownTable(tableLines);
    }
    function add(role, content, recs = []) {
      const row = document.createElement("div");
      row.className = `message ${role}`;
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = role === "user" ? "YOU" : "SHL";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.innerHTML = renderReply(content);
      if (recs.length) {
        const cards = document.createElement("div");
        cards.className = "cards";
        recs.forEach((rec, i) => {
          const card = document.createElement("div");
          card.className = "rec";
          card.innerHTML = `<a href="${rec.url}" target="_blank" rel="noopener">${i + 1}. ${esc(rec.name)}</a><br><span class="pill">${esc(rec.test_type)}</span>`;
          cards.appendChild(card);
        });
        bubble.appendChild(cards);
      }
      row.appendChild(avatar); row.appendChild(bubble); chat.appendChild(row); chat.scrollTop = chat.scrollHeight;
    }
    async function sendMessage(text) {
      const clean = text.trim(); if (!clean) return;
      messages.push({role: "user", content: clean});
      add("user", clean); input.value = ""; send.disabled = true; send.textContent = "Thinking";
      try {
        const res = await fetch("/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({messages})});
        const raw = await res.text();
        let data = null;
        try { data = JSON.parse(raw); } catch (_) { /* handled below with HTTP context */ }
        if (!res.ok) {
          const detail = data?.detail || raw.replace(/<[^>]*>/g, " ").replace(/\\s+/g, " ").trim().slice(0, 180);
          throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ""}`);
        }
        if (!data) throw new Error(`HTTP ${res.status}: server returned HTML instead of JSON`);
        messages.push({role: "assistant", content: data.reply});
        add("assistant", data.reply, data.recommendations || []);
      } catch (err) {
        add("assistant", `API error: ${err.message}`);
      } finally {
        send.disabled = false; send.textContent = "Send"; input.focus();
      }
    }
    form.addEventListener("submit", e => { e.preventDefault(); sendMessage(input.value); });
    input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input.value); } });
    document.querySelectorAll(".example").forEach(b => b.addEventListener("click", () => sendMessage(b.textContent)));
    reset.addEventListener("click", () => { messages.length = 0; chat.innerHTML = ""; add("assistant", "Hi — tell me the role, seniority, skills, or paste a job description. I’ll clarify if needed and recommend only SHL catalog assessments."); input.focus(); });
    add("assistant", "Hi — tell me the role, seniority, skills, or paste a job description. I’ll clarify if needed and recommend only SHL catalog assessments.");
  </script>
</body>
</html>
"""
