const API = '';  // same origin — served by FastAPI

let sessionId = null;

// ── DOM refs ────────────────────────────────────────────────────────────────
const messages   = document.getElementById('messages');
const typing     = document.getElementById('typing');
const input      = document.getElementById('user-input');
const sendBtn    = document.getElementById('send-btn');
const statusDot  = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// ── Session init ────────────────────────────────────────────────────────────
async function initSession() {
  try {
    const res  = await fetch(`${API}/session`, { method: 'POST' });
    const data = await res.json();
    sessionId  = data.session_id;
    setStatus(true);
    input.disabled   = false;
    sendBtn.disabled = false;
    input.focus();
  } catch {
    setStatus(false);
    setTimeout(initSession, 3000);
  }
}

function setStatus(online) {
  statusDot.className  = `dot ${online ? 'online' : 'offline'}`;
  statusText.textContent = online ? 'Connected' : 'Reconnecting…';
}

// ── Render helpers ──────────────────────────────────────────────────────────
function renderMarkdown(text) {
  return typeof marked !== 'undefined'
    ? marked.parse(text)
    : text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
}

function buildToolBadges(tools) {
  if (!tools || tools.length === 0) return '';
  // Count occurrences of each tool
  const counts = {};
  tools.forEach(t => { counts[t] = (counts[t] || 0) + 1; });
  const badges = Object.entries(counts).map(([name, n]) => {
    const label = name.replace(/_/g, ' ') + (n > 1 ? ` ×${n}` : '');
    return `<span class="tool-badge"><span class="badge-dot"></span>${label}</span>`;
  }).join('');
  return `<div class="tools-row">${badges}</div>`;
}

function appendMessage(role, content, tools = []) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;

  const avatarHtml = role === 'assistant'
    ? `<div class="avatar">♪</div>`
    : `<div class="avatar" style="background:#1a2e1d;border-color:#147a38;color:#1db954">you</div>`;

  const body = role === 'assistant'
    ? renderMarkdown(content) + buildToolBadges(tools)
    : `<p>${escapeHtml(content)}</p>`;

  div.innerHTML = `${avatarHtml}<div class="bubble">${body}</div>`;
  messages.appendChild(div);
  scrollToBottom();
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

// ── Send message ────────────────────────────────────────────────────────────
async function sendMessage(text) {
  if (!text.trim() || !sessionId) return;

  appendMessage('user', text);
  input.value = '';
  autoResize();
  setInputEnabled(false);
  showTyping(true);

  try {
    const res  = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    showTyping(false);
    appendMessage('assistant', data.reply, data.tools_called);
  } catch (err) {
    showTyping(false);
    appendMessage('assistant', `⚠️ Error: ${err.message}. Check the server is running.`);
  } finally {
    setInputEnabled(true);
    input.focus();
  }
}

function setInputEnabled(enabled) {
  input.disabled   = !enabled;
  sendBtn.disabled = !enabled;
}

function showTyping(show) {
  typing.classList.toggle('hidden', !show);
  if (show) scrollToBottom();
}

// ── Input handling ──────────────────────────────────────────────────────────
function autoResize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
}

input.addEventListener('input', autoResize);

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(input.value);
  }
});

sendBtn.addEventListener('click', () => sendMessage(input.value));

// ── Suggestion chips ────────────────────────────────────────────────────────
document.querySelectorAll('.suggestion').forEach(btn => {
  btn.addEventListener('click', () => {
    const msg = btn.dataset.msg;
    if (msg && !input.disabled) sendMessage(msg);
  });
});

// ── Boot ────────────────────────────────────────────────────────────────────
initSession();
