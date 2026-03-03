(function () {
  "use strict";

  // ── Config ──────────────────────────────────────────────────────────
  const DEFAULTS = {
    apiUrl: "",
    position: "right",
    title: "Nova",
    subtitle: "How can I help you today?",
    primaryColor: "#6366f1",
    greeting: "Hello!",
    suggestions: [
      "What are your best security products?",
      "Do you have any active discounts?",
      "What's included in the Business Essentials Bundle?",
    ],
    historyTTL: 24 * 60 * 60 * 1000,
  };

  const STORAGE_KEY = "rag_widget_history";

  // ── Markdown → HTML (basic) ─────────────────────────────────────────
  function renderMarkdown(raw) {
    // Escape HTML first
    var text = raw.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    // Headings: #### / ### / ## / # (must be before line break conversion)
    text = text.replace(/(^|\n)#### (.+?)(\n|$)/g, "$1<strong style=\"font-size:14px\">$2</strong>$3");
    text = text.replace(/(^|\n)### (.+?)(\n|$)/g, "$1<strong style=\"font-size:15px\">$2</strong>$3");
    text = text.replace(/(^|\n)## (.+?)(\n|$)/g, "$1<strong style=\"font-size:16px\">$2</strong>$3");
    text = text.replace(/(^|\n)# (.+?)(\n|$)/g, "$1<strong style=\"font-size:18px\">$2</strong>$3");
    // Bold: **text**
    text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Italic: *text* (but not inside **)
    text = text.replace(/\*(.+?)\*/g, "<em>$1</em>");
    // Inline code: `text`
    text = text.replace(/`(.+?)`/g, "<code>$1</code>");
    // Line breaks
    text = text.replace(/\n/g, "<br>");
    return text;
  }

  // ── LocalStorage Chat History (24H TTL) ─────────────────────────────
  function loadHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const data = JSON.parse(raw);
      if (Date.now() - data.createdAt > DEFAULTS.historyTTL) {
        localStorage.removeItem(STORAGE_KEY);
        return null;
      }
      return data;
    } catch (e) {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
  }

  function saveHistory(sessionId, chatMessages) {
    try {
      const existing = loadHistory();
      const createdAt = (existing && existing.createdAt) || Date.now();
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        sessionId: sessionId,
        messages: chatMessages,
        createdAt: createdAt,
      }));
    } catch (e) {}
  }

  function clearHistory() {
    localStorage.removeItem(STORAGE_KEY);
  }

  // ── Detect API base URL from <script> tag ───────────────────────────
  function detectApiUrl() {
    const scripts = document.querySelectorAll("script[src]");
    for (const s of scripts) {
      if (s.src.includes("widget.js")) {
        return s.getAttribute("data-api-url") || s.src.replace(/\/static\/widget\.js.*$/, "");
      }
    }
    return window.location.origin;
  }

  // ── SVG icons ─────────────────────────────────────────────────────
  const ICON_SPARKLE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/><path d="M19 15l.5 1.5L21 17l-1.5.5L19 19l-.5-1.5L17 17l1.5-.5L19 15z"/><path d="M5 17l.5 1.5L7 19l-1.5.5L5 21l-.5-1.5L3 19l1.5-.5L5 17z"/></svg>';
  const ICON_SPARKLE_SM = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="rag-bot-label-icon"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/><path d="M19 15l.5 1.5L21 17l-1.5.5L19 19l-.5-1.5L17 17l1.5-.5L19 15z"/><path d="M5 17l.5 1.5L7 19l-1.5.5L5 21l-.5-1.5L3 19l1.5-.5L5 17z"/></svg>';
  const ICON_CHEVRON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>';
  const ICON_NEW_CHAT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>';
  const ICON_SEND = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>';
  const ICON_ARROW = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>';

  // ── Styles ──────────────────────────────────────────────────────────
  function injectStyles(cfg) {
    const isLeft = cfg.position === "left";
    const style = document.createElement("style");
    style.textContent = `
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      #rag-widget-container * {
        box-sizing: border-box; margin: 0; padding: 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }

      /* ── Animations ────────────────────────────── */
      @keyframes ragSpin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
      }
      @keyframes ragFadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      @keyframes ragBubbleIn {
        from { opacity: 0; transform: scale(0.8) translateY(10px); }
        to   { opacity: 1; transform: scale(1) translateY(0); }
      }

      /* ── Bubble Button ─────────────────────────── */
      #rag-widget-bubble {
        position: fixed; bottom: 24px; ${isLeft ? "left" : "right"}: 24px;
        height: 48px;
        padding: 0 20px 0 14px;
        border-radius: 24px;
        background: #673ab7;
        color: #fff;
        display: flex; align-items: center; gap: 8px;
        cursor: pointer; z-index: 99999;
        box-shadow: 0 4px 16px rgba(103,58,183,.35);
        transition: transform .2s ease, box-shadow .2s ease;
        border: none; outline: none;
        animation: ragBubbleIn .4s ease backwards;
        font-family: 'Inter', sans-serif;
        font-size: 14px; font-weight: 600;
      }
      #rag-widget-bubble:hover {
        transform: scale(1.04);
        box-shadow: 0 6px 22px rgba(103,58,183,.45);
      }
      #rag-widget-bubble:active { transform: scale(0.97); }
      #rag-widget-bubble svg { width: 20px; height: 20px; color: #fff; flex-shrink: 0; }

      /* ── Chat Window ────────────────────────────── */
      #rag-widget-window {
        position: fixed; bottom: 84px; ${isLeft ? "left" : "right"}: 24px;
        width: 420px; max-width: calc(100vw - 48px);
        height: 680px; max-height: calc(100vh - 110px);
        border-radius: 16px; overflow: hidden;
        display: flex; flex-direction: column;
        z-index: 99999;
        background: #fff;
        box-shadow: 0 8px 40px rgba(0,0,0,.12), 0 0 0 1px rgba(0,0,0,.06);
        opacity: 0;
        transform: scale(0.95) translateY(8px);
        pointer-events: none;
        transition: opacity .2s ease, transform .2s ease;
      }
      #rag-widget-window.open {
        opacity: 1;
        transform: scale(1) translateY(0);
        pointer-events: auto;
      }

      /* ── Header (minimal Kodee-style) ──────────── */
      #rag-widget-header {
        background: #fff;
        color: #1a1a1a;
        padding: 16px 20px;
        display: flex; align-items: center; justify-content: space-between;
        flex-shrink: 0;
        border-bottom: 1px solid #f0f0f0;
      }
      #rag-widget-header-title {
        font-size: 16px; font-weight: 700; color: #1a1a1a;
      }
      #rag-widget-header-actions {
        display: flex; align-items: center; gap: 4px;
      }
      #rag-widget-new-chat, #rag-widget-close {
        background: none; border: none; color: #888;
        cursor: pointer; padding: 4px; border-radius: 6px; display: flex;
        transition: color .2s, background .2s;
      }
      #rag-widget-new-chat:hover, #rag-widget-close:hover { color: #333; background: #f5f5f5; }
      #rag-widget-new-chat svg, #rag-widget-close svg { width: 22px; height: 22px; }

      /* ── Welcome Screen ────────────────────────── */
      #rag-widget-welcome {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center;
        padding: 40px 24px 24px;
        flex-shrink: 0;
      }
      #rag-widget-welcome .rag-welcome-icon {
        width: 56px; height: 56px;
        color: #673ab7;
        margin-bottom: 20px;
      }
      #rag-widget-welcome .rag-welcome-icon svg {
        width: 100%; height: 100%;
      }
      #rag-widget-welcome h2 {
        font-size: 22px; font-weight: 700; color: #1a1a1a;
        margin-bottom: 6px;
      }
      #rag-widget-welcome p {
        font-size: 14px; color: #666;
      }

      /* ── Messages ───────────────────────────────── */
      #rag-widget-messages {
        flex: 1; overflow-y: auto; padding: 16px 16px 8px;
        display: flex; flex-direction: column; gap: 4px;
        background: #fff;
      }
      #rag-widget-messages:empty { display: none; }

      .rag-msg {
        max-width: 85%; font-size: 14px;
        line-height: 1.6; word-wrap: break-word;
        animation: ragFadeUp .25s ease backwards;
      }
      .rag-msg-user {
        align-self: flex-end;
        background: #f3f0ff;
        color: #1a1a1a;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin-top: 8px;
      }

      /* ── Bot message wrapper (label + text) ────── */
      .rag-bot-wrapper {
        align-self: flex-start;
        max-width: 90%;
        animation: ragFadeUp .25s ease backwards;
        margin-top: 8px;
      }
      .rag-bot-label {
        display: flex; align-items: center; gap: 6px;
        margin-bottom: 6px;
      }
      .rag-bot-label-icon {
        width: 18px; height: 18px; color: #673ab7;
      }
      .rag-bot-label-name {
        font-size: 13px; font-weight: 700; color: #1a1a1a;
      }
      .rag-bot-body {
        font-size: 14px; line-height: 1.65; color: #1a1a1a;
        word-wrap: break-word;
      }
      .rag-bot-body strong { font-weight: 700; }
      .rag-bot-body em { font-style: italic; }
      .rag-bot-body code {
        background: #f3f0ff; padding: 1px 5px; border-radius: 4px;
        font-family: 'SF Mono', Consolas, monospace; font-size: 13px;
      }

      /* ── Spinning Logo Loader ──────────────────── */
      #rag-typing {
        align-self: flex-start;
        display: flex; align-items: center; gap: 10px;
        padding: 12px 4px;
        animation: ragFadeUp .25s ease backwards;
      }
      #rag-typing .rag-spin-icon {
        width: 28px; height: 28px; color: #673ab7;
        animation: ragSpin 1.2s linear infinite;
      }
      #rag-typing .rag-spin-icon svg { width: 100%; height: 100%; }
      #rag-typing .rag-typing-text {
        font-size: 13px; color: #999; font-style: italic;
      }

      /* ── Suggestions ───────────────────────────── */
      #rag-widget-suggestions {
        flex-shrink: 0;
        border-top: 1px solid #f0f0f0;
        background: #fff;
        margin-top: 40px;
      }
      .rag-suggestion-item {
        display: flex !important; align-items: center !important; gap: 20px !important;
        padding: 14px 20px 14px 36px !important;
        margin: 0 !important;
        border: none !important; background: none;
        border-bottom: 1px solid #e8e8e8 !important;
        width: 100%; text-align: left;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        font-size: 14px; color: #1a1a1a;
        transition: background .15s;
      }
      .rag-suggestion-item:last-child { border-bottom: none !important; }
      .rag-suggestion-item:hover { background: #faf8ff; }
      .rag-suggestion-item svg {
        width: 24px !important; height: 24px !important; color: #000 !important;
        flex-shrink: 0;
      }

      /* ── Input Area ─────────────────────────────── */
      #rag-widget-input-area {
        padding: 12px 16px;
        background: #fff; flex-shrink: 0;
        border-top: 1px solid #f0f0f0;
      }
      #rag-widget-input-row {
        display: flex; align-items: flex-end; gap: 8px;
        border: 1.5px solid #e0e0e0;
        border-radius: 16px;
        padding: 6px 6px 6px 16px;
        transition: border-color .2s;
      }
      #rag-widget-input-row:focus-within {
        border-color: #673ab7;
      }
      #rag-widget-input {
        flex: 1; border: none; outline: none; resize: none;
        font-size: 14px; line-height: 1.4;
        min-height: 24px; max-height: 100px;
        padding: 6px 0;
        font-family: 'Inter', sans-serif;
        background: transparent;
      }
      #rag-widget-input::placeholder { color: #bbb; }
      #rag-widget-send {
        width: 36px; height: 36px; border-radius: 50%;
        background: #673ab7; border: none; color: #fff;
        cursor: pointer; display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        transition: opacity .15s, transform .15s;
      }
      #rag-widget-send:hover:not(:disabled) { transform: scale(1.06); }
      #rag-widget-send:disabled { opacity: .35; cursor: not-allowed; }
      #rag-widget-send svg { width: 16px; height: 16px; }

      /* ── Footer ─────────────────────────────────── */
      #rag-widget-footer {
        text-align: center; font-size: 11px; color: #aaa;
        padding: 8px 16px;
        background: #fff;
      }

      /* ── State: hide welcome when chatting ─────── */
      #rag-widget-window.chatting #rag-widget-welcome { display: none; }
      #rag-widget-window.chatting #rag-widget-suggestions { display: none; }
      #rag-widget-window.chatting #rag-widget-messages { display: flex; }
    `;
    document.head.appendChild(style);
  }

  // ── Build DOM ───────────────────────────────────────────────────────
  function buildWidget(cfg) {
    const container = document.createElement("div");
    container.id = "rag-widget-container";

    const suggestionsHTML = cfg.suggestions.map(function (text) {
      return '<button class="rag-suggestion-item">' + ICON_ARROW + '<span>' + text + '</span></button>';
    }).join("");

    container.innerHTML = `
      <button id="rag-widget-bubble">${ICON_SPARKLE}<span>Ask Nova</span></button>
      <div id="rag-widget-window">
        <div id="rag-widget-header">
          <span id="rag-widget-header-title">${cfg.title}</span>
          <div id="rag-widget-header-actions">
            <button id="rag-widget-new-chat" aria-label="New chat" title="New chat">${ICON_NEW_CHAT}</button>
            <button id="rag-widget-close" aria-label="Close chat">${ICON_CHEVRON}</button>
          </div>
        </div>

        <div id="rag-widget-welcome">
          <div class="rag-welcome-icon">${ICON_SPARKLE}</div>
          <h2>${cfg.greeting}</h2>
          <p>${cfg.subtitle}</p>
        </div>

        <div id="rag-widget-messages"></div>

        <div id="rag-widget-suggestions">${suggestionsHTML}</div>

        <div id="rag-widget-input-area">
          <div id="rag-widget-input-row">
            <textarea id="rag-widget-input" placeholder="Type your message..." rows="1"></textarea>
            <button id="rag-widget-send" aria-label="Send">${ICON_SEND}</button>
          </div>
        </div>
        <div id="rag-widget-footer">Nova can make mistakes. Double-check replies.</div>
      </div>
    `;
    document.body.appendChild(container);
    return container;
  }

  // ── Widget Logic ────────────────────────────────────────────────────
  function initWidget() {
    const cfg = Object.assign({}, DEFAULTS);

    const scripts = document.querySelectorAll("script[src]");
    for (const s of scripts) {
      if (s.src.includes("widget.js")) {
        if (s.dataset.position) cfg.position = s.dataset.position;
        if (s.dataset.title) cfg.title = s.dataset.title;
        if (s.dataset.subtitle) cfg.subtitle = s.dataset.subtitle;
        if (s.dataset.color) cfg.primaryColor = s.dataset.color;
        if (s.dataset.greeting) cfg.greeting = s.dataset.greeting;
        if (s.dataset.apiUrl) cfg.apiUrl = s.dataset.apiUrl;
        break;
      }
    }
    if (!cfg.apiUrl) cfg.apiUrl = detectApiUrl();

    injectStyles(cfg);
    buildWidget(cfg);

    // DOM refs
    const bubble = document.getElementById("rag-widget-bubble");
    const win = document.getElementById("rag-widget-window");
    const closeBtn = document.getElementById("rag-widget-close");
    const messages = document.getElementById("rag-widget-messages");
    const input = document.getElementById("rag-widget-input");
    const sendBtn = document.getElementById("rag-widget-send");
    const suggestionsEl = document.getElementById("rag-widget-suggestions");
    const newChatBtn = document.getElementById("rag-widget-new-chat");

    let isOpen = false;
    let sessionId = null;
    let isStreaming = false;

    // Toggle open/close
    function toggle() {
      isOpen = !isOpen;
      win.classList.toggle("open", isOpen);
      if (isOpen) input.focus();
    }
    bubble.addEventListener("click", toggle);
    closeBtn.addEventListener("click", toggle);

    // New chat: clear messages, reset session, show welcome screen
    newChatBtn.addEventListener("click", function () {
      if (isStreaming) return;
      messages.innerHTML = "";
      sessionId = null;
      clearHistory();
      win.classList.remove("chatting");
      input.value = "";
      input.style.height = "auto";
    });

    // Suggestion chip clicks
    suggestionsEl.querySelectorAll(".rag-suggestion-item").forEach(function (btn) {
      btn.addEventListener("click", function () {
        input.value = btn.querySelector("span").textContent;
        win.classList.add("chatting");
        send();
      });
    });

    // Auto-resize textarea
    input.addEventListener("input", function () {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 100) + "px";
    });

    // Send on Enter
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
    sendBtn.addEventListener("click", send);

    // ── Messages ──────────────────────────────────────────────────

    function addUserMessage(text) {
      const el = document.createElement("div");
      el.className = "rag-msg rag-msg-user";
      el.textContent = text;
      messages.appendChild(el);
      scrollDown();
    }

    // Creates a Kodee-style bot message: sparkle icon + "Nova" label, then text below
    function createBotWrapper() {
      const wrapper = document.createElement("div");
      wrapper.className = "rag-bot-wrapper";

      const label = document.createElement("div");
      label.className = "rag-bot-label";
      label.innerHTML = ICON_SPARKLE_SM + '<span class="rag-bot-label-name">Nova</span>';

      const body = document.createElement("div");
      body.className = "rag-bot-body";

      wrapper.appendChild(label);
      wrapper.appendChild(body);
      messages.appendChild(wrapper);
      scrollDown();
      return body;
    }

    function addBotMessage(text) {
      const body = createBotWrapper();
      body.innerHTML = renderMarkdown(text); 
      scrollDown();
      return body;
    }

    function addTypingIndicator() {
      const el = document.createElement("div");
      el.id = "rag-typing";
      el.innerHTML = '<div class="rag-spin-icon">' + ICON_SPARKLE + '</div><span class="rag-typing-text">Thinking...</span>';
      messages.appendChild(el);
      scrollDown();
    }

    function removeTypingIndicator() {
      const el = document.getElementById("rag-typing");
      if (el) el.remove();
    }

    function scrollDown() {
      messages.scrollTop = messages.scrollHeight;
    }

    function escapeHtml(str) {
      const div = document.createElement("div");
      div.textContent = str;
      return div.innerHTML;
    }

    // ── Send ──────────────────────────────────────────────────────
    async function send() {
      const text = input.value.trim();
      if (!text || isStreaming) return;

      win.classList.add("chatting");

      addUserMessage(text);
      input.value = "";
      input.style.height = "auto";
      sendBtn.disabled = true;
      isStreaming = true;

      addTypingIndicator();

      const body = {
        query: text,
        session_id: sessionId,
        top_k: 3,
        use_history: true,
      };

      let botBody = null;    // the .rag-bot-body element
      let botWrapper = null;  // the .rag-bot-wrapper element
      let collectedText = "";

      try {
        const response = await fetch(cfg.apiUrl + "/api/v1/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        if (!response.ok) {
          throw new Error("Server error: " + response.status);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();

          let currentEvent = null;

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith("data: ") && currentEvent) {
              const data = JSON.parse(line.slice(6));

              if (currentEvent === "metadata") {
                sessionId = data.session_id || sessionId;
              }

              if (currentEvent === "token") {
                if (!botBody) {
                  removeTypingIndicator();
                  botBody = createBotWrapper();
                  botWrapper = botBody.parentElement;
                }
                collectedText += data.content;
                // Render markdown live as tokens stream in
                botBody.innerHTML = renderMarkdown(collectedText);
                scrollDown();
              }

              if (currentEvent === "done") {
                // Final render with full markdown
                if (botBody) {
                  botBody.innerHTML = renderMarkdown(collectedText);
                }
              }

              if (currentEvent === "error") {
                removeTypingIndicator();
                addBotMessage("Sorry, something went wrong. Please try again.");
              }

              currentEvent = null;
            }
          }
        }

        if (!botBody) {
          removeTypingIndicator();
          addBotMessage("Sorry, I couldn't generate a response. Please try again.");
        }
      } catch (err) {
        console.error("[RAG Widget]", err);
        removeTypingIndicator();
        if (!botBody) {
          addBotMessage("Unable to connect. Please check your connection and try again.");
        }
      } finally {
        isStreaming = false;
        sendBtn.disabled = false;
        input.focus();
      }
    }
  }

  // ── Boot ────────────────────────────────────────────────────────────
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initWidget);
  } else {
    initWidget();
  }
})();
