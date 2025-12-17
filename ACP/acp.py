"""
ACP (Agent Communication Protocol) with Flask GUI
==================================================
Web interface to interact with ACP agents.

Run: uv run python ACP/acp.py
Open: http://localhost:5002
"""

import asyncio
import threading
import queue
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, Response
from openai import OpenAI

from acp_sdk.server import Server
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart

# =============================================================================
# Configuration
# =============================================================================

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_MODEL = "google/gemma-3n-e4b"
ACP_PORT = 8001
FLASK_PORT = 5002

# Global
log_queue = queue.Queue()
conversation_history = []

# =============================================================================
# Logging
# =============================================================================

def live_log(message: str, level: str = "info"):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "action": "🔄", "llm": "🤖", "agent": "🤝"}
    icon = icons.get(level, "•")
    log_entry = {"time": timestamp, "icon": icon, "message": message, "level": level}
    log_queue.put(log_entry)
    print(f"[{timestamp}] {icon} {message}", flush=True)

# =============================================================================
# ACP Server with Agents
# =============================================================================

acp_server = Server()


@acp_server.agent(name="echo", description="Echoes back whatever you send")
async def echo_agent(input: list[Message]):
    live_log("Echo agent called", "agent")
    for message in input:
        yield message


@acp_server.agent(name="greeting", description="Greets based on time of day")
async def greeting_agent(input: list[Message]):
    live_log("Greeting agent called", "agent")
    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    for message in input:
        for part in message.parts:
            if hasattr(part, 'content') and part.content:
                name = part.content.strip() or "friend"
                yield Message(parts=[MessagePart(content=f"{greeting}, {name}! How can I assist you today?")])


@acp_server.agent(name="calculator", description="Evaluates math expressions")
async def calculator_agent(input: list[Message]):
    live_log("Calculator agent called", "agent")
    for message in input:
        for part in message.parts:
            if hasattr(part, 'content') and part.content:
                expr = part.content.strip()
                try:
                    allowed = set("0123456789+-*/.() ")
                    if all(c in allowed for c in expr):
                        result = eval(expr)
                        yield Message(parts=[MessagePart(content=f"{expr} = {result}")])
                    else:
                        yield Message(parts=[MessagePart(content="Error: Invalid characters")])
                except Exception as e:
                    yield Message(parts=[MessagePart(content=f"Error: {str(e)}")])


@acp_server.agent(name="llm", description="AI chat powered by LM Studio")
async def llm_agent(input: list[Message]):
    live_log("LLM agent called", "llm")
    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
        messages = [{"role": "system", "content": "You are a helpful AI assistant. Be concise."}]

        for message in input:
            for part in message.parts:
                if hasattr(part, 'content') and part.content:
                    messages.append({"role": "user", "content": part.content})

        live_log("Sending to LLM...", "action")
        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        result = response.choices[0].message.content
        live_log(f"LLM response: {result[:50]}...", "success")
        yield Message(parts=[MessagePart(content=result)])
    except Exception as e:
        live_log(f"LLM Error: {e}", "error")
        yield Message(parts=[MessagePart(content=f"LLM Error: {str(e)}")])


@acp_server.agent(name="summarizer", description="Summarizes text")
async def summarizer_agent(input: list[Message]):
    live_log("Summarizer agent called", "llm")
    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
        for message in input:
            for part in message.parts:
                if hasattr(part, 'content') and part.content:
                    live_log("Summarizing...", "action")
                    response = client.chat.completions.create(
                        model=LM_STUDIO_MODEL,
                        messages=[
                            {"role": "system", "content": "Summarize in 2-3 bullet points. Be concise."},
                            {"role": "user", "content": part.content}
                        ],
                        max_tokens=200
                    )
                    yield Message(parts=[MessagePart(content=response.choices[0].message.content)])
    except Exception as e:
        yield Message(parts=[MessagePart(content=f"Error: {str(e)}")])


@acp_server.agent(name="translator", description="Translates text")
async def translator_agent(input: list[Message]):
    live_log("Translator agent called", "llm")
    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
        for message in input:
            for part in message.parts:
                if hasattr(part, 'content') and part.content:
                    content = part.content
                    if ":" in content:
                        lang, text = content.split(":", 1)
                    else:
                        lang, text = "Lithuanian", content

                    live_log(f"Translating to {lang}...", "action")
                    response = client.chat.completions.create(
                        model=LM_STUDIO_MODEL,
                        messages=[
                            {"role": "system", "content": f"Translate to {lang.strip()}. Only output the translation."},
                            {"role": "user", "content": text.strip()}
                        ],
                        max_tokens=500
                    )
                    yield Message(parts=[MessagePart(content=f"[{lang.strip()}] {response.choices[0].message.content}")])
    except Exception as e:
        yield Message(parts=[MessagePart(content=f"Error: {str(e)}")])


@acp_server.agent(name="coder", description="Generates code snippets")
async def coder_agent(input: list[Message]):
    live_log("Coder agent called", "llm")
    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
        for message in input:
            for part in message.parts:
                if hasattr(part, 'content') and part.content:
                    live_log("Generating code...", "action")
                    response = client.chat.completions.create(
                        model=LM_STUDIO_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a code assistant. Write clean, well-commented code. Always wrap code in markdown code blocks with the language specified."},
                            {"role": "user", "content": part.content}
                        ],
                        max_tokens=1000
                    )
                    yield Message(parts=[MessagePart(content=response.choices[0].message.content)])
    except Exception as e:
        yield Message(parts=[MessagePart(content=f"Error: {str(e)}")])


# =============================================================================
# Flask Web GUI
# =============================================================================

flask_app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ACP Agent Dashboard</title>
    <style>
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-card: rgba(255,255,255,0.05);
            --text-primary: #fff;
            --text-secondary: #888;
            --accent: #8b5cf6;
            --accent-light: #a78bfa;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Fira Code', monospace;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            min-height: 100vh;
            color: var(--text-primary);
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }

        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header h1 span { color: var(--accent); }
        header .subtitle { color: var(--text-secondary); }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

        .card {
            background: var(--bg-card);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
        }
        .card h2 {
            font-size: 1.2em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .agents-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        .agent-btn {
            padding: 15px 10px;
            background: rgba(139, 92, 246, 0.1);
            border: 2px solid transparent;
            border-radius: 10px;
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        .agent-btn:hover {
            background: rgba(139, 92, 246, 0.2);
            border-color: var(--accent);
        }
        .agent-btn.active {
            background: var(--accent);
            color: #000;
            font-weight: bold;
        }
        .agent-btn .icon { font-size: 1.5em; display: block; margin-bottom: 5px; pointer-events: none; }
        .agent-btn .name { font-size: 0.85em; pointer-events: none; }
        .agent-btn * { pointer-events: none; }

        .input-area {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        textarea {
            flex: 1;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            min-height: 100px;
        }
        textarea:focus { outline: none; border-color: var(--accent); }

        .btn {
            padding: 15px 30px;
            background: var(--accent);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn:hover { background: var(--accent-light); transform: translateY(-2px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .response-area {
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 15px;
            min-height: 200px;
            max-height: 400px;
            overflow-y: auto;
        }
        .response-area pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 14px;
            line-height: 1.6;
        }
        .response-area code {
            background: rgba(139, 92, 246, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
        }
        .response-area .code-block {
            background: #1e1e2e;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            overflow-x: auto;
        }

        .log-area {
            background: rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            font-size: 12px;
        }
        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .log-entry .time { color: var(--text-secondary); margin-right: 10px; }

        .history-item {
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .history-item:hover { background: rgba(139, 92, 246, 0.1); }
        .history-item .agent { color: var(--accent); font-weight: bold; }
        .history-item .preview { color: var(--text-secondary); font-size: 0.85em; margin-top: 5px; }

        .status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 15px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
            border-radius: 20px;
            font-size: 0.85em;
        }
        .status .dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .quick-actions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .quick-btn {
            padding: 6px 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            color: var(--text-secondary);
            font-size: 0.8em;
            cursor: pointer;
        }
        .quick-btn:hover { background: rgba(139, 92, 246, 0.2); color: var(--text-primary); }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.active { display: block; }
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(139, 92, 246, 0.3);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤝 <span>ACP</span> Agent Dashboard</h1>
            <p class="subtitle">Agent Communication Protocol - Interactive Control Panel</p>
            <div style="margin-top: 15px;">
                <span class="status"><span class="dot"></span> ACP Server Running on :8001</span>
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <h2>🎯 Select Agent</h2>
                <div class="agents-grid">
                    <button type="button" class="agent-btn active" onclick="selectAgent('llm', this)">
                        <span class="icon">🤖</span>
                        <span class="name">LLM Chat</span>
                    </button>
                    <button type="button" class="agent-btn" onclick="selectAgent('echo', this)">
                        <span class="icon">🔊</span>
                        <span class="name">Echo</span>
                    </button>
                    <button type="button" class="agent-btn" onclick="selectAgent('greeting', this)">
                        <span class="icon">👋</span>
                        <span class="name">Greeting</span>
                    </button>
                    <button type="button" class="agent-btn" onclick="selectAgent('calculator', this)">
                        <span class="icon">🧮</span>
                        <span class="name">Calculator</span>
                    </button>
                    <button type="button" class="agent-btn" onclick="selectAgent('summarizer', this)">
                        <span class="icon">📝</span>
                        <span class="name">Summarizer</span>
                    </button>
                    <button type="button" class="agent-btn" onclick="selectAgent('translator', this)">
                        <span class="icon">🌍</span>
                        <span class="name">Translator</span>
                    </button>
                    <button type="button" class="agent-btn" onclick="selectAgent('coder', this)">
                        <span class="icon">💻</span>
                        <span class="name">Coder</span>
                    </button>
                </div>
                <div id="selectedAgentDisplay" style="margin-bottom: 15px; padding: 10px; background: rgba(139, 92, 246, 0.2); border-radius: 8px; text-align: center;">
                    Selected: <strong id="currentAgent">llm</strong>
                </div>

                <h2>💬 Message</h2>
                <div class="input-area">
                    <textarea id="message" placeholder="Enter your message here..."></textarea>
                </div>
                <div class="quick-actions">
                    <button class="quick-btn" data-text="Hello, world!">Hello World</button>
                    <button class="quick-btn" data-text="2 + 2 * 10">Math: 2+2*10</button>
                    <button class="quick-btn" data-text="Spanish: Good morning">Translate ES</button>
                    <button class="quick-btn" data-text="Write a Python function to reverse a string">Code Example</button>
                    <button class="quick-btn" data-text="Explain quantum computing in simple terms">Explain</button>
                </div>
                <div style="margin-top: 15px; display: flex; gap: 10px;">
                    <button class="btn" id="sendBtn" onclick="sendMessage()">🚀 Send to Agent</button>
                    <button class="btn" style="background: #333;" onclick="clearAll()">🗑️ Clear</button>
                </div>
            </div>

            <div class="card">
                <h2>📤 Response</h2>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <div>Processing with <span id="loadingAgent">agent</span>...</div>
                </div>
                <div class="response-area" id="response">
                    <pre>Select an agent and send a message to get started.</pre>
                </div>
            </div>
        </div>

        <div class="grid" style="margin-top: 20px;">
            <div class="card">
                <h2>📜 Live Logs</h2>
                <div class="log-area" id="logs"></div>
            </div>

            <div class="card">
                <h2>📚 History</h2>
                <div id="history" style="max-height: 300px; overflow-y: auto;">
                    <p style="color: var(--text-secondary);">No history yet.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedAgent = 'llm';
        let history = JSON.parse(localStorage.getItem('acp_history') || '[]');

        // Agent selection function
        function selectAgent(agent, btn) {
            alert('Selected: ' + agent);
            selectedAgent = agent;
            document.querySelectorAll('.agent-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('currentAgent').textContent = agent;
        }

        // Quick actions
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.getElementById('message').value = btn.dataset.text;
            });
        });

        // Send message
        async function sendMessage() {
            const message = document.getElementById('message').value.trim();
            if (!message) return;

            const btn = document.getElementById('sendBtn');
            const loading = document.getElementById('loading');
            const response = document.getElementById('response');

            btn.disabled = true;
            loading.classList.add('active');
            document.getElementById('loadingAgent').textContent = selectedAgent;
            response.innerHTML = '';

            try {
                const res = await fetch('/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ agent: selectedAgent, message: message })
                });

                const data = await res.json();

                if (data.success) {
                    response.innerHTML = formatResponse(data.response);
                    addToHistory(selectedAgent, message, data.response);
                } else {
                    response.innerHTML = `<pre style="color: var(--error);">Error: ${data.error}</pre>`;
                }
            } catch (err) {
                response.innerHTML = `<pre style="color: var(--error);">Error: ${err.message}</pre>`;
            }

            btn.disabled = false;
            loading.classList.remove('active');
        }

        function formatResponse(text) {
            // Format code blocks
            text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, '<div class="code-block"><code>$2</code></div>');
            text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
            text = text.replace(/\n/g, '<br>');
            return `<pre>${text}</pre>`;
        }

        function addToHistory(agent, message, response) {
            history.unshift({ agent, message, response, time: new Date().toISOString() });
            if (history.length > 20) history.pop();
            localStorage.setItem('acp_history', JSON.stringify(history));
            renderHistory();
        }

        function renderHistory() {
            const container = document.getElementById('history');
            if (history.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary);">No history yet.</p>';
                return;
            }
            container.innerHTML = history.map((item, i) => `
                <div class="history-item" onclick="loadHistory(${i})">
                    <span class="agent">${item.agent}</span>
                    <span style="color: var(--text-secondary); font-size: 0.8em;">${new Date(item.time).toLocaleTimeString()}</span>
                    <div class="preview">${item.message.substring(0, 50)}${item.message.length > 50 ? '...' : ''}</div>
                </div>
            `).join('');
        }

        function loadHistory(index) {
            const item = history[index];
            document.getElementById('message').value = item.message;
            document.getElementById('response').innerHTML = formatResponse(item.response);

            // Select the agent
            document.querySelectorAll('.agent-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.agent === item.agent);
            });
            selectedAgent = item.agent;
        }

        function clearAll() {
            document.getElementById('message').value = '';
            document.getElementById('response').innerHTML = '<pre>Select an agent and send a message to get started.</pre>';
        }

        // Live logs
        const logsContainer = document.getElementById('logs');
        const eventSource = new EventSource('/stream-logs');
        eventSource.onmessage = (event) => {
            const log = JSON.parse(event.data);
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="time">${log.time}</span>${log.icon} ${log.message}`;
            logsContainer.appendChild(entry);
            logsContainer.scrollTop = logsContainer.scrollHeight;
        };

        // Keyboard shortcut
        document.getElementById('message').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                sendMessage();
            }
        });

        // Init
        renderHistory();
    </script>
</body>
</html>
'''


@flask_app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@flask_app.route('/stream-logs')
def stream_logs():
    def generate():
        while True:
            try:
                log = log_queue.get(timeout=30)
                yield f"data: {jsonify(log).get_data(as_text=True)}\n\n"
            except:
                yield f"data: {{}}\n\n"
    return Response(generate(), mimetype='text/event-stream')


@flask_app.route('/execute', methods=['POST'])
def execute():
    data = request.json
    agent_name = data.get('agent', 'echo')
    message = data.get('message', '')

    live_log(f"Executing agent: {agent_name}", "action")

    try:
        # Run async client in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def call_agent():
            async with Client(base_url=f"http://localhost:{ACP_PORT}") as client:
                run = await client.run_sync(
                    agent=agent_name,
                    input=[Message(parts=[MessagePart(content=message)])]
                )
                results = []
                for msg in run.output:
                    for part in msg.parts:
                        if hasattr(part, 'content'):
                            results.append(part.content)
                return "\n".join(results)

        result = loop.run_until_complete(call_agent())
        loop.close()

        live_log(f"Agent {agent_name} completed", "success")
        return jsonify({"success": True, "response": result})

    except Exception as e:
        live_log(f"Error: {str(e)}", "error")
        return jsonify({"success": False, "error": str(e)})


# =============================================================================
# Main
# =============================================================================

def run_acp_server():
    """Run ACP server in background thread"""
    live_log(f"Starting ACP server on port {ACP_PORT}...", "info")
    acp_server.run(port=ACP_PORT)


def main():
    print(f"""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   🤝 ACP Agent Dashboard                                   ║
║                                                            ║
║   Web UI:     http://localhost:{FLASK_PORT}                      ║
║   ACP Server: http://localhost:{ACP_PORT}                       ║
║                                                            ║
║   Available Agents:                                        ║
║   • llm        - AI Chat (LM Studio)                       ║
║   • echo       - Echo messages                             ║
║   • greeting   - Time-based greeting                       ║
║   • calculator - Math expressions                          ║
║   • summarizer - Text summarization                        ║
║   • translator - Language translation                      ║
║   • coder      - Code generation                           ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Start ACP server in background
    acp_thread = threading.Thread(target=run_acp_server, daemon=True)
    acp_thread.start()

    import time
    time.sleep(2)  # Wait for ACP server to start

    live_log("ACP server started", "success")
    live_log(f"Starting Flask UI on port {FLASK_PORT}...", "info")

    # Start Flask
    flask_app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
