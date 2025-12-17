import requests
import time
import logging
import threading
import json
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, Response
from bs4 import BeautifulSoup
from python_a2a import A2AServer, agent, skill, run_server, TaskStatus, TaskState
from openai import OpenAI
from playwright.sync_api import sync_playwright
import queue

# LM Studio connection
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_MODEL = "google/gemma-3n-e4b"

# Extraction settings (configurable)
MAX_CONTENT_CHARS = 100000  # Max characters to extract from page
MAX_HEADLINES = 100  # Max headlines to extract
MAX_SCROLL_ITERATIONS = 10  # Scroll attempts for infinite scroll
SCROLL_WAIT_MS = 1500  # Wait between scrolls
LOAD_MORE_ATTEMPTS = 5  # How many times to try "Load more"

# Global log queue for streaming to web
log_queue = queue.Queue()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def live_log(message: str, level: str = "info"):
    """Log message and add to queue for web streaming"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "action": "🔄", "llm": "🤖", "web": "🌐", "parse": "📄", "browser": "🖥️"}
    icon = icons.get(level, "•")
    log_entry = {"time": timestamp, "icon": icon, "message": message, "level": level}
    log_queue.put(log_entry)
    print(f"[{timestamp}] {icon} {message}", flush=True)

# Flask Web App
web_app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal Web Scraper Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        h1 span { color: #00d4ff; }
        .subtitle { color: #888; font-size: 1.1em; }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 { margin-bottom: 20px; color: #00d4ff; font-size: 1.3em; }

        .btn-group { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .btn-primary { background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); color: #000; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0,212,255,0.4); }
        .btn-secondary { background: rgba(255,255,255,0.1); color: #fff; }
        .btn-secondary:hover { background: rgba(255,255,255,0.2); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; }

        input, textarea, select {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 1em;
            margin-bottom: 15px;
        }
        input:focus, textarea:focus, select:focus { outline: none; border-color: #00d4ff; }
        textarea { min-height: 100px; resize: vertical; }
        select { cursor: pointer; }

        .log-container {
            background: #000;
            border-radius: 10px;
            padding: 15px;
            height: 400px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.85em;
            line-height: 1.6;
        }
        .log-entry { padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .log-time { color: #666; margin-right: 10px; }
        .log-info { color: #00d4ff; }
        .log-success { color: #00ff88; }
        .log-warning { color: #ffaa00; }
        .log-error { color: #ff4444; }
        .log-action { color: #aa88ff; }
        .log-llm { color: #ff88ff; }
        .log-web { color: #00d4ff; }
        .log-browser { color: #ff9900; }

        .result-container {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 20px;
            min-height: 200px;
            max-height: 500px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
            line-height: 1.6;
        }

        .status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
        .status-ready { background: rgba(0,255,136,0.2); color: #00ff88; }
        .status-busy { background: rgba(255,170,0,0.2); color: #ffaa00; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; animation: pulse 1.5s infinite; }
        .status-ready .status-dot { background: #00ff88; }
        .status-busy .status-dot { background: #ffaa00; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        .quick-actions { margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); }
        .quick-actions label { display: block; margin-bottom: 8px; color: #888; font-size: 0.9em; }

        .preset-btns { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; }
        .preset-btn {
            padding: 6px 12px;
            font-size: 0.85em;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 6px;
            color: #ccc;
            cursor: pointer;
        }
        .preset-btn:hover { background: rgba(255,255,255,0.15); color: #fff; }

        .toggle-group {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            align-items: center;
        }
        .toggle-group label { margin: 0; color: #fff; }
        .toggle {
            width: 50px;
            height: 26px;
            background: rgba(255,255,255,0.2);
            border-radius: 13px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }
        .toggle.active { background: #00d4ff; }
        .toggle::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            background: #fff;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: left 0.3s;
        }
        .toggle.active::after { left: 26px; }

        .mode-indicator {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .mode-simple { background: rgba(0,212,255,0.2); color: #00d4ff; }
        .mode-browser { background: rgba(255,153,0,0.2); color: #ff9900; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌐 Universal <span>Web Scraper</span></h1>
            <p class="subtitle">Powered by LM Studio + Playwright + A2A Protocol</p>
        </header>

        <div id="status" class="status status-ready">
            <span class="status-dot"></span>
            <span id="status-text">Ready</span>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📤 Commands</h2>

                <div class="toggle-group">
                    <label>Scraping Mode:</label>
                    <div class="toggle" id="modeToggle" onclick="toggleMode()"></div>
                    <span id="modeLabel" class="mode-indicator mode-simple">Simple HTTP</span>
                </div>

                <div class="quick-actions" style="margin-top: 0; padding-top: 0; border-top: none;">
                    <label>Website URL to scrape:</label>
                    <input type="text" id="urlInput" placeholder="https://example.com" value="https://news.ycombinator.com">

                    <label>Quick presets:</label>
                    <div class="preset-btns">
                        <button class="preset-btn" onclick="setUrl('https://news.ycombinator.com')">Hacker News</button>
                        <button class="preset-btn" onclick="setUrl('https://lite.cnn.com')">CNN Lite</button>
                        <button class="preset-btn" onclick="setUrl('https://text.npr.org')">NPR Text</button>
                        <button class="preset-btn" onclick="setUrl('https://lobste.rs')">Lobsters</button>
                        <button class="preset-btn" onclick="setUrl('https://www.bbc.com/news')">BBC (JS)</button>
                        <button class="preset-btn" onclick="setUrl('https://techcrunch.com')">TechCrunch (JS)</button>
                    </div>
                </div>

                <div class="btn-group">
                    <button class="btn btn-primary" onclick="scrapeHeadlines()">
                        🗞️ Get Headlines
                    </button>
                    <button class="btn btn-primary" onclick="scrapeAndAnalyze()">
                        🔍 Analyze Content
                    </button>
                    <button class="btn btn-secondary" onclick="scrapeFullPage()">
                        📄 Full Page Text
                    </button>
                </div>

                <div class="quick-actions">
                    <label>Summarize specific article URL:</label>
                    <input type="text" id="articleUrl" placeholder="https://example.com/article/...">
                    <button class="btn btn-secondary" onclick="summarizeArticle()">📝 Summarize Article</button>
                </div>

                <div class="quick-actions">
                    <label>Ask AI about the content:</label>
                    <textarea id="customInput" placeholder="Ask anything about the scraped content..."></textarea>
                    <button class="btn btn-primary" onclick="askQuestion()">
                        🚀 Ask AI
                    </button>
                </div>
            </div>

            <div class="card">
                <h2>📋 Live Log</h2>
                <div class="log-container" id="logContainer"></div>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h2>📤 Result</h2>
            <div class="result-container" id="resultContainer">
                Results will appear here...
            </div>
        </div>
    </div>

    <script>
        let isProcessing = false;
        let useBrowser = false;

        const eventSource = new EventSource('/stream-logs');
        eventSource.onmessage = function(e) {
            const log = JSON.parse(e.data);
            if (log.message !== 'heartbeat') addLogEntry(log);
        };

        function addLogEntry(log) {
            const container = document.getElementById('logContainer');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="log-time">${log.time}</span><span class="log-${log.level}">${log.icon} ${log.message}</span>`;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }

        function toggleMode() {
            useBrowser = !useBrowser;
            const toggle = document.getElementById('modeToggle');
            const label = document.getElementById('modeLabel');
            if (useBrowser) {
                toggle.classList.add('active');
                label.textContent = '🖥️ Browser (JS)';
                label.className = 'mode-indicator mode-browser';
            } else {
                toggle.classList.remove('active');
                label.textContent = 'Simple HTTP';
                label.className = 'mode-indicator mode-simple';
            }
        }

        function setStatus(busy) {
            isProcessing = busy;
            const status = document.getElementById('status');
            const statusText = document.getElementById('status-text');
            const buttons = document.querySelectorAll('.btn');

            if (busy) {
                status.className = 'status status-busy';
                statusText.textContent = 'Processing...';
                buttons.forEach(b => b.disabled = true);
            } else {
                status.className = 'status status-ready';
                statusText.textContent = 'Ready';
                buttons.forEach(b => b.disabled = false);
            }
        }

        function setUrl(url) {
            document.getElementById('urlInput').value = url;
        }

        async function sendCommand(cmd, data = {}) {
            if (isProcessing) return;
            setStatus(true);

            try {
                const response = await fetch('/execute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({command: cmd, useBrowser: useBrowser, ...data})
                });
                const result = await response.json();
                document.getElementById('resultContainer').textContent = result.result || result.error;
            } catch (err) {
                document.getElementById('resultContainer').textContent = 'Error: ' + err.message;
            }

            setStatus(false);
        }

        function scrapeHeadlines() {
            const url = document.getElementById('urlInput').value.trim();
            if (url) sendCommand('headlines', {url: url});
        }

        function scrapeAndAnalyze() {
            const url = document.getElementById('urlInput').value.trim();
            if (url) sendCommand('analyze', {url: url});
        }

        function scrapeFullPage() {
            const url = document.getElementById('urlInput').value.trim();
            if (url) sendCommand('fullpage', {url: url});
        }

        function summarizeArticle() {
            const url = document.getElementById('articleUrl').value.trim();
            if (url) sendCommand('summarize', {url: url});
        }

        function askQuestion() {
            const question = document.getElementById('customInput').value.trim();
            const url = document.getElementById('urlInput').value.trim();
            if (question) sendCommand('ask', {url: url, question: question});
        }

        document.getElementById('customInput').addEventListener('keypress', e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askQuestion(); }
        });
    </script>
</body>
</html>
'''

@web_app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@web_app.route('/stream-logs')
def stream_logs():
    def generate():
        while True:
            try:
                log = log_queue.get(timeout=30)
                yield f"data: {json.dumps(log)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'time': '', 'icon': '', 'message': 'heartbeat', 'level': 'info'})}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@web_app.route('/execute', methods=['POST'])
def execute():
    data = request.json
    command = data.get('command', '')
    url = data.get('url', '')
    question = data.get('question', '')
    use_browser = data.get('useBrowser', False)

    mode = "Browser" if use_browser else "HTTP"
    live_log(f"Command: {command} | Mode: {mode} | URL: {url[:50]}...", "action")

    try:
        result = agent_instance.execute_command(command, url, question, use_browser)
        return jsonify({'result': result})
    except Exception as e:
        live_log(f"Error: {e}", "error")
        return jsonify({'error': str(e)})


@agent(
    name="UniversalScraper",
    description="Universal web scraper with LM Studio LLM and Playwright browser support.",
    version="2.0.0"
)
class UniversalScraperAgent(A2AServer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        live_log("Initializing Universal Scraper Agent...", "action")
        live_log(f"LM Studio: {LM_STUDIO_URL}", "info")
        live_log(f"Model: {LM_STUDIO_MODEL}", "info")

        self.llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        self.last_scraped_content = ""
        self.last_scraped_url = ""

        try:
            live_log("Testing LM Studio connection...", "action")
            models = self.llm_client.models.list()
            live_log(f"Connected! Models: {[m.id for m in models.data]}", "success")
        except Exception as e:
            live_log(f"LM Studio connection failed: {e}", "error")

        live_log("Playwright browser ready!", "browser")
        live_log("Agent ready!", "success")

    def execute_command(self, command: str, url: str = "", question: str = "", use_browser: bool = False) -> str:
        """Execute a command from web interface"""
        live_log(f"Executing: {command}", "action")

        if command == "headlines":
            return self.get_headlines(url, use_browser)
        elif command == "analyze":
            return self.analyze_page(url, use_browser)
        elif command == "fullpage":
            return self.get_full_text(url, use_browser)
        elif command == "summarize":
            return self.summarize_article(url, use_browser)
        elif command == "ask":
            return self.ask_about_content(url, question, use_browser)
        else:
            return f"Unknown command: {command}"

    def _fetch_with_browser(self, url: str, wait_time: int = 3, scroll: bool = True, deep_scroll: bool = False) -> tuple:
        """Fetch page using Playwright headless browser with aggressive content loading"""
        live_log(f"🖥️ Launching headless browser...", "browser")

        try:
            with sync_playwright() as p:
                live_log("Starting Chromium...", "browser")
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080}
                )
                page = context.new_page()

                live_log(f"Navigating to {url}...", "browser")
                start = time.time()
                page.goto(url, wait_until="networkidle", timeout=60000)
                live_log(f"Page loaded in {time.time()-start:.2f}s", "success")

                # Wait for dynamic content
                live_log(f"Waiting {wait_time}s for JS content...", "browser")
                page.wait_for_timeout(wait_time * 1000)

                # Scroll to load more content - now with configurable iterations
                if scroll:
                    scroll_count = MAX_SCROLL_ITERATIONS if deep_scroll else max(5, MAX_SCROLL_ITERATIONS // 2)
                    live_log(f"Scrolling page {scroll_count} times to load content...", "browser")
                    last_height = 0

                    for i in range(scroll_count):
                        # Scroll to bottom
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(SCROLL_WAIT_MS)

                        # Check if we've reached the end (no new content loaded)
                        new_height = page.evaluate("document.body.scrollHeight")
                        if new_height == last_height and i > 2:
                            live_log(f"Scroll {i+1}/{scroll_count} - reached end of content", "info")
                            break
                        last_height = new_height
                        live_log(f"Scroll {i+1}/{scroll_count} complete (height: {new_height:,}px)", "info")

                # Extended "Load more" button detection with multiple languages and patterns
                load_more_selectors = [
                    # English
                    'button:has-text("Load more")', 'button:has-text("Show more")',
                    'button:has-text("More stories")', 'button:has-text("See more")',
                    'button:has-text("View more")', 'button:has-text("Read more")',
                    'a:has-text("Load more")', 'a:has-text("Show more")',
                    'a:has-text("More")', 'a:has-text("Next")',
                    # Class-based
                    '[class*="load-more"]', '[class*="show-more"]', '[class*="loadmore"]',
                    '[class*="load_more"]', '[class*="show_more"]',
                    '[class*="pagination"] a', '[class*="next-page"]',
                    '[data-testid*="load-more"]', '[data-testid*="show-more"]',
                    # Button with icons
                    'button[aria-label*="more"]', 'button[aria-label*="load"]',
                    # Lithuanian
                    'button:has-text("Daugiau")', 'a:has-text("Daugiau")',
                    'button:has-text("Rodyti daugiau")', 'a:has-text("Kitas")',
                    # German
                    'button:has-text("Mehr laden")', 'button:has-text("Mehr anzeigen")',
                    # French
                    'button:has-text("Voir plus")', 'button:has-text("Charger plus")',
                    # Spanish
                    'button:has-text("Ver más")', 'button:has-text("Cargar más")',
                ]

                # Try clicking load more buttons multiple times
                load_more_clicks = 0
                for attempt in range(LOAD_MORE_ATTEMPTS):
                    clicked = False
                    for selector in load_more_selectors:
                        try:
                            locator = page.locator(selector)
                            if locator.count() > 0 and locator.first.is_visible():
                                live_log(f"Found 'Load more' button (attempt {attempt+1}), clicking...", "browser")
                                locator.first.click()
                                page.wait_for_timeout(2000)
                                load_more_clicks += 1
                                clicked = True

                                # Scroll after clicking to load new content
                                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                                page.wait_for_timeout(1000)
                                break
                        except:
                            continue

                    if not clicked:
                        break

                if load_more_clicks > 0:
                    live_log(f"Clicked 'Load more' {load_more_clicks} times", "success")

                # Handle pagination - try to get content from multiple pages
                # (optional, just logs if pagination exists)
                pagination_selectors = [
                    '[class*="pagination"]', '[class*="pager"]',
                    'nav[aria-label*="pagination"]', '.page-numbers'
                ]
                for sel in pagination_selectors:
                    try:
                        if page.locator(sel).count() > 0:
                            live_log(f"Pagination detected - consider scraping multiple pages", "info")
                            break
                    except:
                        continue

                # Get page content
                html = page.content()
                live_log(f"Got {len(html):,} bytes of rendered HTML", "success")

                browser.close()

                soup = BeautifulSoup(html, "html.parser")

                # Remove unwanted elements (but keep more content)
                for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
                    tag.decompose()

                return soup, html

        except Exception as e:
            live_log(f"Browser fetch failed: {e}", "error")
            return None, str(e)

    def _fetch_with_requests(self, url: str) -> tuple:
        """Fetch page using simple HTTP requests"""
        live_log(f"Fetching with HTTP: {url}", "web")

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

        try:
            start = time.time()
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            live_log(f"HTTP {response.status_code} in {time.time()-start:.2f}s ({len(response.text):,} bytes)", "success")

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript", "iframe", "svg"]):
                tag.decompose()

            return soup, response.text

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [401, 403]:
                live_log(f"Access blocked ({e.response.status_code}). Try Browser mode!", "error")
                return None, f"This website blocks automated access. Enable 'Browser (JS)' mode to try with a real browser."
            live_log(f"HTTP Error: {e}", "error")
            return None, str(e)
        except Exception as e:
            live_log(f"Fetch failed: {e}", "error")
            return None, str(e)

    def _fetch_page(self, url: str, use_browser: bool = False) -> tuple:
        """Fetch page using either browser or HTTP"""
        if use_browser:
            return self._fetch_with_browser(url)
        else:
            return self._fetch_with_requests(url)

    def _extract_headlines(self, soup, url: str) -> list:
        """Extract headlines/titles from any website"""
        live_log("Extracting headlines...", "parse")
        headlines = []
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

        # Extended selectors for better coverage
        selectors = [
            # Article-based selectors
            'article h1', 'article h2', 'article h3',
            'article a', 'article .title',
            # Common class patterns
            '.headline', '.title', '.post-title', '.entry-title', '.article-title',
            '[class*="headline"]', '[class*="title"]', '[class*="heading"]',
            '.titleline a', '.storylink',
            '[data-click-id="body"] h3',
            'h1 a', 'h2 a', 'h3 a',
            '.card-title', '.news-title', '.story-title',
            # News site specific
            '.teaser-title', '.feed-title', '.item-title',
            '[class*="card"] h2', '[class*="card"] h3',
            '[class*="story"] h2', '[class*="story"] h3',
            '[class*="post"] h2', '[class*="post"] h3',
            '[class*="article"] h2', '[class*="article"] h3',
            # Link-based (for link-heavy sites like HN)
            '.athing .titleline > a',
            '.submission .title a',
        ]

        seen_titles = set()

        for selector in selectors:
            if len(headlines) >= MAX_HEADLINES:
                break
            try:
                elements = soup.select(selector)
                for el in elements[:100]:  # Check more elements per selector
                    if len(headlines) >= MAX_HEADLINES:
                        break

                    title = el.get_text(strip=True)
                    title = re.sub(r'\s+', ' ', title)

                    if title and 10 < len(title) < 500 and title not in seen_titles:
                        seen_titles.add(title)

                        link = None
                        if el.name == 'a':
                            link = el.get('href')
                        else:
                            link_el = el.find('a')
                            if link_el:
                                link = link_el.get('href')

                        if link and not link.startswith('http'):
                            link = urljoin(base_url, link)

                        headlines.append({"title": title, "url": link})
                        if len(headlines) <= 50:  # Log first 50 only
                            live_log(f"Found: {title[:60]}...", "info")
            except Exception:
                continue

        if len(headlines) < 10:
            live_log("Trying fallback extraction (h1/h2/h3)...", "parse")
            for tag in soup.find_all(['h1', 'h2', 'h3'], limit=200):
                if len(headlines) >= MAX_HEADLINES:
                    break

                title = tag.get_text(strip=True)
                title = re.sub(r'\s+', ' ', title)

                if title and 10 < len(title) < 500 and title not in seen_titles:
                    seen_titles.add(title)
                    link = None
                    a_tag = tag.find('a') or tag.find_parent('a')
                    if a_tag:
                        link = a_tag.get('href')
                        if link and not link.startswith('http'):
                            link = urljoin(base_url, link)

                    headlines.append({"title": title, "url": link})

        if len(headlines) < 10:
            live_log("Trying list-based extraction...", "parse")
            for li in soup.find_all('li', limit=200):
                if len(headlines) >= MAX_HEADLINES:
                    break

                a_tag = li.find('a')
                if a_tag:
                    title = a_tag.get_text(strip=True)
                    title = re.sub(r'\s+', ' ', title)

                    if title and 15 < len(title) < 300 and title not in seen_titles:
                        seen_titles.add(title)
                        link = a_tag.get('href')
                        if link and not link.startswith('http'):
                            link = urljoin(base_url, link)
                        headlines.append({"title": title, "url": link})

        # Also try to get all standalone links that look like article titles
        if len(headlines) < 20:
            live_log("Trying link-based extraction...", "parse")
            for a in soup.find_all('a', limit=500):
                if len(headlines) >= MAX_HEADLINES:
                    break

                title = a.get_text(strip=True)
                title = re.sub(r'\s+', ' ', title)

                # Filter for article-like links (reasonable length, not navigation)
                if title and 25 < len(title) < 200 and title not in seen_titles:
                    # Skip common navigation patterns
                    if any(skip in title.lower() for skip in ['sign in', 'log in', 'subscribe', 'menu', 'home', 'contact']):
                        continue

                    seen_titles.add(title)
                    link = a.get('href')
                    if link and not link.startswith('http'):
                        link = urljoin(base_url, link)
                    headlines.append({"title": title, "url": link})

        live_log(f"Total headlines extracted: {len(headlines)}", "success")
        return headlines[:MAX_HEADLINES]

    def _extract_main_content(self, soup) -> str:
        """Extract main content from page"""
        live_log("Extracting main content...", "parse")

        content_selectors = [
            'article', 'main', '.content', '.post-content', '.article-content',
            '.entry-content', '.story-body', '#content', '.body-text'
        ]

        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = elements[0].get_text(separator='\n', strip=True)
                if len(content) > 200:
                    break

        if len(content) < 200:
            paragraphs = soup.find_all('p')
            content = '\n'.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)

        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)

        live_log(f"Extracted {len(content):,} characters", "success")
        return content[:10000]

    def _ask_llm(self, prompt: str) -> str:
        """Send prompt to LM Studio"""
        live_log(f"Calling LLM ({len(prompt):,} chars)...", "llm")

        try:
            live_log("Waiting for LLM response...", "llm")
            start = time.time()

            response = self.llm_client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes web content. Be concise and informative."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            result = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else "N/A"
            live_log(f"LLM done in {time.time()-start:.2f}s (tokens: {tokens})", "success")
            return result

        except Exception as e:
            live_log(f"LLM failed: {e}", "error")
            return f"LLM Error: {str(e)}"

    @skill(name="get_headlines", description="Extract headlines from any website")
    def get_headlines(self, url: str, use_browser: bool = False) -> str:
        live_log(f"SKILL: get_headlines (browser={use_browser})", "action")

        soup, raw = self._fetch_page(url, use_browser)
        if soup is None:
            return f"Failed to fetch page: {raw}"

        headlines = self._extract_headlines(soup, url)

        if not headlines:
            return f"No headlines found on {url}. Try enabling 'Browser (JS)' mode for dynamic content."

        result = f"📰 Headlines from {urlparse(url).netloc}:\n\n"
        for i, h in enumerate(headlines, 1):
            result += f"{i}. {h['title']}\n"
            if h.get('url'):
                result += f"   🔗 {h['url']}\n"

        self.last_scraped_content = result
        self.last_scraped_url = url

        return result

    @skill(name="analyze_page", description="Analyze page content with AI")
    def analyze_page(self, url: str, use_browser: bool = False) -> str:
        live_log(f"SKILL: analyze_page (browser={use_browser})", "action")

        soup, raw = self._fetch_page(url, use_browser)
        if soup is None:
            return f"Failed to fetch page: {raw}"

        headlines = self._extract_headlines(soup, url)
        content = self._extract_main_content(soup)

        headlines_text = "\n".join(f"- {h['title']}" for h in headlines[:15])

        prompt = f"""Analyze this webpage content from {url}:

HEADLINES:
{headlines_text}

CONTENT EXCERPT:
{content[:4000]}

Please provide:
1. What type of website/page is this?
2. Main topics or themes
3. Key highlights or interesting points
4. Brief summary (2-3 sentences)"""

        result = self._ask_llm(prompt)

        self.last_scraped_content = content
        self.last_scraped_url = url

        return result

    @skill(name="get_full_text", description="Get full page text content")
    def get_full_text(self, url: str, use_browser: bool = False) -> str:
        live_log(f"SKILL: get_full_text (browser={use_browser})", "action")

        soup, raw = self._fetch_page(url, use_browser)
        if soup is None:
            return f"Failed to fetch page: {raw}"

        content = self._extract_main_content(soup)

        self.last_scraped_content = content
        self.last_scraped_url = url

        return f"📄 Content from {urlparse(url).netloc}:\n\n{content}"

    @skill(name="summarize_article", description="Summarize a specific article")
    def summarize_article(self, url: str, use_browser: bool = False) -> str:
        live_log(f"SKILL: summarize_article (browser={use_browser})", "action")

        soup, raw = self._fetch_page(url, use_browser)
        if soup is None:
            return f"Failed to fetch page: {raw}"

        content = self._extract_main_content(soup)

        if len(content) < 100:
            return "Could not extract enough content from this article."

        prompt = f"""Summarize this article in 3-5 sentences:

{content}

Focus on the key points and main message."""

        return self._ask_llm(prompt)

    @skill(name="ask_about_content", description="Ask AI about scraped content")
    def ask_about_content(self, url: str, question: str, use_browser: bool = False) -> str:
        live_log(f"SKILL: ask_about_content (browser={use_browser})", "action")

        if url != self.last_scraped_url or not self.last_scraped_content:
            soup, raw = self._fetch_page(url, use_browser)
            if soup is None:
                return f"Failed to fetch page: {raw}"
            self.last_scraped_content = self._extract_main_content(soup)
            self.last_scraped_url = url

        prompt = f"""Based on this web content from {url}:

{self.last_scraped_content[:5000]}

User question: {question}

Please answer the question based on the content above."""

        return self._ask_llm(prompt)

    def handle_task(self, task):
        """Handle A2A tasks"""
        live_log("📨 A2A TASK RECEIVED", "action")

        input_content = task.message.content
        input_text = input_content.get("text", "") if isinstance(input_content, dict) else str(input_content)
        live_log(f"Task: {input_text}", "info")

        if "http" in input_text:
            urls = re.findall(r'https?://\S+', input_text)
            if urls:
                result = self.get_headlines(urls[0], use_browser=False)
            else:
                result = "Please provide a valid URL"
        else:
            result = "Please provide a URL to scrape"

        task.artifacts = [{"parts": [{"type": "text", "text": result}]}]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        live_log("✅ Task completed", "success")
        return task


# Global agent instance
agent_instance = None


def run_web_server():
    """Run Flask web server"""
    live_log("Starting web server on http://localhost:5001", "info")
    web_app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)


def run_a2a_server():
    """Run A2A server"""
    live_log("Starting A2A server on port 8000", "info")
    run_server(agent_instance, port=8000)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🌐 UNIVERSAL WEB SCRAPER AGENT")
    print("  📡 LM Studio + Playwright + A2A")
    print("=" * 60 + "\n")

    print(f"Configuration:")
    print(f"  • LM Studio:  {LM_STUDIO_URL}")
    print(f"  • Model:      {LM_STUDIO_MODEL}")
    print(f"  • A2A Port:   8000")
    print(f"  • Web UI:     http://localhost:5001")
    print(f"  • Browser:    Playwright Chromium")
    print()

    # Create agent
    agent_instance = UniversalScraperAgent(url="http://localhost:8000")

    # Start web server in background thread
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()

    print("\n" + "=" * 60)
    print("  🟢 SERVERS READY!")
    print("  🌐 Open http://localhost:5001 in your browser")
    print("=" * 60 + "\n")

    # Run A2A server in main thread
    run_a2a_server()
