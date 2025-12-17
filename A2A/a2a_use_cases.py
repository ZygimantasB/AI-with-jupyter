import requests
import time
import logging
import threading
import json
import re
import base64
import xml.etree.ElementTree as ET
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

# Settings
MAX_CONTENT_CHARS = 100000
MAX_HEADLINES = 100
MAX_SCROLL_ITERATIONS = 10
SCROLL_WAIT_MS = 1500
MAX_PAGES = 5

# Global storage
log_queue = queue.Queue()
scheduled_jobs = {}  # Store scheduled scraping jobs
content_cache = {}  # Store previous scrapes for diff

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

def live_log(message: str, level: str = "info"):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "action": "🔄", "llm": "🤖", "web": "🌐", "parse": "📄", "browser": "🖥️", "page": "📑", "schedule": "⏰", "diff": "🔄", "api": "🔌"}
    icon = icons.get(level, "•")
    log_entry = {"time": timestamp, "icon": icon, "message": message, "level": level}
    log_queue.put(log_entry)
    print(f"[{timestamp}] {icon} {message}", flush=True)

web_app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal Web Scraper Pro v3.0</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: rgba(255,255,255,0.05);
            --text-primary: #fff;
            --text-secondary: #888;
            --accent: #00d4ff;
            --accent-dark: #0099cc;
            --success: #00ff88;
            --warning: #ffaa00;
            --error: #ff4444;
        }
        .light-theme {
            --bg-primary: #f5f5f5;
            --bg-secondary: #e8e8e8;
            --bg-card: rgba(0,0,0,0.05);
            --text-primary: #1a1a2e;
            --text-secondary: #666;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            min-height: 100vh;
            color: var(--text-primary);
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 15px; }
        header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 15px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px;
        }
        .logo h1 { font-size: 1.6em; }
        .logo h1 span { color: var(--accent); }
        .logo .subtitle { color: var(--text-secondary); font-size: 0.85em; }
        .header-actions { display: flex; gap: 10px; align-items: center; }

        .tabs { display: flex; gap: 3px; margin-bottom: 15px; flex-wrap: wrap; }
        .tab {
            padding: 8px 14px; background: var(--bg-card); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px 6px 0 0; cursor: pointer; color: var(--text-secondary); font-size: 0.85em;
        }
        .tab:hover { background: rgba(255,255,255,0.1); }
        .tab.active { background: var(--accent); color: #000; font-weight: 600; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

        .card {
            background: var(--bg-card); border-radius: 10px; padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 { margin-bottom: 12px; color: var(--accent); font-size: 1.1em; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
        .card h3 { margin: 12px 0 8px; color: var(--text-secondary); font-size: 0.85em; }

        .btn-group { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
        .btn {
            padding: 8px 14px; border: none; border-radius: 5px; cursor: pointer;
            font-size: 0.85em; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;
        }
        .btn-primary { background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%); color: #000; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(0,212,255,0.3); }
        .btn-secondary { background: rgba(255,255,255,0.1); color: var(--text-primary); }
        .btn-secondary:hover { background: rgba(255,255,255,0.2); }
        .btn-success { background: var(--success); color: #000; }
        .btn-warning { background: var(--warning); color: #000; }
        .btn-small { padding: 5px 10px; font-size: 0.75em; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }

        input, textarea, select {
            width: 100%; padding: 8px 10px; border: 1px solid rgba(255,255,255,0.2);
            border-radius: 5px; background: rgba(0,0,0,0.3); color: var(--text-primary);
            font-size: 0.9em; margin-bottom: 8px;
        }
        input:focus, textarea:focus, select:focus { outline: none; border-color: var(--accent); }
        textarea { min-height: 70px; resize: vertical; font-family: monospace; }
        label { display: block; margin-bottom: 4px; color: var(--text-secondary); font-size: 0.8em; }

        .log-container {
            background: #000; border-radius: 6px; padding: 10px; height: 300px;
            overflow-y: auto; font-family: monospace; font-size: 0.75em; line-height: 1.4;
        }
        .log-entry { padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .log-time { color: #666; margin-right: 6px; }
        .log-info { color: var(--accent); }
        .log-success { color: var(--success); }
        .log-warning { color: var(--warning); }
        .log-error { color: var(--error); }
        .log-action { color: #aa88ff; }
        .log-llm { color: #ff88ff; }
        .log-browser { color: #ff9900; }
        .log-schedule { color: #88ff88; }
        .log-diff { color: #ffff88; }
        .log-api { color: #88ffff; }

        .result-container {
            background: rgba(0,0,0,0.3); border-radius: 6px; padding: 12px;
            min-height: 120px; max-height: 350px; overflow-y: auto;
            white-space: pre-wrap; font-family: monospace; font-size: 0.8em; line-height: 1.4;
        }

        .toggle-group { display: flex; gap: 8px; margin-bottom: 10px; align-items: center; }
        .toggle-group label { margin: 0; }
        .toggle {
            width: 40px; height: 22px; background: rgba(255,255,255,0.2);
            border-radius: 11px; position: relative; cursor: pointer;
        }
        .toggle.active { background: var(--accent); }
        .toggle::after {
            content: ''; position: absolute; width: 18px; height: 18px;
            background: #fff; border-radius: 50%; top: 2px; left: 2px; transition: left 0.2s;
        }
        .toggle.active::after { left: 20px; }

        .preset-btns { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; }
        .preset-btn {
            padding: 4px 8px; font-size: 0.75em; background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15); border-radius: 4px; color: #ccc; cursor: pointer;
        }
        .preset-btn:hover { background: rgba(255,255,255,0.15); color: #fff; }

        .status {
            display: inline-flex; align-items: center; gap: 5px;
            padding: 5px 10px; border-radius: 12px; font-size: 0.8em;
        }
        .status-ready { background: rgba(0,255,136,0.2); color: var(--success); }
        .status-busy { background: rgba(255,170,0,0.2); color: var(--warning); }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; animation: pulse 1.5s infinite; }
        .status-ready .status-dot { background: var(--success); }
        .status-busy .status-dot { background: var(--warning); }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        .item-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px; background: rgba(0,0,0,0.2); border-radius: 5px; margin-bottom: 6px;
        }
        .item-row:hover { background: rgba(0,0,0,0.3); }
        .item-info { flex: 1; }
        .item-url { font-size: 0.8em; color: var(--accent); word-break: break-all; }
        .item-meta { font-size: 0.7em; color: var(--text-secondary); margin-top: 2px; }

        .badge {
            display: inline-block; padding: 2px 6px; border-radius: 8px;
            font-size: 0.65em; margin-left: 4px;
        }
        .badge-new { background: var(--success); color: #000; }
        .badge-changed { background: var(--warning); color: #000; }
        .badge-running { background: var(--accent); color: #000; }

        .theme-toggle {
            width: 36px; height: 36px; border-radius: 50%; border: none;
            cursor: pointer; font-size: 1.1em; background: var(--bg-card); color: var(--text-primary);
        }

        .search-box {
            display: flex; gap: 8px; margin-bottom: 10px;
        }
        .search-box input { flex: 1; margin: 0; }

        .highlight-new { background: rgba(0,255,136,0.2); padding: 2px 4px; border-radius: 3px; }
        .highlight-removed { background: rgba(255,68,68,0.2); text-decoration: line-through; }

        .api-endpoint {
            background: rgba(0,0,0,0.4); padding: 8px; border-radius: 5px;
            font-family: monospace; font-size: 0.8em; margin-bottom: 8px;
        }
        .api-method { color: var(--success); font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <h1>🌐 Web Scraper <span>Pro v3</span></h1>
                <p class="subtitle">LM Studio + Playwright + Advanced Features</p>
            </div>
            <div class="header-actions">
                <div id="status" class="status status-ready">
                    <span class="status-dot"></span>
                    <span id="status-text">Ready</span>
                </div>
                <button class="theme-toggle" onclick="toggleTheme()">🌙</button>
            </div>
        </header>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('scrape')">🔍 Scrape</div>
            <div class="tab" onclick="switchTab('batch')">📦 Batch</div>
            <div class="tab" onclick="switchTab('schedule')">⏰ Schedule</div>
            <div class="tab" onclick="switchTab('diff')">🔄 Diff</div>
            <div class="tab" onclick="switchTab('translate')">🌍 Translate</div>
            <div class="tab" onclick="switchTab('links')">🔗 Links</div>
            <div class="tab" onclick="switchTab('sitemap')">🗺️ Sitemap</div>
            <div class="tab" onclick="switchTab('screenshot')">📸 Screenshot</div>
            <div class="tab" onclick="switchTab('rss')">📡 RSS</div>
            <div class="tab" onclick="switchTab('webhook')">🔔 Webhooks</div>
            <div class="tab" onclick="switchTab('api')">🔌 API</div>
            <div class="tab" onclick="switchTab('history')">📜 History</div>
        </div>

        <!-- SCRAPE TAB -->
        <div id="tab-scrape" class="tab-content active">
            <div class="grid">
                <div class="card">
                    <h2>📤 Scrape</h2>
                    <div class="toggle-group">
                        <label>Browser:</label>
                        <div class="toggle" id="modeToggle" onclick="toggleMode()"></div>
                        <label>Multi-page:</label>
                        <div class="toggle" id="multiPageToggle" onclick="toggleMultiPage()"></div>
                    </div>
                    <label>URL:</label>
                    <input type="text" id="urlInput" placeholder="https://example.com" value="https://news.ycombinator.com">
                    <div class="preset-btns">
                        <button class="preset-btn" onclick="setUrl('https://news.ycombinator.com')">HN</button>
                        <button class="preset-btn" onclick="setUrl('https://lobste.rs')">Lobsters</button>
                        <button class="preset-btn" onclick="setUrl('https://lite.cnn.com')">CNN</button>
                        <button class="preset-btn" onclick="setUrl('https://text.npr.org')">NPR</button>
                        <button class="preset-btn" onclick="setUrl('https://techcrunch.com')">TC</button>
                        <button class="preset-btn" onclick="setUrl('https://www.bbc.com/news')">BBC</button>
                        <button class="preset-btn" onclick="setUrl('https://arstechnica.com')">Ars</button>
                        <button class="preset-btn" onclick="setUrl('https://slashdot.org')">Slashdot</button>
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-primary" onclick="scrapeHeadlines()">🗞️ Headlines</button>
                        <button class="btn btn-primary" onclick="scrapeAnalyze()">🔍 Analyze</button>
                        <button class="btn btn-secondary" onclick="scrapeReadable()">📖 Readable</button>
                        <button class="btn btn-secondary" onclick="categorizeContent()">🏷️ Categorize</button>
                    </div>
                    <label>Keywords filter:</label>
                    <input type="text" id="keywordFilter" placeholder="AI, tech, startup...">
                    <label>Custom selector:</label>
                    <input type="text" id="customSelector" placeholder=".title, h2.headline">
                    <div class="search-box">
                        <input type="text" id="searchInResults" placeholder="🔍 Search in results..." oninput="searchResults()">
                    </div>
                    <h3>Ask AI:</h3>
                    <textarea id="customInput" placeholder="Ask about the content..."></textarea>
                    <button class="btn btn-primary" onclick="askQuestion()">🚀 Ask</button>
                </div>
                <div class="card">
                    <h2>📋 Log</h2>
                    <div class="log-container" id="logContainer"></div>
                </div>
            </div>
            <div class="card" style="margin-top:15px;">
                <h2>📤 Result
                    <button class="btn btn-small btn-secondary" onclick="copyResult()">📋</button>
                    <button class="btn btn-small btn-secondary" onclick="exportJSON()">JSON</button>
                    <button class="btn btn-small btn-secondary" onclick="exportCSV()">CSV</button>
                </h2>
                <div class="result-container" id="resultContainer">Results appear here...</div>
            </div>
        </div>

        <!-- BATCH TAB -->
        <div id="tab-batch" class="tab-content">
            <div class="card">
                <h2>📦 Batch Processing</h2>
                <label>URLs (one per line):</label>
                <textarea id="batchUrls" style="min-height:120px;">https://news.ycombinator.com
https://lobste.rs
https://lite.cnn.com</textarea>
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="runBatch()">🚀 Scrape All</button>
                </div>
                <div class="result-container" id="batchResult">Results appear here...</div>
            </div>
        </div>

        <!-- SCHEDULE TAB -->
        <div id="tab-schedule" class="tab-content">
            <div class="card">
                <h2>⏰ Scheduled Scraping</h2>
                <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:10px;">Set up automatic recurring scrapes</p>
                <label>URL to monitor:</label>
                <input type="text" id="scheduleUrl" placeholder="https://example.com" value="https://news.ycombinator.com">
                <label>Interval:</label>
                <select id="scheduleInterval">
                    <option value="60">Every 1 minute</option>
                    <option value="300">Every 5 minutes</option>
                    <option value="900">Every 15 minutes</option>
                    <option value="1800">Every 30 minutes</option>
                    <option value="3600" selected>Every 1 hour</option>
                    <option value="86400">Every 24 hours</option>
                </select>
                <div class="btn-group">
                    <button class="btn btn-success" onclick="addSchedule()">➕ Add Schedule</button>
                </div>
                <h3>Active Schedules:</h3>
                <div id="schedulesList"></div>
            </div>
        </div>

        <!-- DIFF TAB -->
        <div id="tab-diff" class="tab-content">
            <div class="card">
                <h2>🔄 Content Diff - What's New?</h2>
                <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:10px;">Compare current content with previous scrape to see changes</p>
                <label>URL:</label>
                <input type="text" id="diffUrl" placeholder="https://example.com" value="https://news.ycombinator.com">
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="runDiff()">🔄 Check for Changes</button>
                    <button class="btn btn-secondary" onclick="clearCache()">🗑️ Clear Cache</button>
                </div>
                <div class="result-container" id="diffResult">Diff results appear here. New items will be highlighted.</div>
            </div>
        </div>

        <!-- TRANSLATE TAB -->
        <div id="tab-translate" class="tab-content">
            <div class="card">
                <h2>🌍 Translation</h2>
                <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:10px;">Translate scraped content to English using AI</p>
                <label>URL with foreign content:</label>
                <input type="text" id="translateUrl" placeholder="https://example.com" value="https://www.delfi.lt">
                <label>Target language:</label>
                <select id="targetLang">
                    <option value="English" selected>English</option>
                    <option value="Spanish">Spanish</option>
                    <option value="French">French</option>
                    <option value="German">German</option>
                    <option value="Chinese">Chinese</option>
                    <option value="Japanese">Japanese</option>
                </select>
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="translateContent()">🌍 Translate</button>
                </div>
                <div class="result-container" id="translateResult">Translated content appears here...</div>
            </div>
        </div>

        <!-- LINKS TAB -->
        <div id="tab-links" class="tab-content">
            <div class="card">
                <h2>🔗 Link Extraction</h2>
                <label>URL:</label>
                <input type="text" id="linksUrl" value="https://news.ycombinator.com">
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="extractLinks()">🔗 Extract</button>
                    <button class="btn btn-secondary" onclick="checkRobots()">🤖 Check robots.txt</button>
                </div>
                <div class="result-container" id="linksResult">Links appear here...</div>
            </div>
        </div>

        <!-- SITEMAP TAB -->
        <div id="tab-sitemap" class="tab-content">
            <div class="card">
                <h2>🗺️ Sitemap Parser</h2>
                <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:10px;">Extract all URLs from a website's sitemap.xml</p>
                <label>Sitemap URL or domain:</label>
                <input type="text" id="sitemapUrl" placeholder="https://example.com/sitemap.xml" value="https://news.ycombinator.com">
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="parseSitemap()">🗺️ Parse Sitemap</button>
                </div>
                <div class="result-container" id="sitemapResult" style="max-height:400px;">Sitemap URLs appear here...</div>
            </div>
        </div>

        <!-- SCREENSHOT TAB -->
        <div id="tab-screenshot" class="tab-content">
            <div class="card">
                <h2>📸 Screenshot</h2>
                <label>URL:</label>
                <input type="text" id="screenshotUrl" value="https://news.ycombinator.com">
                <div class="toggle-group">
                    <label>Full page:</label>
                    <div class="toggle active" id="fullPageToggle"></div>
                </div>
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="takeScreenshot()">📸 Capture</button>
                </div>
                <div id="screenshotResult" style="text-align:center;margin-top:10px;"></div>
            </div>
        </div>

        <!-- RSS TAB -->
        <div id="tab-rss" class="tab-content">
            <div class="card">
                <h2>📡 RSS Feed Generator</h2>
                <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:10px;">Generate an RSS feed from any webpage's headlines</p>
                <label>Source URL:</label>
                <input type="text" id="rssUrl" value="https://news.ycombinator.com">
                <label>Feed title:</label>
                <input type="text" id="rssFeedTitle" value="My Custom Feed">
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="generateRSS()">📡 Generate RSS</button>
                </div>
                <div class="result-container" id="rssResult" style="max-height:400px;">RSS XML appears here...</div>
            </div>
        </div>

        <!-- WEBHOOK TAB -->
        <div id="tab-webhook" class="tab-content">
            <div class="card">
                <h2>🔔 Webhook Notifications</h2>
                <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:10px;">Send scrape results to Slack, Discord, or custom webhooks</p>
                <label>Webhook URL:</label>
                <input type="text" id="webhookUrl" placeholder="https://hooks.slack.com/services/...">
                <label>Webhook type:</label>
                <select id="webhookType">
                    <option value="slack">Slack</option>
                    <option value="discord">Discord</option>
                    <option value="custom">Custom (JSON POST)</option>
                </select>
                <div class="btn-group">
                    <button class="btn btn-success" onclick="saveWebhook()">💾 Save Webhook</button>
                    <button class="btn btn-primary" onclick="testWebhook()">🧪 Test</button>
                </div>
                <h3>Saved Webhooks:</h3>
                <div id="webhooksList"></div>
            </div>
        </div>

        <!-- API TAB -->
        <div id="tab-api" class="tab-content">
            <div class="card">
                <h2>🔌 REST API</h2>
                <p style="color:var(--text-secondary);font-size:0.85em;margin-bottom:10px;">Programmatic access to scraper functionality</p>

                <h3>Available Endpoints:</h3>

                <div class="api-endpoint">
                    <span class="api-method">POST</span> /api/scrape
                    <pre style="color:var(--text-secondary);font-size:0.75em;margin-top:5px;">{"url": "https://...", "mode": "headlines|analyze|fulltext"}</pre>
                </div>

                <div class="api-endpoint">
                    <span class="api-method">POST</span> /api/screenshot
                    <pre style="color:var(--text-secondary);font-size:0.75em;margin-top:5px;">{"url": "https://...", "fullPage": true}</pre>
                </div>

                <div class="api-endpoint">
                    <span class="api-method">POST</span> /api/translate
                    <pre style="color:var(--text-secondary);font-size:0.75em;margin-top:5px;">{"url": "https://...", "targetLang": "English"}</pre>
                </div>

                <div class="api-endpoint">
                    <span class="api-method">GET</span> /api/sitemap?url=https://...
                </div>

                <div class="api-endpoint">
                    <span class="api-method">GET</span> /api/robots?url=https://...
                </div>

                <h3>Example:</h3>
                <div class="api-endpoint">
                    <pre style="color:var(--text-secondary);font-size:0.75em;">curl -X POST http://localhost:5001/api/scrape \\
  -H "Content-Type: application/json" \\
  -d '{"url": "https://news.ycombinator.com", "mode": "headlines"}'</pre>
                </div>
            </div>
        </div>

        <!-- HISTORY TAB -->
        <div id="tab-history" class="tab-content">
            <div class="card">
                <h2>📜 History</h2>
                <input type="text" id="historyFilter" placeholder="🔍 Filter..." oninput="filterHistory()">
                <div id="historyList"></div>
                <div class="btn-group" style="margin-top:10px;">
                    <button class="btn btn-secondary" onclick="clearHistory()">🗑️ Clear</button>
                    <button class="btn btn-secondary" onclick="exportHistory()">💾 Export</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isProcessing = false;
        let useBrowser = false;
        let multiPage = false;
        let isDarkTheme = true;
        let lastResult = null;
        let lastHeadlines = [];
        let history = [];
        let webhooks = [];
        let schedules = [];

        try {
            history = JSON.parse(localStorage.getItem('scrapeHistory') || '[]');
            webhooks = JSON.parse(localStorage.getItem('webhooks') || '[]');
            schedules = JSON.parse(localStorage.getItem('schedules') || '[]');
            isDarkTheme = localStorage.getItem('darkTheme') !== 'false';
        } catch(e) {}

        if (!isDarkTheme) document.body.classList.add('light-theme');
        renderHistory();
        renderWebhooks();
        renderSchedules();

        const eventSource = new EventSource('/stream-logs');
        eventSource.onmessage = e => {
            const log = JSON.parse(e.data);
            if (log.message !== 'heartbeat') addLogEntry(log);
        };

        function addLogEntry(log) {
            const c = document.getElementById('logContainer');
            const e = document.createElement('div');
            e.className = 'log-entry';
            e.innerHTML = `<span class="log-time">${log.time}</span><span class="log-${log.level}">${log.icon} ${log.message}</span>`;
            c.appendChild(e);
            c.scrollTop = c.scrollHeight;
            if (c.children.length > 300) c.removeChild(c.firstChild);
        }

        function switchTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab-' + name).classList.add('active');
        }

        function toggleMode() { useBrowser = !useBrowser; document.getElementById('modeToggle').classList.toggle('active', useBrowser); }
        function toggleMultiPage() { multiPage = !multiPage; document.getElementById('multiPageToggle').classList.toggle('active', multiPage); }
        function toggleTheme() { isDarkTheme = !isDarkTheme; document.body.classList.toggle('light-theme', !isDarkTheme); localStorage.setItem('darkTheme', isDarkTheme); }
        function setUrl(url) { document.getElementById('urlInput').value = url; }
        function setStatus(busy) {
            isProcessing = busy;
            document.getElementById('status').className = busy ? 'status status-busy' : 'status status-ready';
            document.getElementById('status-text').textContent = busy ? 'Processing...' : 'Ready';
        }

        async function api(endpoint, data) {
            setStatus(true);
            try {
                const r = await fetch(endpoint, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
                return await r.json();
            } finally { setStatus(false); }
        }

        async function scrapeHeadlines() {
            const url = document.getElementById('urlInput').value.trim();
            const kw = document.getElementById('keywordFilter').value;
            const sel = document.getElementById('customSelector').value;
            const r = await api('/execute', {command: 'headlines', url, useBrowser, multiPage, keywords: kw, customSelector: sel});
            lastResult = r.result; lastHeadlines = r.headlines || [];
            document.getElementById('resultContainer').textContent = lastResult;
            addHistory(url, 'headlines', lastResult);
        }

        async function scrapeAnalyze() {
            const url = document.getElementById('urlInput').value.trim();
            const r = await api('/execute', {command: 'analyze', url, useBrowser});
            lastResult = r.result;
            document.getElementById('resultContainer').textContent = lastResult;
            addHistory(url, 'analyze', lastResult);
        }

        async function scrapeReadable() {
            const url = document.getElementById('urlInput').value.trim();
            const r = await api('/execute', {command: 'readable', url, useBrowser});
            lastResult = r.result;
            document.getElementById('resultContainer').textContent = lastResult;
        }

        async function categorizeContent() {
            const url = document.getElementById('urlInput').value.trim();
            const r = await api('/execute', {command: 'categorize', url, useBrowser});
            lastResult = r.result;
            document.getElementById('resultContainer').textContent = lastResult;
        }

        async function askQuestion() {
            const url = document.getElementById('urlInput').value.trim();
            const q = document.getElementById('customInput').value.trim();
            const r = await api('/execute', {command: 'ask', url, question: q, useBrowser});
            document.getElementById('resultContainer').textContent = r.result;
        }

        function searchResults() {
            const term = document.getElementById('searchInResults').value.toLowerCase();
            const container = document.getElementById('resultContainer');
            if (!term) { container.innerHTML = lastResult || ''; return; }
            const lines = (lastResult || '').split('\\n');
            const filtered = lines.filter(l => l.toLowerCase().includes(term));
            container.textContent = filtered.length ? filtered.join('\\n') : 'No matches found';
        }

        async function runBatch() {
            const urls = document.getElementById('batchUrls').value.trim().split('\\n').filter(u => u.trim());
            let output = 'BATCH RESULTS\\n' + '='.repeat(40) + '\\n';
            for (let i = 0; i < urls.length; i++) {
                document.getElementById('batchResult').textContent = `Processing ${i+1}/${urls.length}...`;
                const r = await api('/execute', {command: 'headlines', url: urls[i].trim()});
                output += `\\n${r.headlines?.length || 0} headlines from ${urls[i]}\\n`;
            }
            document.getElementById('batchResult').textContent = output;
        }

        async function runDiff() {
            const url = document.getElementById('diffUrl').value.trim();
            const r = await api('/diff', {url});
            document.getElementById('diffResult').innerHTML = r.result;
        }

        function clearCache() { api('/clear-cache', {}); alert('Cache cleared!'); }

        async function translateContent() {
            const url = document.getElementById('translateUrl').value.trim();
            const lang = document.getElementById('targetLang').value;
            const r = await api('/translate', {url, targetLang: lang});
            document.getElementById('translateResult').textContent = r.result;
        }

        async function extractLinks() {
            const url = document.getElementById('linksUrl').value.trim();
            const r = await api('/extract-links', {url});
            const links = r.links || [];
            let out = `Found ${links.length} links\\n\\nINTERNAL:\\n`;
            links.filter(l => l.type === 'internal').slice(0,20).forEach(l => out += `  ${l.text.slice(0,40)} -> ${l.href}\\n`);
            out += `\\nEXTERNAL:\\n`;
            links.filter(l => l.type === 'external').slice(0,20).forEach(l => out += `  ${l.text.slice(0,40)} -> ${l.href}\\n`);
            document.getElementById('linksResult').textContent = out;
        }

        async function checkRobots() {
            const url = document.getElementById('linksUrl').value.trim();
            const r = await api('/robots', {url});
            document.getElementById('linksResult').textContent = r.result;
        }

        async function parseSitemap() {
            const url = document.getElementById('sitemapUrl').value.trim();
            const r = await api('/sitemap', {url});
            document.getElementById('sitemapResult').textContent = r.result;
        }

        async function takeScreenshot() {
            const url = document.getElementById('screenshotUrl').value.trim();
            const full = document.getElementById('fullPageToggle').classList.contains('active');
            const r = await api('/screenshot', {url, fullPage: full});
            if (r.screenshot) {
                document.getElementById('screenshotResult').innerHTML = `
                    <img src="data:image/png;base64,${r.screenshot}" style="max-width:100%;border-radius:8px;">
                    <br><a href="data:image/png;base64,${r.screenshot}" download="screenshot.png" class="btn btn-primary" style="margin-top:10px;">💾 Download</a>`;
            }
        }

        async function generateRSS() {
            const url = document.getElementById('rssUrl').value.trim();
            const title = document.getElementById('rssFeedTitle').value.trim();
            const r = await api('/generate-rss', {url, title});
            document.getElementById('rssResult').textContent = r.rss;
        }

        function addSchedule() {
            const url = document.getElementById('scheduleUrl').value.trim();
            const interval = parseInt(document.getElementById('scheduleInterval').value);
            schedules.push({id: Date.now(), url, interval, active: true});
            localStorage.setItem('schedules', JSON.stringify(schedules));
            renderSchedules();
            api('/add-schedule', {url, interval});
        }

        function removeSchedule(id) {
            schedules = schedules.filter(s => s.id !== id);
            localStorage.setItem('schedules', JSON.stringify(schedules));
            renderSchedules();
            api('/remove-schedule', {id});
        }

        function renderSchedules() {
            const c = document.getElementById('schedulesList');
            if (!schedules.length) { c.innerHTML = '<p style="color:var(--text-secondary)">No schedules</p>'; return; }
            c.innerHTML = schedules.map(s => `
                <div class="item-row">
                    <div class="item-info">
                        <div class="item-url">${s.url}</div>
                        <div class="item-meta">Every ${s.interval}s</div>
                    </div>
                    <button class="btn btn-small btn-secondary" onclick="removeSchedule(${s.id})">🗑️</button>
                </div>`).join('');
        }

        function saveWebhook() {
            const url = document.getElementById('webhookUrl').value.trim();
            const type = document.getElementById('webhookType').value;
            if (!url) return;
            webhooks.push({id: Date.now(), url, type});
            localStorage.setItem('webhooks', JSON.stringify(webhooks));
            renderWebhooks();
        }

        function removeWebhook(id) {
            webhooks = webhooks.filter(w => w.id !== id);
            localStorage.setItem('webhooks', JSON.stringify(webhooks));
            renderWebhooks();
        }

        function testWebhook() {
            const url = document.getElementById('webhookUrl').value.trim();
            const type = document.getElementById('webhookType').value;
            api('/test-webhook', {url, type});
            alert('Test sent!');
        }

        function renderWebhooks() {
            const c = document.getElementById('webhooksList');
            if (!webhooks.length) { c.innerHTML = '<p style="color:var(--text-secondary)">No webhooks</p>'; return; }
            c.innerHTML = webhooks.map(w => `
                <div class="item-row">
                    <div class="item-info">
                        <div class="item-url">${w.type}: ${w.url.slice(0,40)}...</div>
                    </div>
                    <button class="btn btn-small btn-secondary" onclick="removeWebhook(${w.id})">🗑️</button>
                </div>`).join('');
        }

        function addHistory(url, cmd, result) {
            history.unshift({id: Date.now(), url, cmd, time: new Date().toISOString(), preview: (result||'').slice(0,80)});
            if (history.length > 50) history = history.slice(0,50);
            localStorage.setItem('scrapeHistory', JSON.stringify(history));
            renderHistory();
        }

        function renderHistory() {
            const c = document.getElementById('historyList');
            const filter = (document.getElementById('historyFilter')?.value || '').toLowerCase();
            const filtered = history.filter(h => h.url.toLowerCase().includes(filter));
            if (!filtered.length) { c.innerHTML = '<p style="color:var(--text-secondary)">No history</p>'; return; }
            c.innerHTML = filtered.slice(0,20).map(h => `
                <div class="item-row">
                    <div class="item-info">
                        <div class="item-url">${h.url}</div>
                        <div class="item-meta">${h.cmd} • ${new Date(h.time).toLocaleString()}</div>
                    </div>
                    <button class="btn btn-small btn-secondary" onclick="setUrl('${h.url}');switchTab('scrape');">↗️</button>
                </div>`).join('');
        }

        function filterHistory() { renderHistory(); }
        function clearHistory() { history = []; localStorage.setItem('scrapeHistory', '[]'); renderHistory(); }
        function exportHistory() { download(JSON.stringify(history,null,2), 'history.json', 'application/json'); }

        function copyResult() { navigator.clipboard.writeText(lastResult || ''); }
        function exportJSON() { download(JSON.stringify({result: lastResult, headlines: lastHeadlines},null,2), 'result.json', 'application/json'); }
        function exportCSV() {
            let csv = 'Title,URL\\n';
            lastHeadlines.forEach(h => csv += `"${(h.title||'').replace(/"/g,'""')}","${h.url||''}"\\n`);
            download(csv, 'headlines.csv', 'text/csv');
        }
        function download(content, name, type) {
            const a = document.createElement('a');
            a.href = URL.createObjectURL(new Blob([content], {type}));
            a.download = name; a.click();
        }
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
    cmd = data.get('command', '')
    url = data.get('url', '')
    question = data.get('question', '')
    use_browser = data.get('useBrowser', False)
    multi_page = data.get('multiPage', False)
    keywords = data.get('keywords', '')
    custom_sel = data.get('customSelector', '')

    live_log(f"Execute: {cmd} | {url[:40]}...", "action")

    try:
        result, headlines = agent_instance.execute_command(cmd, url, question, use_browser, multi_page, keywords, custom_sel)
        return jsonify({'result': result, 'headlines': headlines})
    except Exception as e:
        live_log(f"Error: {e}", "error")
        return jsonify({'error': str(e), 'headlines': []})

@web_app.route('/diff', methods=['POST'])
def diff_content():
    url = request.json.get('url', '')
    live_log(f"Running diff for: {url}", "diff")
    try:
        result = agent_instance.diff_content(url)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

@web_app.route('/clear-cache', methods=['POST'])
def clear_cache():
    content_cache.clear()
    live_log("Cache cleared", "info")
    return jsonify({'success': True})

@web_app.route('/translate', methods=['POST'])
def translate():
    url = request.json.get('url', '')
    lang = request.json.get('targetLang', 'English')
    live_log(f"Translating {url} to {lang}", "llm")
    try:
        result = agent_instance.translate_content(url, lang)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

@web_app.route('/extract-links', methods=['POST'])
def extract_links():
    url = request.json.get('url', '')
    try:
        links = agent_instance.extract_all_links(url)
        return jsonify({'links': links})
    except Exception as e:
        return jsonify({'error': str(e), 'links': []})

@web_app.route('/robots', methods=['POST'])
def check_robots():
    url = request.json.get('url', '')
    try:
        result = agent_instance.check_robots_txt(url)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

@web_app.route('/sitemap', methods=['POST'])
def parse_sitemap():
    url = request.json.get('url', '')
    try:
        result = agent_instance.parse_sitemap(url)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

@web_app.route('/screenshot', methods=['POST'])
def screenshot():
    url = request.json.get('url', '')
    full_page = request.json.get('fullPage', True)
    try:
        data = agent_instance.take_screenshot(url, full_page)
        return jsonify({'screenshot': data})
    except Exception as e:
        return jsonify({'error': str(e)})

@web_app.route('/generate-rss', methods=['POST'])
def generate_rss():
    url = request.json.get('url', '')
    title = request.json.get('title', 'Custom Feed')
    try:
        rss = agent_instance.generate_rss(url, title)
        return jsonify({'rss': rss})
    except Exception as e:
        return jsonify({'error': str(e)})

@web_app.route('/add-schedule', methods=['POST'])
def add_schedule():
    url = request.json.get('url', '')
    interval = request.json.get('interval', 3600)
    job_id = f"schedule_{int(time.time())}"
    scheduled_jobs[job_id] = {'url': url, 'interval': interval, 'last_run': 0}
    live_log(f"Schedule added: {url} every {interval}s", "schedule")
    return jsonify({'success': True, 'id': job_id})

@web_app.route('/remove-schedule', methods=['POST'])
def remove_schedule():
    job_id = request.json.get('id', '')
    if str(job_id) in scheduled_jobs:
        del scheduled_jobs[str(job_id)]
    live_log(f"Schedule removed: {job_id}", "schedule")
    return jsonify({'success': True})

@web_app.route('/test-webhook', methods=['POST'])
def test_webhook():
    url = request.json.get('url', '')
    wh_type = request.json.get('type', 'custom')
    try:
        agent_instance.send_webhook(url, wh_type, "Test message from Web Scraper Pro!")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

# REST API Endpoints
@web_app.route('/api/scrape', methods=['POST'])
def api_scrape():
    live_log("API: /api/scrape", "api")
    data = request.json
    url = data.get('url', '')
    mode = data.get('mode', 'headlines')
    try:
        result, headlines = agent_instance.execute_command(mode, url, '', False, False, '', '')
        return jsonify({'success': True, 'result': result, 'headlines': headlines})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@web_app.route('/api/screenshot', methods=['POST'])
def api_screenshot():
    live_log("API: /api/screenshot", "api")
    data = request.json
    url = data.get('url', '')
    full = data.get('fullPage', True)
    try:
        img = agent_instance.take_screenshot(url, full)
        return jsonify({'success': True, 'screenshot': img})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@web_app.route('/api/translate', methods=['POST'])
def api_translate():
    live_log("API: /api/translate", "api")
    data = request.json
    url = data.get('url', '')
    lang = data.get('targetLang', 'English')
    try:
        result = agent_instance.translate_content(url, lang)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@web_app.route('/api/sitemap', methods=['GET'])
def api_sitemap():
    live_log("API: /api/sitemap", "api")
    url = request.args.get('url', '')
    try:
        result = agent_instance.parse_sitemap(url)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@web_app.route('/api/robots', methods=['GET'])
def api_robots():
    live_log("API: /api/robots", "api")
    url = request.args.get('url', '')
    try:
        result = agent_instance.check_robots_txt(url)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@agent(name="WebScraperProV3", description="Advanced web scraper with scheduling, diff, translation, RSS, webhooks, and API", version="3.0.0")
class UniversalScraperAgent(A2AServer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        live_log("Initializing Web Scraper Pro v3.0...", "action")
        self.llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        self.last_content = ""
        self.last_url = ""
        self.last_headlines = []

        try:
            models = self.llm_client.models.list()
            live_log(f"LLM Connected: {[m.id for m in models.data]}", "success")
        except Exception as e:
            live_log(f"LLM connection failed: {e}", "error")

        live_log("Agent ready!", "success")

    def execute_command(self, cmd, url, question='', use_browser=False, multi_page=False, keywords='', custom_sel=''):
        if cmd == 'headlines':
            return self.get_headlines(url, use_browser, multi_page, keywords, custom_sel)
        elif cmd == 'analyze':
            return self.analyze_page(url, use_browser), []
        elif cmd == 'readable':
            return self.get_readable(url, use_browser), []
        elif cmd == 'categorize':
            return self.categorize_content(url, use_browser), []
        elif cmd == 'ask':
            return self.ask_about(url, question, use_browser), []
        return "Unknown command", []

    def _fetch(self, url, use_browser=False):
        if use_browser:
            return self._fetch_browser(url)
        return self._fetch_http(url)

    def _fetch_browser(self, url):
        live_log(f"Browser fetch: {url[:50]}...", "browser")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1920, "height": 1080})
                page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(2000)
                for _ in range(3):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1000)
                html = page.content()
                browser.close()
                soup = BeautifulSoup(html, "html.parser")
                for t in soup(["script", "style", "noscript"]): t.decompose()
                return soup, html
        except Exception as e:
            live_log(f"Browser error: {e}", "error")
            return None, str(e)

    def _fetch_http(self, url):
        live_log(f"HTTP fetch: {url[:50]}...", "web")
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for t in soup(["script", "style", "noscript"]): t.decompose()
            live_log(f"Got {len(r.text):,} bytes", "success")
            return soup, r.text
        except Exception as e:
            live_log(f"HTTP error: {e}", "error")
            return None, str(e)

    def _extract_headlines(self, soup, url, custom_sel=''):
        headlines = []
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        selectors = ([s.strip() for s in custom_sel.split(',') if s.strip()] if custom_sel else []) + [
            '.titleline a', '.storylink', 'article h2', 'article h3', '.headline', '.title',
            '[class*="title"]', 'h1 a', 'h2 a', 'h3 a', '.card-title'
        ]
        seen = set()
        for sel in selectors:
            if len(headlines) >= MAX_HEADLINES: break
            try:
                for el in soup.select(sel)[:50]:
                    title = re.sub(r'\s+', ' ', el.get_text(strip=True))
                    if title and 10 < len(title) < 300 and title not in seen:
                        seen.add(title)
                        link = el.get('href') if el.name == 'a' else (el.find('a') or {}).get('href')
                        if link and not link.startswith('http'): link = urljoin(base, link)
                        headlines.append({"title": title, "url": link})
            except: pass
        live_log(f"Extracted {len(headlines)} headlines", "success")
        return headlines

    def _extract_content(self, soup):
        for sel in ['article', 'main', '.content', '.post-content', '#content']:
            els = soup.select(sel)
            if els:
                text = '\n'.join(e.get_text(separator='\n', strip=True) for e in els)
                if len(text) > 300: return text[:MAX_CONTENT_CHARS]
        paras = '\n'.join(p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text()) > 20)
        return paras[:MAX_CONTENT_CHARS] if paras else soup.get_text(separator='\n', strip=True)[:MAX_CONTENT_CHARS]

    def _ask_llm(self, prompt):
        live_log(f"LLM: {len(prompt):,} chars...", "llm")
        try:
            r = self.llm_client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=[{"role": "system", "content": "You are a helpful assistant. Be concise."},
                          {"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=1500
            )
            live_log("LLM done", "success")
            return r.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {e}"

    def get_headlines(self, url, use_browser=False, multi_page=False, keywords='', custom_sel=''):
        soup, _ = self._fetch(url, use_browser)
        if not soup: return "Failed to fetch", []
        headlines = self._extract_headlines(soup, url, custom_sel)
        if keywords:
            kw_list = [k.strip().lower() for k in keywords.split(',')]
            headlines = [h for h in headlines if any(k in h['title'].lower() for k in kw_list)]
        if not headlines: return "No headlines found", []
        result = f"📰 {len(headlines)} headlines from {urlparse(url).netloc}:\n\n"
        for i, h in enumerate(headlines, 1):
            result += f"{i}. {h['title']}\n"
            if h.get('url'): result += f"   🔗 {h['url']}\n"
        self.last_headlines = headlines
        return result, headlines

    def analyze_page(self, url, use_browser=False):
        soup, _ = self._fetch(url, use_browser)
        if not soup: return "Failed to fetch"
        headlines = self._extract_headlines(soup, url)
        content = self._extract_content(soup)
        prompt = f"Analyze {url}:\n\nHEADLINES:\n" + '\n'.join(f"- {h['title']}" for h in headlines[:20])
        prompt += f"\n\nCONTENT:\n{content[:5000]}\n\nProvide: 1) Type 2) Topics 3) Key points 4) Summary"
        return self._ask_llm(prompt)

    def get_readable(self, url, use_browser=False):
        soup, _ = self._fetch(url, use_browser)
        if not soup: return "Failed to fetch"
        # Remove ads, nav, sidebars
        for sel in ['nav', 'header', 'footer', 'aside', '.ad', '.sidebar', '.menu', '.nav']:
            for el in soup.select(sel): el.decompose()
        content = self._extract_content(soup)
        return f"📖 Readable content from {urlparse(url).netloc}:\n\n{content}"

    def categorize_content(self, url, use_browser=False):
        soup, _ = self._fetch(url, use_browser)
        if not soup: return "Failed to fetch"
        headlines = self._extract_headlines(soup, url)
        content = self._extract_content(soup)
        prompt = f"Categorize each headline by topic (Tech, Business, Science, Politics, Entertainment, Sports, Other):\n\n"
        prompt += '\n'.join(f"- {h['title']}" for h in headlines[:30])
        prompt += "\n\nFormat: [CATEGORY] Title"
        return self._ask_llm(prompt)

    def ask_about(self, url, question, use_browser=False):
        if url != self.last_url:
            soup, _ = self._fetch(url, use_browser)
            if soup: self.last_content = self._extract_content(soup)
            self.last_url = url
        return self._ask_llm(f"Content from {url}:\n{self.last_content[:8000]}\n\nQuestion: {question}")

    def diff_content(self, url):
        soup, _ = self._fetch(url, False)
        if not soup: return "Failed to fetch"
        current = self._extract_headlines(soup, url)
        current_titles = {h['title'] for h in current}

        cache_key = urlparse(url).netloc
        previous_titles = content_cache.get(cache_key, set())
        content_cache[cache_key] = current_titles

        new_items = current_titles - previous_titles
        removed_items = previous_titles - current_titles

        result = f"🔄 DIFF for {url}\n{'='*40}\n\n"
        if not previous_titles:
            result += "First scan - no previous data. Run again later to see changes.\n\n"
        result += f"✅ NEW ({len(new_items)}):\n"
        for t in list(new_items)[:20]:
            result += f'<span class="highlight-new">+ {t}</span>\n'
        result += f"\n❌ REMOVED ({len(removed_items)}):\n"
        for t in list(removed_items)[:20]:
            result += f'<span class="highlight-removed">- {t}</span>\n'
        result += f"\n📊 Total: {len(current)} items"
        return result

    def translate_content(self, url, target_lang):
        soup, _ = self._fetch(url, False)
        if not soup: return "Failed to fetch"
        headlines = self._extract_headlines(soup, url)
        content = self._extract_content(soup)
        text = '\n'.join(h['title'] for h in headlines[:15]) + '\n\n' + content[:3000]
        prompt = f"Translate the following to {target_lang}. Keep formatting:\n\n{text}"
        return self._ask_llm(prompt)

    def extract_all_links(self, url):
        soup, _ = self._fetch_http(url)
        if not soup: return []
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        links, seen = [], set()
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            if not href or href.startswith('#') or href.startswith('javascript:'): continue
            if not href.startswith('http'): href = urljoin(base, href)
            if href in seen: continue
            seen.add(href)
            link_type = 'internal' if urlparse(href).netloc == urlparse(url).netloc else 'external'
            links.append({'href': href, 'text': a.get_text(strip=True)[:60], 'type': link_type})
        return links

    def check_robots_txt(self, url):
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        robots_url = f"{base}/robots.txt"
        live_log(f"Checking {robots_url}", "web")
        try:
            r = requests.get(robots_url, timeout=10)
            if r.status_code == 200:
                return f"🤖 robots.txt for {base}:\n\n{r.text[:3000]}"
            return f"No robots.txt found (status {r.status_code})"
        except Exception as e:
            return f"Error: {e}"

    def parse_sitemap(self, url):
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        sitemap_urls = [f"{base}/sitemap.xml", f"{base}/sitemap_index.xml", f"{base}/sitemap/sitemap.xml"]
        if url.endswith('.xml'): sitemap_urls.insert(0, url)

        for sitemap_url in sitemap_urls:
            try:
                live_log(f"Trying {sitemap_url}", "web")
                r = requests.get(sitemap_url, timeout=10)
                if r.status_code == 200 and '<url' in r.text.lower():
                    root = ET.fromstring(r.content)
                    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                    urls = [loc.text for loc in root.findall('.//sm:loc', ns)][:100]
                    if not urls:
                        urls = [loc.text for loc in root.findall('.//loc')][:100]
                    result = f"🗺️ Sitemap: {sitemap_url}\nFound {len(urls)} URLs:\n\n"
                    for u in urls: result += f"  {u}\n"
                    return result
            except: continue
        return "No sitemap found. Try providing direct sitemap URL."

    def take_screenshot(self, url, full_page=True):
        live_log(f"Screenshot: {url}", "browser")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1920, "height": 1080})
                page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(2000)
                img = page.screenshot(full_page=full_page)
                browser.close()
                live_log("Screenshot captured", "success")
                return base64.b64encode(img).decode('utf-8')
        except Exception as e:
            live_log(f"Screenshot error: {e}", "error")
            return None

    def generate_rss(self, url, title):
        soup, _ = self._fetch(url, False)
        if not soup: return "Failed to fetch"
        headlines = self._extract_headlines(soup, url)
        rss = f'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>{title}</title>
<link>{url}</link>
<description>Generated by Web Scraper Pro</description>
<lastBuildDate>{datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")}</lastBuildDate>
'''
        for h in headlines[:50]:
            rss += f'''<item>
<title><![CDATA[{h['title']}]]></title>
<link>{h.get('url', url)}</link>
</item>
'''
        rss += '</channel>\n</rss>'
        return rss

    def send_webhook(self, webhook_url, wh_type, message):
        live_log(f"Sending webhook to {webhook_url[:30]}...", "web")
        try:
            if wh_type == 'slack':
                payload = {"text": message}
            elif wh_type == 'discord':
                payload = {"content": message}
            else:
                payload = {"message": message, "timestamp": datetime.utcnow().isoformat()}
            requests.post(webhook_url, json=payload, timeout=10)
            live_log("Webhook sent", "success")
        except Exception as e:
            live_log(f"Webhook error: {e}", "error")

    def handle_task(self, task):
        live_log("A2A Task received", "action")
        content = task.message.content
        text = content.get("text", "") if isinstance(content, dict) else str(content)
        urls = re.findall(r'https?://\S+', text)
        if urls:
            result, _ = self.get_headlines(urls[0])
        else:
            result = "Please provide a URL"
        task.artifacts = [{"parts": [{"type": "text", "text": result}]}]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task


def run_scheduler():
    """Background thread for scheduled scraping"""
    while True:
        time.sleep(30)
        now = time.time()
        for job_id, job in list(scheduled_jobs.items()):
            if now - job['last_run'] >= job['interval']:
                live_log(f"Running scheduled job: {job['url']}", "schedule")
                try:
                    agent_instance.get_headlines(job['url'])
                    job['last_run'] = now
                except Exception as e:
                    live_log(f"Schedule error: {e}", "error")


agent_instance = None

def run_web_server():
    web_app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)

def run_a2a_server():
    run_server(agent_instance, port=8000)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🌐 WEB SCRAPER PRO v3.0")
    print("  Advanced Features: Schedule, Diff, Translate, RSS, API")
    print("=" * 60 + "\n")

    agent_instance = UniversalScraperAgent(url="http://localhost:8000")

    # Start scheduler thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    # Start web server
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()

    print("\n🟢 READY! Open http://localhost:5001\n")

    run_a2a_server()
