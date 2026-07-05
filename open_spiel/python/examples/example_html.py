# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tiny HTML/JS UI for OpenSpiel games (default: python_qwixx).

Runs a local HTTP server and exposes endpoints to inspect and step the game.
"""

import json
import random
import threading
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Prefer the local repository checkout over any installed package so newly
# added Python games (e.g., python_qwixx) are visible.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from absl import app
from absl import flags

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("game_string", "python_qwixx", "Game string to load.")
flags.DEFINE_string("host", "127.0.0.1", "Bind host.")
flags.DEFINE_integer("port", 8765, "Bind port.")

_HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenSpiel HTML Example</title>
  <style>
    :root {
      --bg: #101318;
      --card: #171b22;
      --line: #2a3240;
      --text: #e6edf3;
      --muted: #9aa7b5;
      --acc: #58a6ff;
      --ok: #3fb950;
      --warn: #f2cc60;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(1200px 600px at 20% -10%, #1f2937 0%, var(--bg) 60%);
      color: var(--text);
      font: 15px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      padding: 20px;
    }
    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
    }
    h1 { font-size: 18px; margin: 0 0 12px; }
    h2 { font-size: 14px; color: var(--muted); margin: 0 0 10px; }
    .row { display: flex; gap: 8px; margin: 8px 0; }
    button {
      border: 1px solid #3a4659;
      background: #1f2937;
      color: var(--text);
      padding: 7px 10px;
      border-radius: 8px;
      cursor: pointer;
    }
    button:hover { border-color: var(--acc); }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      margin-right: 8px;
    }
    .state {
      white-space: pre;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
      max-height: 68vh;
      background: #0f141b;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .action-btn {
      text-align: left;
      background: #122033;
      border-color: #28476a;
    }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>OpenSpiel HTML Viewer</h1>
      <h2 id="gameLabel"></h2>
      <div class="row">
        <button id="newBtn">New Game</button>
        <button id="randomBtn">Random Step</button>
        <button id="refreshBtn">Refresh</button>
      </div>
      <div id="meta"></div>
      <div style="margin-top: 10px;" class="warn">
        Tip: For chance nodes, use Random Step. For decision nodes, click an action.
      </div>
    </div>

    <div class="card">
      <h2>State</h2>
      <div id="stateText" class="state"></div>
      <h2 style="margin-top: 14px;">Legal Actions</h2>
      <div id="actions" class="actions"></div>
      <div id="returns" style="margin-top: 10px;"></div>
    </div>
  </div>

  <script>
    async function fetchJSON(url, options = {}) {
      const res = await fetch(url, options);
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`${res.status}: ${t}`);
      }
      return res.json();
    }

    async function refresh() {
      const s = await fetchJSON('/state');
      document.getElementById('gameLabel').textContent = `Game: ${s.game}`;
      document.getElementById('stateText').textContent = s.state_text;

      const mode = s.is_terminal ? 'TERMINAL' : (s.is_chance_node ? 'CHANCE' : `PLAYER ${s.current_player}`);
      document.getElementById('meta').innerHTML =
        `<span class="pill">mode: ${mode}</span>` +
        `<span class="pill">step: ${s.step_count}</span>`;

      const actionsDiv = document.getElementById('actions');
      actionsDiv.innerHTML = '';
      for (const a of s.legal_actions) {
        const b = document.createElement('button');
        b.className = 'action-btn';
        b.textContent = `${a.id}: ${a.label}`;
        b.onclick = async () => {
          await fetchJSON('/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: a.id }),
          });
          await refresh();
        };
        actionsDiv.appendChild(b);
      }

      const returnsDiv = document.getElementById('returns');
      returnsDiv.innerHTML = s.is_terminal
        ? `<span class="ok">Returns: [${s.returns.join(', ')}]</span>`
        : '';
    }

    document.getElementById('newBtn').onclick = async () => {
      await fetchJSON('/new', { method: 'POST' });
      await refresh();
    };

    document.getElementById('randomBtn').onclick = async () => {
      await fetchJSON('/random', { method: 'POST' });
      await refresh();
    };

    document.getElementById('refreshBtn').onclick = refresh;
    refresh();
  </script>
</body>
</html>
"""


class _GameSession:
  """Mutable in-memory game session for a single local user."""

  def __init__(self, game_string):
    self._lock = threading.Lock()
    self.game_string = game_string
    self.game = pyspiel.load_game(game_string)
    self.state = self.game.new_initial_state()
    self.step_count = 0

  def reset(self):
    with self._lock:
      self.state = self.game.new_initial_state()
      self.step_count = 0

  def _random_action(self):
    if self.state.is_chance_node():
      outcomes = self.state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      return random.choices(action_list, weights=prob_list, k=1)[0]
    legal = self.state.legal_actions(self.state.current_player())
    if not legal:
      return None
    return random.choice(legal)

  def apply_action(self, action):
    with self._lock:
      if self.state.is_terminal():
        return
      self.state.apply_action(action)
      self.step_count += 1

  def apply_random(self):
    with self._lock:
      if self.state.is_terminal():
        return
      action = self._random_action()
      if action is not None:
        self.state.apply_action(action)
        self.step_count += 1

  def as_dict(self):
    with self._lock:
      if self.state.is_terminal():
        legal = []
      elif self.state.is_chance_node():
        legal = [
            {
                "id": int(action),
                "label": f"{self.state.action_to_string(pyspiel.PlayerId.CHANCE, action)} (p={prob:.3f})",
            }
            for action, prob in self.state.chance_outcomes()
        ]
      else:
        cp = self.state.current_player()
        legal = [
            {
                "id": int(action),
                "label": self.state.action_to_string(cp, action),
            }
            for action in self.state.legal_actions(cp)
        ]

      return {
          "game": self.game_string,
          "state_text": str(self.state),
          "is_terminal": self.state.is_terminal(),
          "is_chance_node": self.state.is_chance_node() if not self.state.is_terminal() else False,
          "current_player": int(self.state.current_player()),
          "step_count": self.step_count,
          "legal_actions": legal,
          "returns": list(self.state.returns()),
      }


def _make_handler(session):
  class Handler(BaseHTTPRequestHandler):
    """HTTP handler serving a tiny single-page viewer and JSON endpoints."""

    def _write_json(self, payload, status=HTTPStatus.OK):
      data = json.dumps(payload).encode("utf-8")
      self.send_response(status)
      self.send_header("Content-Type", "application/json; charset=utf-8")
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)

    def _write_text(self, payload, content_type="text/plain; charset=utf-8", status=HTTPStatus.OK):
      data = payload.encode("utf-8")
      self.send_response(status)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)

    def do_GET(self):
      parsed = urlparse(self.path)
      if parsed.path == "/":
        self._write_text(_HTML_PAGE, "text/html; charset=utf-8")
      elif parsed.path == "/state":
        self._write_json(session.as_dict())
      else:
        self._write_text("Not found", status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
      parsed = urlparse(self.path)

      if parsed.path == "/new":
        session.reset()
        self._write_json({"ok": True})
        return

      if parsed.path == "/random":
        session.apply_random()
        self._write_json({"ok": True})
        return

      if parsed.path == "/action":
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
          payload = json.loads(raw.decode("utf-8"))
          action = int(payload["action"])
        except Exception:
          self._write_json({"error": "Invalid JSON/action"}, status=HTTPStatus.BAD_REQUEST)
          return
        try:
          session.apply_action(action)
        except Exception as exc:
          self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
          return
        self._write_json({"ok": True})
        return

      if parsed.path == "/set_game":
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        params = parse_qs(raw.decode("utf-8"))
        game_string = params.get("game", [None])[0]
        if not game_string:
          self._write_json({"error": "Missing game parameter"}, status=HTTPStatus.BAD_REQUEST)
          return
        try:
          session.game_string = game_string
          session.game = pyspiel.load_game(game_string)
          session.reset()
        except Exception as exc:
          self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
          return
        self._write_json({"ok": True})
        return

      self._write_text("Not found", status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt, *args):
      # Keep terminal output concise.
      del fmt, args

  return Handler


def main(argv):
  del argv

  session = _GameSession(FLAGS.game_string)
  handler_cls = _make_handler(session)
  server = ThreadingHTTPServer((FLAGS.host, FLAGS.port), handler_cls)

  print(f"Serving OpenSpiel HTML viewer at http://{FLAGS.host}:{FLAGS.port}")
  print(f"Game: {FLAGS.game_string}")
  print("Press Ctrl+C to stop.")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()


if __name__ == "__main__":
  app.run(main)