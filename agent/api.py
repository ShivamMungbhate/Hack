
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from .adapter import AgentAdapter

adapters = {}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_json(204, {})

    def do_POST(self):
        if self.path not in ("/process", "/reset", "/undo"):
            self.send_json(404, {"error": "Unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("Invalid request body size")

            body = json.loads(self.rfile.read(length))
            session = body.get("session", "default")

            if not isinstance(session, str) or not session:
                raise ValueError("session must be a non-empty string")
            if len(session) > 128:
                raise ValueError("session is too long")

            adapter = adapters.setdefault(session, AgentAdapter())

            if self.path == "/reset":
                adapter.clear()
                adapter.reset_gesture_lock()
                self.send_json(200, {"status": "reset"})
                return

            if self.path == "/undo":
                self.send_json(200, {"result": adapter.undo()})
                return

            result = adapter.process_event(body.get("event", {}))
            self.send_json(200, result)

        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            self.send_json(500, {"error": "Agent processing failed"})

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    print("SignBridge Agent API: http://127.0.0.1:8001")
    HTTPServer(("127.0.0.1", 8001), Handler).serve_forever()
