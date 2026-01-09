#!/usr/bin/env python3
"""
Simple HTTP test server for WandB MCP Server.
This wraps the MCP server functions in HTTP endpoints for easy testing.
"""
import asyncio
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_wandb_server.tools import list_runs, get_run_summary, get_run_metrics, compare_runs
from mcp_wandb_server.resources import read_resource


class MCPTestHandler(BaseHTTPRequestHandler):
    """HTTP handler for testing MCP server endpoints."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        # Set CORS headers for browser testing
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            # Route: /runs - List recent runs
            if path == '/runs' or path == '/runs/':
                try:
                    limit = int(query.get('limit', [10])[0])
                except (ValueError, IndexError):
                    limit = 10
                
                state = query.get('state', [None])[0]
                
                result = asyncio.run(list_runs(limit=limit, state=state))
                # Ensure result is valid JSON
                try:
                    # Parse to validate it's JSON
                    data = json.loads(result)
                    # Re-encode to ensure proper formatting
                    result = json.dumps(data, indent=None)
                except json.JSONDecodeError:
                    # If it's not JSON, wrap it
                    result = json.dumps({"error": "Invalid response from MCP server", "raw": result})
                
                self.wfile.write(result.encode('utf-8'))
                return
            
            # Route: /runs/recent - Get recent runs (resource)
            elif path == '/runs/recent':
                result = asyncio.run(read_resource('wandb://runs/recent'))
                if result and len(result) > 0:
                    self.wfile.write(result[0].text.encode('utf-8'))
                else:
                    self.wfile.write(json.dumps({"error": "No data"}).encode('utf-8'))
                return
            
            # Route: /runs/{run_id}/summary - Get run summary
            elif path.startswith('/runs/') and path.endswith('/summary'):
                run_id = path.split('/')[2]
                result = asyncio.run(get_run_summary(run_id))
                self.wfile.write(result.encode('utf-8'))
                return
            
            # Route: /runs/{run_id}/metrics - Get run metrics
            elif path.startswith('/runs/') and '/metrics' in path:
                parts = path.split('/')
                if len(parts) >= 4 and parts[3] == 'metrics':
                    run_id = parts[2]
                    metric_keys = query.get('metrics', [None])[0]
                    if metric_keys:
                        metric_keys = metric_keys.split(',')
                    result = asyncio.run(get_run_metrics(run_id, metric_keys))
                    self.wfile.write(result.encode('utf-8'))
                    return
            
            # Route: /runs/count - Count total runs
            elif path == '/runs/count':
                try:
                    # Get a large number of runs to count
                    result = asyncio.run(list_runs(limit=1000))
                    data = json.loads(result)
                    count = len(data) if isinstance(data, list) else 0
                    response = {"total_runs": count}
                    self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
                except Exception as e:
                    response = {"error": str(e)}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                return
            
            # Route: /health - Health check
            elif path == '/health' or path == '/':
                response = {
                    "status": "ok",
                    "server": "wandb-mcp-test-server",
                    "endpoints": {
                        "/runs": "List recent runs (query: ?limit=10&state=finished)",
                        "/runs/count": "Count total runs",
                        "/runs/recent": "Get recent runs resource",
                        "/runs/{run_id}/summary": "Get run summary",
                        "/runs/{run_id}/metrics": "Get run metrics (query: ?metrics=key1,key2)",
                        "/health": "Health check"
                    }
                }
                self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
                return
            
            # 404 for unknown routes
            else:
                self.send_response(404)
                response = {"error": f"Unknown endpoint: {path}"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return
                
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            self.send_response(500)
            response = {
                "error": error_msg,
                "traceback": traceback_str
            }
            self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to reduce log noise."""
        pass


def run_server(port=8000):
    """Run the HTTP test server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, MCPTestHandler)
    print(f"🚀 WandB MCP Test Server running on http://localhost:{port}")
    print(f"\nAvailable endpoints:")
    print(f"  GET http://localhost:{port}/health")
    print(f"  GET http://localhost:{port}/runs?limit=10")
    print(f"  GET http://localhost:{port}/runs/recent")
    print(f"  GET http://localhost:{port}/runs/{{run_id}}/summary")
    print(f"  GET http://localhost:{port}/runs/{{run_id}}/metrics?metrics=train_total_loss")
    print(f"\nPress Ctrl+C to stop\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        httpd.shutdown()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)

