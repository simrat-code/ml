# mcp_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
import requests
import os

app = FastAPI(title="Cybersecurity MCP Server")

# ==== Data Models ====
class CVERequest(BaseModel):
    cve_id: str

class LogResponse(BaseModel):
    logs: str

class CVEResponse(BaseModel):
    cve_id: str
    description: str
    severity: str

# ==== Endpoints (MCP methods) ====

@app.get("/get_system_logs", response_model=LogResponse)
def get_system_logs():
    # log_path = "/var/log/syslog"
    log_path = os.path.join(os.getcwd(), "mock", "syslog.txt")  # For testing purposes, use a mock log file
    if not os.path.exists(log_path):
        return {"logs": "No syslog found."}
    with open(log_path, "r") as f:
        lines = f.readlines()[-20:]  # Last 20 lines for safety
    return {"logs": "".join(lines)}

@app.post("/get_cve_info", response_model=CVEResponse)
def get_cve_info(req: CVERequest):
    # Normally you'd call NVD API here.
    # This is a mocked response for simplicity.
    cve_db = {
        "CVE-2024-1234": {"desc": "Buffer overflow in OpenSSL.", "sev": "High"},
        "CVE-2023-5678": {"desc": "Privilege escalation in Linux kernel.", "sev": "Critical"},
    }
    data = cve_db.get(req.cve_id, {"desc": "Unknown CVE.", "sev": "Unknown"})
    return {"cve_id": req.cve_id, "description": data["desc"], "severity": data["sev"]}

# ==== MCP Protocol Metadata ====
@app.get("/mcp/metadata")
def get_metadata():
    """
    This describes to the AI agent what capabilities this MCP server exposes.
    """
    return {
        "protocol": "model-context-protocol",
        "version": "1.0",
        "tools": [
            {"name": "get_system_logs", "method": "GET"},
            {"name": "get_cve_info", "method": "POST", "params": ["cve_id"]}
        ]
    }

# ==== Run Server ====
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
