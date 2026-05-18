"""
mcp_servers/parts_server.py
----------------------------
MCP server that exposes a parts catalog lookup tool.
Uses a local mock catalog (extend with real AutoZone / TecDoc API later).

Usage (run as standalone process):
  python src/mcp_servers/parts_server.py

Tool exposed: lookup_parts(dtc_code: str, make: str) -> list[dict]
  Returns: part name, part number, estimated price, availability
"""

import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

server = Server("parts-catalog")

PARTS_CATALOG = {
    "P0301": {
        "BMW":        [{"name": "Crankshaft Position Sensor Connector", "part_no": "13627797870", "price_eur": 28.50}],
        "VOLKSWAGEN": [{"name": "Ignition Coil Pack",                  "part_no": "06K905110",   "price_eur": 45.00}],
        "DEFAULT":    [{"name": "Spark Plug Set",                      "part_no": "GENERIC-P0301","price_eur": 35.00}],
    },
    "P0420": {
        "BMW":        [{"name": "Upstream O2 Sensor",         "part_no": "11787589071", "price_eur": 89.00}],
        "VOLKSWAGEN": [{"name": "Lambda Sensor (upstream)",   "part_no": "06H906262E",  "price_eur": 75.00}],
        "DEFAULT":    [{"name": "Oxygen Sensor (upstream)",   "part_no": "GENERIC-P0420","price_eur": 80.00}],
    },
    "P0087": {
        "VOLKSWAGEN": [{"name": "High Pressure Fuel Pump",    "part_no": "06K127025M",  "price_eur": 320.00}],
        "DEFAULT":    [{"name": "Fuel Pressure Regulator",   "part_no": "GENERIC-P0087","price_eur": 65.00}],
    },
    "P0730": {
        "MERCEDES-BENZ": [{"name": "Transmission Solenoid Kit", "part_no": "A0002770101", "price_eur": 185.00}],
        "DEFAULT":       [{"name": "ATF Filter Kit",            "part_no": "GENERIC-P0730","price_eur": 55.00}],
    },
    "P0016": {
        "MERCEDES-BENZ": [{"name": "Timing Chain Kit",   "part_no": "A6540500300", "price_eur": 420.00}],
        "DEFAULT":       [{"name": "Timing Chain Tensioner","part_no": "GENERIC-P0016","price_eur": 95.00}],
    },
    "P0171": {
        "AUDI":   [{"name": "Mass Airflow Sensor",   "part_no": "06H906461F", "price_eur": 145.00}],
        "DEFAULT":[{"name": "Air Intake Boot",       "part_no": "GENERIC-P0171","price_eur": 35.00}],
    },
    "P0299": {
        "BMW":    [{"name": "EGR Bypass Actuator",   "part_no": "11717823212", "price_eur": 210.00}],
        "DEFAULT":[{"name": "Turbo Boost Sensor",    "part_no": "GENERIC-P0299","price_eur": 55.00}],
    },
}


def _lookup(dtc_code: str, make: str) -> list:
    code = dtc_code.upper().strip()
    make = make.upper().strip()
    entry = PARTS_CATALOG.get(code, {})
    parts = entry.get(make) or entry.get("DEFAULT") or []
    return parts


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="lookup_parts",
            description="Look up replacement parts for a given DTC code and vehicle make.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dtc_code": {"type": "string", "description": "OBD-II DTC fault code, e.g. P0301"},
                    "make":     {"type": "string", "description": "Vehicle make, e.g. BMW"},
                },
                "required": ["dtc_code"],
            },
        )
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:
    if name == "lookup_parts":
        dtc  = arguments.get("dtc_code", "")
        make = arguments.get("make", "DEFAULT")
        parts = _lookup(dtc, make)
        result = {"dtc_code": dtc, "make": make, "parts": parts}
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    asyncio.run(stdio_server(server))
