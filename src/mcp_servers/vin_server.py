"""
mcp_servers/vin_server.py
--------------------------
MCP (Model Context Protocol) server that exposes a VIN decoding tool.
Calls the free NHTSA vPIC API to decode a Vehicle Identification Number.

This server is started as a subprocess and communicates via stdio.
Agents call it through the MCP client interface.

Usage (run as standalone process):
  python src/mcp_servers/vin_server.py

Tool exposed: decode_vin(vin: str) -> dict
  Returns: make, model, year, engine, fuel type
"""

import asyncio
import json
import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

server = Server("vin-decoder")

NHTSA_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"

FIELDS_WANTED = {
    "Make":               "make",
    "Model":              "model",
    "Model Year":         "year",
    "Engine Configuration": "engine",
    "Fuel Type - Primary": "fuel_type",
    "Plant Country":      "plant_country",
}


def _decode_vin_nhtsa(vin: str) -> dict:
    """Call NHTSA public API and extract key vehicle fields."""
    try:
        response = requests.get(NHTSA_URL.format(vin=vin), timeout=10)
        data = response.json()
        results = data.get("Results", [])
        vehicle = {"vin": vin}
        for item in results:
            key = FIELDS_WANTED.get(item["Variable"])
            if key and item.get("Value") and item["Value"] != "Not Applicable":
                vehicle[key] = item["Value"]
        return vehicle
    except Exception as exc:
        return {"vin": vin, "error": str(exc)}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="decode_vin",
            description=(
                "Decode a Vehicle Identification Number (VIN) to get "
                "make, model, year, engine, and fuel type."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "vin": {
                        "type": "string",
                        "description": "17-character Vehicle Identification Number",
                    }
                },
                "required": ["vin"],
            },
        )
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:
    if name == "decode_vin":
        vin = arguments.get("vin", "")
        result = _decode_vin_nhtsa(vin)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    asyncio.run(stdio_server(server))
