import os
import time
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AutoDiag AI",
    description="AI-powered automotive fault diagnosis for German OEM vehicles",
    version="1.0.0",
)


class DiagnoseRequest(BaseModel):
    vin:       str
    dtc_codes: List[str]


class DiagnoseResponse(BaseModel):
    report:       str
    vehicle_info: dict
    sources_used: int
    time_seconds: float


@app.get("/")
def root():
    return {"message": "AutoDiag AI is running. POST to /diagnose"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(request: DiagnoseRequest):
    """
    Run the full diagnosis pipeline:
    VIN decode -> TSB retrieval -> CrewAI agents -> formatted report
    """
    if not request.vin or len(request.vin) < 5:
        raise HTTPException(status_code=400, detail="VIN must be at least 5 characters")
    if not request.dtc_codes:
        raise HTTPException(status_code=400, detail="At least one DTC code required")

    start = time.time()

    try:
        from src.graph import build_graph

        graph = build_graph()
        state = graph.invoke({
            "vin":          request.vin,
            "dtc_codes":    request.dtc_codes,
            "vehicle_info": {},
            "tsb_docs":     [],
            "crew_result":  "",
            "final_report": "",
            "error":        None,
        })

        return DiagnoseResponse(
            report=state["final_report"],
            vehicle_info=state["vehicle_info"],
            sources_used=len(state.get("tsb_docs", [])),
            time_seconds=round(time.time() - start, 2),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
