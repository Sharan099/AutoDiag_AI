"""
Agents:
  1. DiagnosisAgent   - interprets DTC codes using TSB context
  2. RootCauseAgent   - identifies the underlying failure mechanism
  3. PartsAgent       - determines required replacement parts
  4. ReportAgent      - writes the final mechanic-readable report

"""

import os
import json
import requests
from crewai import Agent, Task, Crew, Process, LLM
from langchain_ollama import OllamaLLM
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
LLM_MODEL  = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b")

def get_llm():
    return LLM(
        model=f"ollama/{LLM_MODEL}",
        base_url=OLLAMA_URL,
        temperature=0.1,
    )


@tool
def parts_lookup_tool(input_str: str) -> str:

    """
    Look up replacement parts for a DTC code. Input format: 'DTC_CODE,MAKE' e.g. 'P0301,BMW'

    """
    try:
        parts = input_str.split(",")
        dtc  = parts[0].strip() if len(parts) > 0 else ""
        make = parts[1].strip() if len(parts) > 1 else "DEFAULT"

        from src.mcp_servers.parts_server import _lookup
        result = _lookup(dtc, make)
        return json.dumps(result) if result else f"No parts found for {dtc} on {make}"
    except Exception as exc:
        return f"Parts lookup error: {exc}"


def build_agents(llm):
    diagnosis_agent = Agent(
        role="Senior Automotive Diagnostic Engineer",
        goal="Interpret OBD-II DTC fault codes and identify the faulty component",
        backstory=(
            "You have 15 years of experience diagnosing BMW, VW, Audi, "
            "Mercedes-Benz, and Porsche vehicles. You specialise in interpreting "
            "diagnostic trouble codes and cross-referencing TSBs."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    root_cause_agent = Agent(
        role="Root Cause Analysis Specialist",
        goal="Identify why the component failed and the underlying mechanism",
        backstory=(
            "You are an expert in automotive failure mode analysis. "
            "You explain not just what failed, but exactly why, "
            "and what conditions led to the failure."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    parts_agent = Agent(
        role="Parts and Supply Chain Specialist",
        goal="Identify the exact replacement parts with part numbers and prices",
        backstory=(
            "You have deep knowledge of OEM and aftermarket parts for "
            "German vehicles. You always provide part numbers, prices, "
            "and whether OEM or quality aftermarket is acceptable."
        ),
        tools=[parts_lookup_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    report_agent = Agent(
        role="Service Advisor Report Writer",
        goal="Write a clear, structured repair report that mechanics can follow",
        backstory=(
            "You have worked as a service advisor at authorised BMW and VW dealers. "
            "You write reports that are technically accurate but also "
            "clear enough for junior mechanics to follow."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return diagnosis_agent, root_cause_agent, parts_agent, report_agent


def build_tasks(diagnosis_agent, root_cause_agent, parts_agent, report_agent,
                vehicle_info: dict, dtc_codes: list, tsb_context: str):

    context_block = (
        f"Vehicle: {vehicle_info.get('make','')} "
        f"{vehicle_info.get('model','')} "
        f"({vehicle_info.get('year','')})\n"
        f"DTC Codes: {', '.join(dtc_codes)}\n\n"
        f"Relevant TSB / Recall data:\n{tsb_context}"
    )

    task1 = Task(
        description=(
            f"Analyse these DTC codes and identify which component has failed.\n\n{context_block}"
        ),
        expected_output=(
            "Component name, description of the fault, and which DTC relates to which component."
        ),
        agent=diagnosis_agent,
    )

    task2 = Task(
        description="Based on the diagnosis, explain exactly why this component failed.",
        expected_output=(
            "Root cause explanation covering: failure mechanism, "
            "contributing factors, and how to confirm the diagnosis."
        ),
        agent=root_cause_agent,
        context=[task1],
    )

    task3 = Task(
        description=(
            f"Find the replacement parts needed. "
            f"Use the parts_lookup tool with format 'DTC,MAKE'. "
            f"DTC: {dtc_codes[0] if dtc_codes else 'UNKNOWN'}, "
            f"Make: {vehicle_info.get('make', 'DEFAULT')}"
        ),
        expected_output="List of parts with part numbers and prices in EUR.",
        agent=parts_agent,
        context=[task1, task2],
    )

    task4 = Task(
        description="Write the final repair report combining all findings.",
        expected_output=(
            "Structured report with: 1) Fault Summary 2) Root Cause "
            "3) Repair Steps 4) Parts List 5) Estimated Time"
        ),
        agent=report_agent,
        context=[task1, task2, task3],
    )

    return [task1, task2, task3, task4]


def run_diagnosis_crew(vehicle_info: dict, dtc_codes: list, tsb_context: str) -> str:
    """
    Run the 4-agent CrewAI crew and return the final report string.
    """
    llm = get_llm()
    diagnosis_agent, root_cause_agent, parts_agent, report_agent = build_agents(llm)
    tasks = build_tasks(
        diagnosis_agent, root_cause_agent, parts_agent, report_agent,
        vehicle_info, dtc_codes, tsb_context,
    )

    crew = Crew(
        agents=[diagnosis_agent, root_cause_agent, parts_agent, report_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    sample_vehicle = {"make": "BMW", "model": "3 Series", "year": "2021"}
    sample_tsb = (
        "Make: BMW | Model: 3 Series | Year: 2021 | Component: ENGINE\n"
        "Summary: Cylinder 1 misfire P0301. Replace crankshaft sensor connector #13627797870."
    )
    print("Running CrewAI diagnosis ...")
    report = run_diagnosis_crew(
        vehicle_info=sample_vehicle,
        dtc_codes=["P0301"],
        tsb_context=sample_tsb,
    )
    print("\n=== CREW REPORT ===")
    print(report)
