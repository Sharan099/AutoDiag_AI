import sys
import os
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AutoDiag AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.step-box {
    padding: 1rem 1.25rem;
    border-radius: 10px;
    margin-bottom: 0.75rem;
    border-left: 4px solid #ccc;
    background: #f9f9f9;
}
.step-box.running  { border-left-color: #f59e0b; background: #fffbeb; }
.step-box.done     { border-left-color: #10b981; background: #ecfdf5; }
.step-box.pending  { border-left-color: #d1d5db; background: #f9fafb; }
.step-box.error    { border-left-color: #ef4444; background: #fef2f2; }

.step-title  { font-weight: 600; font-size: 15px; margin-bottom: 4px; }
.step-detail { font-size: 13px; color: #555; }

.metric-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.metric-val  { font-size: 22px; font-weight: 700; color: #065f46; }
.metric-lbl  { font-size: 12px; color: #6b7280; margin-top: 2px; }

.source-chip {
    display: inline-block;
    background: #ede9fe;
    color: #4c1d95;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    margin: 2px;
}
.report-box {
    background: #1e1e2e;
    color: #cdd6f4;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-family: monospace;
    font-size: 13px;
    white-space: pre-wrap;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ── sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/car-engine.png", width=60)
    st.title("AutoDiag AI")
    st.caption("Automotive Fault Diagnosis · German OEMs")

    st.divider()
    st.subheader("About")
    st.markdown("""
This app runs the full **AutoDiag AI** pipeline:

1. 🔍 Decodes your VIN via NHTSA API
2. 📄 Retrieves relevant TSBs from the vector store
3. 🤖 Runs 4 CrewAI agents
4. 📋 Generates a structured repair report

**LLM:** `llama3.2:1b` via Ollama (local)
**Embeddings:** `BAAI/bge-small-en-v1.5`
    """)

    st.divider()
    st.subheader("Ollama status")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            st.success("Ollama is running")
            for m in models:
                st.caption(f"  ✓ {m}")
        else:
            st.error("Ollama not responding")
    except Exception:
        st.error("Ollama offline — start with: `ollama serve`")

    st.divider()
    st.subheader("Sample DTCs")
    st.markdown("""
| Code | Description |
|------|-------------|
| P0301 | Cylinder 1 misfire |
| P0420 | Catalyst efficiency |
| P0087 | Fuel pressure low |
| P0730 | Incorrect gear ratio |
| P0171 | System too lean |
| P0299 | Turbo underboost |
    """)


# ── main page ────────────────────────────────────────────────
st.title("🔧 AutoDiag AI — Fault Diagnosis")
st.markdown("Enter a VIN and DTC codes to run the full multi-agent diagnosis pipeline.")

# ── input form ───────────────────────────────────────────────
with st.form("diagnosis_form"):
    col1, col2 = st.columns([2, 3])

    with col1:
        vin_input = st.text_input(
            "Vehicle VIN",
            value="WBA8E9G50HNU60241",
            placeholder="17-character VIN",
            help="Enter a real VIN. NHTSA decodes make/model/year automatically.",
        )

    with col2:
        dtc_input = st.text_input(
            "DTC Codes (comma-separated)",
            value="P0301",
            placeholder="P0301, P0420, P0087",
            help="One or more OBD-II fault codes separated by commas.",
        )

    submitted = st.form_submit_button(
        "▶  Run Diagnosis", type="primary", use_container_width=True
    )

st.divider()

# ── pipeline execution ───────────────────────────────────────
if submitted:
    dtc_codes = [d.strip().upper() for d in dtc_input.split(",") if d.strip()]

    if not vin_input or len(vin_input) < 5:
        st.error("Please enter a valid VIN (at least 5 characters).")
        st.stop()

    if not dtc_codes:
        st.error("Please enter at least one DTC code.")
        st.stop()

    # ── layout: steps on left, live output on right
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Pipeline steps")
        step1_ph = st.empty()
        step2_ph = st.empty()
        step3_ph = st.empty()
        step4_ph = st.empty()

    with right:
        st.subheader("Live output")
        output_ph = st.empty()

    def render_step(placeholder, icon, title, detail, status):
        css = f"step-box {status}"
        placeholder.markdown(
            f'<div class="{css}">'
            f'<div class="step-title">{icon} {title}</div>'
            f'<div class="step-detail">{detail}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # initialise all steps as pending
    render_step(step1_ph, "🔍", "Decode VIN",       "Waiting...", "pending")
    render_step(step2_ph, "📄", "Retrieve TSBs",    "Waiting...", "pending")
    render_step(step3_ph, "🤖", "Run CrewAI agents","Waiting...", "pending")
    render_step(step4_ph, "📋", "Format report",    "Waiting...", "pending")

    pipeline_start = time.time()
    state = {
        "vin":          vin_input,
        "dtc_codes":    dtc_codes,
        "vehicle_info": {},
        "tsb_docs":     [],
        "crew_result":  "",
        "final_report": "",
        "error":        None,
    }

    # ── STEP 1: VIN decode ─────────────────────────────────
    render_step(step1_ph, "🔍", "Decode VIN", "Calling NHTSA API...", "running")
    output_ph.info("Decoding VIN via NHTSA API...")
    t0 = time.time()

    try:
        NHTSA_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"
        fields = {
            "Make": "make", "Model": "model", "Model Year": "year",
            "Engine Configuration": "engine", "Fuel Type - Primary": "fuel_type",
        }
        resp = requests.get(NHTSA_URL.format(vin=vin_input), timeout=10)
        results = resp.json().get("Results", [])
        vehicle = {"vin": vin_input}
        for item in results:
            key = fields.get(item["Variable"])
            if key and item.get("Value") and item["Value"] != "Not Applicable":
                vehicle[key] = item["Value"]
        state["vehicle_info"] = vehicle

        label = f"{vehicle.get('make','')} {vehicle.get('model','')} {vehicle.get('year','')}"
        render_step(step1_ph, "🔍", "Decode VIN",
                    f"✓ {label}  ({time.time()-t0:.1f}s)", "done")
        output_ph.success(f"VIN decoded: **{label}**")

    except Exception as e:
        state["vehicle_info"] = {"vin": vin_input}
        render_step(step1_ph, "🔍", "Decode VIN",
                    f"Network error — using VIN only", "error")
        output_ph.warning(f"VIN decode failed: {e}. Continuing with VIN only.")

    # ── STEP 2: TSB retrieval ──────────────────────────────
    render_step(step2_ph, "📄", "Retrieve TSBs",
                "Searching vector store...", "running")
    output_ph.info("Searching FAISS vector store for relevant TSBs...")
    t0 = time.time()

    try:
        from src.rag import load_retriever

        vehicle = state["vehicle_info"]
        query   = (
            f"{vehicle.get('make','')} {vehicle.get('model','')} "
            f"{vehicle.get('year','')} DTC: {' '.join(dtc_codes)}"
        )
        retriever = load_retriever()
        docs      = retriever.invoke(query)
        tsb_texts = [d.page_content for d in docs]
        state["tsb_docs"] = tsb_texts

        render_step(step2_ph, "📄", "Retrieve TSBs",
                    f"✓ {len(tsb_texts)} chunks retrieved  ({time.time()-t0:.1f}s)", "done")

        with output_ph.container():
            st.success(f"Retrieved **{len(tsb_texts)} TSB chunks**")
            for i, chunk in enumerate(tsb_texts[:3], 1):
                with st.expander(f"Chunk {i} — {chunk[:60]}..."):
                    st.text(chunk)

    except Exception as e:
        render_step(step2_ph, "📄", "Retrieve TSBs",
                    f"Error: {str(e)[:60]}", "error")
        output_ph.error(f"TSB retrieval failed: {e}")
        st.stop()

    # ── STEP 3: CrewAI agents ──────────────────────────────
    render_step(step3_ph, "🤖", "Run CrewAI agents",
                "4 agents working...", "running")

    agent_names = [
        "Diagnosis Agent",
        "Root Cause Agent",
        "Parts Agent",
        "Report Agent",
    ]

    with output_ph.container():
        st.info("Running 4 CrewAI agents sequentially...")
        agent_bars = {}
        for name in agent_names:
            agent_bars[name] = st.empty()
            agent_bars[name].markdown(f"⏳ **{name}** — waiting...")

    t0 = time.time()

    try:
        from src.agents import run_diagnosis_crew

        # update agent status markers as crew runs
        for name in agent_names:
            agent_bars[name].markdown(f"🔄 **{name}** — running...")

        tsb_context  = "\n\n---\n\n".join(state["tsb_docs"][:3])
        crew_result  = run_diagnosis_crew(
            vehicle_info=state["vehicle_info"],
            dtc_codes=dtc_codes,
            tsb_context=tsb_context,
        )
        state["crew_result"] = crew_result

        for name in agent_names:
            agent_bars[name].markdown(f"✅ **{name}** — done")

        render_step(step3_ph, "🤖", "Run CrewAI agents",
                    f"✓ All 4 agents completed  ({time.time()-t0:.1f}s)", "done")

    except Exception as e:
        render_step(step3_ph, "🤖", "Run CrewAI agents",
                    f"Error: {str(e)[:60]}", "error")
        output_ph.error(f"CrewAI failed: {e}")
        st.stop()

    # ── STEP 4: format report ──────────────────────────────
    render_step(step4_ph, "📋", "Format report",
                "Assembling...", "running")
    t0 = time.time()

    vehicle = state["vehicle_info"]
    report  = (
        "=" * 50 + "\n"
        "AUTODIAG AI — DIAGNOSTIC REPORT\n"
        + "=" * 50 + "\n\n"
        f"Vehicle : {vehicle.get('make','')} {vehicle.get('model','')} ({vehicle.get('year','')})\n"
        f"VIN     : {vehicle.get('vin','N/A')}\n"
        f"DTC     : {', '.join(dtc_codes)}\n"
        f"Sources : {len(state['tsb_docs'])} TSB documents retrieved\n\n"
        "DIAGNOSIS:\n"
        + "-" * 30 + "\n"
        + state["crew_result"]
        + "\n\nGenerated by AutoDiag AI — for professional use only.\n"
    )
    state["final_report"] = report
    total_time = round(time.time() - pipeline_start, 1)

    render_step(step4_ph, "📋", "Format report",
                f"✓ Report ready  ({time.time()-t0:.1f}s)", "done")

    # ── RESULTS ────────────────────────────────────────────
    st.divider()
    st.subheader("Results")

    # summary metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f'<div class="metric-card"><div class="metric-val">'
        f'{vehicle.get("make","—")}</div>'
        f'<div class="metric-lbl">Make</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card"><div class="metric-val">'
        f'{vehicle.get("year","—")}</div>'
        f'<div class="metric-lbl">Year</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card"><div class="metric-val">'
        f'{len(dtc_codes)}</div>'
        f'<div class="metric-lbl">DTC codes</div></div>',
        unsafe_allow_html=True,
    )
    m4.markdown(
        f'<div class="metric-card"><div class="metric-val">'
        f'{total_time}s</div>'
        f'<div class="metric-lbl">Total time</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # tabs for report / raw sources
    tab1, tab2, tab3 = st.tabs(["📋 Diagnosis Report", "📄 TSB Sources", "🔍 Raw Crew Output"])

    with tab1:
        st.markdown(
            f'<div class="report-box">{report}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇  Download Report (.txt)",
            data=report,
            file_name=f"autodiag_{vin_input[:8]}_{dtc_codes[0]}.txt",
            mime="text/plain",
        )

    with tab2:
        if state["tsb_docs"]:
            for i, chunk in enumerate(state["tsb_docs"], 1):
                with st.expander(f"TSB Chunk {i}"):
                    st.text(chunk)
        else:
            st.info("No TSB chunks were retrieved.")

    with tab3:
        st.text_area(
            "Raw CrewAI output",
            value=state["crew_result"],
            height=400,
        )

# ── empty state (no submission yet) ──────────────────────────
else:
    st.markdown("""
<div style="text-align:center; padding: 3rem 0; color: #9ca3af;">
    <div style="font-size: 48px;">🔧</div>
    <div style="font-size: 18px; margin-top: 1rem;">Enter a VIN and DTC codes above and click Run Diagnosis</div>
    <div style="font-size: 14px; margin-top: 0.5rem;">The full LangGraph + CrewAI pipeline will run and show results here</div>
</div>
    """, unsafe_allow_html=True)
