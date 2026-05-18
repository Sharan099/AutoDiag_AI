"""
create_sample_data.py
---------------------
Creates synthetic automotive TSB and recall data so you can run
the full pipeline immediately without downloading NHTSA files.
Run this first if you haven't downloaded NHTSA data yet.

Usage: python src/create_sample_data.py
"""

import os
import pandas as pd

PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

TSB_RECORDS = [
    {
        "Make": "BMW", "Model": "3 Series", "Model Year": 2021,
        "NHTSA Components": "ENGINE",
        "NHTSA ID Number": "TSB-BMW-001",
        "document_text": (
            "Make: BMW\nModel: 3 Series\nYear: 2021\nComponent: ENGINE\n"
            "TSB ID: TSB-BMW-001\n"
            "Summary: Intermittent engine misfire P0301 may occur on vehicles with B58 engine. "
            "Root cause is a loose crankshaft position sensor connector caused by vibration. "
            "Fix: replace connector harness using part #13627797870. Torque to 5 Nm. "
            "Clear all DTCs and perform 20-minute road test after repair."
        )
    },
    {
        "Make": "BMW", "Model": "5 Series", "Model Year": 2020,
        "NHTSA Components": "ENGINE",
        "NHTSA ID Number": "TSB-BMW-002",
        "document_text": (
            "Make: BMW\nModel: 5 Series\nYear: 2020\nComponent: ENGINE\n"
            "TSB ID: TSB-BMW-002\n"
            "Summary: P0420 catalyst system efficiency below threshold on B48 engine. "
            "Common cause is failed upstream oxygen sensor (part #11787589071). "
            "Check for exhaust leaks before replacing catalytic converter. "
            "Replace O2 sensor first, clear codes, retest after 100km drive cycle."
        )
    },
    {
        "Make": "VOLKSWAGEN", "Model": "Golf", "Model Year": 2022,
        "NHTSA Components": "FUEL SYSTEM",
        "NHTSA ID Number": "TSB-VW-001",
        "document_text": (
            "Make: VOLKSWAGEN\nModel: Golf\nYear: 2022\nComponent: FUEL SYSTEM\n"
            "TSB ID: TSB-VW-001\n"
            "Summary: DTC P0087 fuel rail pressure too low on EA888 engine. "
            "High-pressure fuel pump (HPFP) wear causes insufficient fuel pressure at high RPM. "
            "Replace HPFP part #06K127025M. Also inspect low-pressure fuel lines for restrictions. "
            "After replacement, run fuel system adaptation using VCDS diagnostic tool."
        )
    },
    {
        "Make": "VOLKSWAGEN", "Model": "Tiguan", "Model Year": 2021,
        "NHTSA Components": "ELECTRICAL SYSTEM",
        "NHTSA ID Number": "TSB-VW-002",
        "document_text": (
            "Make: VOLKSWAGEN\nModel: Tiguan\nYear: 2021\nComponent: ELECTRICAL SYSTEM\n"
            "TSB ID: TSB-VW-002\n"
            "Summary: U0100 lost communication with ECM/PCM may occur after battery replacement. "
            "Cause: voltage drop during battery swap disrupts CAN bus gateway. "
            "Solution: perform basic settings reset using VCDS software version 21.9 or newer. "
            "Connect to OBD port, navigate to Gateway module, run adaptation channel reset."
        )
    },
    {
        "Make": "MERCEDES-BENZ", "Model": "C-Class", "Model Year": 2021,
        "NHTSA Components": "TRANSMISSION",
        "NHTSA ID Number": "TSB-MB-001",
        "document_text": (
            "Make: MERCEDES-BENZ\nModel: C-Class\nYear: 2021\nComponent: TRANSMISSION\n"
            "TSB ID: TSB-MB-001\n"
            "Summary: P0730 incorrect gear ratio detected on 9G-Tronic transmission. "
            "Valve body solenoid contamination causes erratic shifting. "
            "Perform ATF fluid flush and replace solenoid kit A0002770101. "
            "Transmission adaptation reset required using XENTRY diagnostic software after repair."
        )
    },
    {
        "Make": "MERCEDES-BENZ", "Model": "E-Class", "Model Year": 2020,
        "NHTSA Components": "ENGINE",
        "NHTSA ID Number": "TSB-MB-002",
        "document_text": (
            "Make: MERCEDES-BENZ\nModel: E-Class\nYear: 2020\nComponent: ENGINE\n"
            "TSB ID: TSB-MB-002\n"
            "Summary: P0016 camshaft position timing over-advanced on OM654 diesel. "
            "Stretch in timing chain causes P0016 and rough idle. "
            "Inspect timing chain tensioner and guides. Replace chain kit A6540500300 if stretch exceeds 2mm. "
            "Engine oil quality is critical - use only MB 229.51 specification oil to prevent recurrence."
        )
    },
    {
        "Make": "AUDI", "Model": "A4", "Model Year": 2022,
        "NHTSA Components": "ENGINE",
        "NHTSA ID Number": "TSB-AUDI-001",
        "document_text": (
            "Make: AUDI\nModel: A4\nYear: 2022\nComponent: ENGINE\n"
            "TSB ID: TSB-AUDI-001\n"
            "Summary: P0171 system too lean bank 1 on 2.0 TFSI engine. "
            "Intake manifold leak or faulty mass airflow sensor. "
            "Inspect intake boot for cracks and check MAF sensor values with VCDS. "
            "MAF reading at idle should be 3-5 g/s. Replace MAF sensor 06H906461F if out of range."
        )
    },
    {
        "Make": "PORSCHE", "Model": "Cayenne", "Model Year": 2021,
        "NHTSA Components": "BRAKES",
        "NHTSA ID Number": "TSB-POR-001",
        "document_text": (
            "Make: PORSCHE\nModel: Cayenne\nYear: 2021\nComponent: BRAKES\n"
            "TSB ID: TSB-POR-001\n"
            "Summary: C1511 PCCB ceramic brake system fault. Brake pad wear sensor resistance out of range. "
            "Resistance should be 1000-1200 ohms when new. Replace worn sensors part #95B698907B. "
            "Clear fault with PASM diagnostic. Verify brake pad thickness minimum 3mm before clearing."
        )
    },
    {
        "Make": "BMW", "Model": "X5", "Model Year": 2022,
        "NHTSA Components": "ENGINE",
        "NHTSA ID Number": "TSB-BMW-003",
        "document_text": (
            "Make: BMW\nModel: X5\nYear: 2022\nComponent: ENGINE\n"
            "TSB ID: TSB-BMW-003\n"
            "Summary: P0299 turbocharger underboost condition on B57 diesel engine. "
            "EGR cooler bypass valve sticking causes reduced boost pressure. "
            "Clean EGR valve assembly and replace bypass actuator 11717823212. "
            "Boost pressure at 3000 RPM should be minimum 1.8 bar. Verify with ISTA after repair."
        )
    },
    {
        "Make": "VOLKSWAGEN", "Model": "Passat", "Model Year": 2020,
        "NHTSA Components": "EXHAUST SYSTEM",
        "NHTSA ID Number": "TSB-VW-003",
        "document_text": (
            "Make: VOLKSWAGEN\nModel: Passat\nYear: 2020\nComponent: EXHAUST SYSTEM\n"
            "TSB ID: TSB-VW-003\n"
            "Summary: P2002 diesel particulate filter efficiency below threshold on TDI engine. "
            "Excessive short-trip driving prevents DPF regeneration. "
            "Perform forced DPF regeneration using VCDS. Drive at motorway speed for 30 minutes. "
            "If pressure differential exceeds 60mbar replace DPF 5Q0254700HX."
        )
    },
]

RECALL_RECORDS = [
    {
        "MAKETXT": "BMW", "MODELTXT": "3 Series", "YEARTXT": "2021",
        "COMPNAME": "ENGINE",
        "document_text": (
            "RECALL — Make: BMW\nModel: 3 Series\nYear: 2021\nComponent: ENGINE\n"
            "Defect: Fuel injector O-ring may crack due to thermal cycling causing fuel leak.\n"
            "Consequence: Fuel leak in engine bay creates fire risk.\n"
            "Remedy: Dealer will replace all fuel injector O-rings free of charge. "
            "Part #13537619680. Repair time approximately 2 hours."
        )
    },
    {
        "MAKETXT": "VOLKSWAGEN", "MODELTXT": "Golf", "YEARTXT": "2022",
        "COMPNAME": "AIRBAGS",
        "document_text": (
            "RECALL — Make: VOLKSWAGEN\nModel: Golf\nYear: 2022\nComponent: AIRBAGS\n"
            "Defect: Passenger airbag inflator may deploy with excessive force.\n"
            "Consequence: Metal fragments could cause injury to occupants.\n"
            "Remedy: Dealer will replace airbag inflator assembly 5G0880204K at no cost. "
            "Repair takes approximately 1 hour."
        )
    },
    {
        "MAKETXT": "MERCEDES-BENZ", "MODELTXT": "C-Class", "YEARTXT": "2021",
        "COMPNAME": "STEERING",
        "document_text": (
            "RECALL — Make: MERCEDES-BENZ\nModel: C-Class\nYear: 2021\nComponent: STEERING\n"
            "Defect: Electric power steering software may cause momentary loss of power assist.\n"
            "Consequence: Increased steering effort required, crash risk at low speeds.\n"
            "Remedy: Software update via XENTRY. Update takes 45 minutes. "
            "Dealers notified October 2022."
        )
    },
]


def create_sample_data():
    tsbs_df = pd.DataFrame(TSB_RECORDS)
    recalls_df = pd.DataFrame(RECALL_RECORDS)

    tsbs_df.to_csv(f"{PROC_DIR}/tsbs_german.csv", index=False)
    recalls_df.to_csv(f"{PROC_DIR}/recalls_german.csv", index=False)

    print(f"Created {len(tsbs_df)} sample TSB records -> data/processed/tsbs_german.csv")
    print(f"Created {len(recalls_df)} sample Recall records -> data/processed/recalls_german.csv")
    print("Sample data ready. Run: python src/build_vectorstore.py")


if __name__ == "__main__":
    create_sample_data()
