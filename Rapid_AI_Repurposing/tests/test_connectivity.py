"""
test_connectivity.py
====================
Consolidated external service connectivity test suite for Rapid AI:
- Local Ollama LLM endpoint (Llama 3.2)
- PubChem PUG-REST API
- ClinicalTrials.gov v2 API
"""

import sys
import argparse
import requests

def test_ollama():
    print("--- [1/2] Testing Local Ollama (Llama 3.2) ---")
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "llama3.2:latest",
        "prompt": "Say 'OK' in one word.",
        "stream": False
    }
    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print(f"  [PASS] Ollama HTTP Response: {resp.json().get('response', '').strip()}")
            return True
        else:
            print(f"  [FAIL] Ollama HTTP Error: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"  [SKIP] Ollama not running or unreachable ({e})")
        return False

def test_external_apis():
    print("--- [2/2] Testing External Biomedical APIs ---")
    # Test PubChem
    pubchem_pass = False
    try:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/property/CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            smiles = resp.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
            print(f"  [PASS] PubChem API: Aspirin SMILES -> {smiles}")
            pubchem_pass = True
        else:
            print(f"  [FAIL] PubChem API: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [FAIL] PubChem API: {e}")

    # Test ClinicalTrials.gov
    ct_pass = False
    try:
        url = "https://clinicaltrials.gov/api/v2/studies?query.term=somatotropin+fracture&pageSize=1"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            total = resp.json().get('totalCount', 'N/A')
            print(f"  [PASS] ClinicalTrials.gov v2 API: Search returned {total} study records")
            ct_pass = True
        else:
            print(f"  [FAIL] ClinicalTrials.gov API: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [FAIL] ClinicalTrials.gov API: {e}")

    return pubchem_pass and ct_pass

def main():
    parser = argparse.ArgumentParser(description="Rapid AI Connectivity Test Suite")
    parser.add_argument("--mode", choices=["all", "ollama", "apis"], default="all")
    args = parser.parse_args()

    if args.mode in ["all", "ollama"]:
        test_ollama()
    if args.mode in ["all", "apis"]:
        test_external_apis()

if __name__ == "__main__":
    main()
