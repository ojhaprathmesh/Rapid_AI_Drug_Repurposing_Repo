import requests
import json

def test_pubchem(drug_name):
    print(f"Testing PubChem for: {drug_name}")
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"SMILES: {resp.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']}")
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"PubChem Exception: {e}")

def test_clinical_trials(drug, disease):
    print(f"Testing ClinicalTrials.gov for: {drug} + {disease}")
    try:
        url = f"https://clinicaltrials.gov/api/v2/studies?query.term={drug}+{disease}&pageSize=1"
        resp = requests.get(url, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Total Count: {resp.json().get('totalCount', 'Missing')}")
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"ClinicalTrials Exception: {e}")

if __name__ == "__main__":
    test_pubchem("Aspirin")
    print("-" * 20)
    test_clinical_trials("Aspirin", "Diabetes")
