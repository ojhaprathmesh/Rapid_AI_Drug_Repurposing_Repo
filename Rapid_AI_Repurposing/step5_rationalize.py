import os
import pandas as pd
import ollama
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR
PREDICTIONS_PATH = os.path.join(PROJECT_DIR, "top_50_repurposing_predictions.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "clinical_rationalization_report.md")

# SETTINGS
NUM_CANDIDATES = 10  # Explain top 10 for speed, change to 50 for full report
MODEL_NAME = "llama3.2"

print(f"--- STARTING STEP 5: Clinical Rationalization with {MODEL_NAME} ---")

def generate_rationale(drug, disease):
    prompt = f"""
    You are a clinical pharmacologist and molecular biologist.
    We have an AI model that predicts drug repurposing candidates using a Graph Neural Network on the DRKG dataset.
    
    The AI has predicted that the drug '{drug}' could potentially treat '{disease}'.
    
    Task:
    Provide a concise (2-3 paragraph) scientific rationalization for this prediction. 
    Include:
    1. Potential mechanism of action (e.g., pathway inhibition, receptor binding).
    2. Biological plausibility (why this drug might work for this specific pathology).
    3. Any known literature context or related mechanisms if applicable.
    
    Format the response in professional medical prose.
    """
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error generating rationale: {str(e)}"

def main():
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"Error: Could not find {PREDICTIONS_PATH}. Please run Step 4 first.")
        return

    df = pd.read_csv(PREDICTIONS_PATH).head(NUM_CANDIDATES)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# Clinical Rationalization Report: Top {NUM_CANDIDATES} Drug Repurposing Candidates\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model used: {MODEL_NAME}\n\n")
        f.write("---\n\n")
        
        for index, row in df.iterrows():
            drug = row['drug_name']
            disease = row['disease_name']
            score = row['repurposing_score']
            rank = row.get('rank', index + 1)
            
            print(f"Processing Rank {rank}: {drug} for {disease}...")
            
            rationale = generate_rationale(drug, disease)
            
            f.write(f"## Rank {rank}: {drug} → {disease}\n")
            f.write(f"**AI Repurposing Score:** {score}\n\n")
            f.write(f"### Scientific Rationale\n")
            f.write(f"{rationale}\n\n")
            f.write("---\n\n")
            
    print(f"\n--- STEP 5 COMPLETE: Full report saved to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()
