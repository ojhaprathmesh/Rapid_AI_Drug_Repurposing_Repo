import ollama
import json

def check():
    print("--- Ollama Diagnostic ---")
    try:
        models = ollama.list()
        print("Available Models:")
        found_llama = False
        for m in models.get('models', []):
            name = m.get('name')
            print(f" - {name}")
            if 'llama3.2' in name:
                found_llama = True
        
        if not found_llama:
            print("\nWARNING: 'llama3.2' not found in your Ollama library!")
            return

        print("\nTesting simple generation...")
        resp = ollama.generate(model='llama3.2', prompt='Quick test: say hi.', stream=False)
        print(f"Success! Response: {resp['response']}")

    except Exception as e:
        print(f"\nERROR: Could not connect to Ollama API. {e}")
        print("Ensure the Ollama application is open and running in your taskbar.")

if __name__ == "__main__":
    check()
