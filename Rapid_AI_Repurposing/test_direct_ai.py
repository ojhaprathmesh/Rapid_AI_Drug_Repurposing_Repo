import requests
import json

def test_direct():
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "llama3.2:latest",
        "prompt": "Say 'Connection Successful' in one word.",
        "stream": False
    }
    
    print(f"Testing direct POST to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json().get('response')}")
        else:
            print(f"Error Body: {response.text}")
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    test_direct()
