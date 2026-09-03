import requests
import json

def test_no_proxy():
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "llama3.2:latest",
        "prompt": "Say 'OK'",
        "stream": False
    }
    
    print(f"Testing direct POST to {url} (BYPASSING PROXIES)...")
    try:
        # Explicitly disable proxies
        session = requests.Session()
        session.trust_env = False 
        
        response = session.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json().get('response')}")
        else:
            print(f"Error Body: {response.text}")
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    test_no_proxy()
