import requests
import time
import sys

def validate():
    base_url = "http://localhost:8000"
    print(f"--- Validating OpenEnv Server at {base_url} ---")
    
    # 1. Health Check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("[PASS] /health endpoint is live.")
        else:
            print(f"[FAIL] /health returned status {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Could not connect to /health: {e}")
        print("\nMake sure your server is running (e.g., uvicorn server:app --port 8000)")
        return

    # 2. Reset Check
    try:
        payload = {"task_id": "easy", "seed": 42}
        response = requests.post(f"{base_url}/reset", json=payload, timeout=5)
        if response.status_code == 200:
            obs = response.json()
            print("[PASS] /reset endpoint successful.")
            print(f"       Initial Observation keys: {list(obs.keys())}")
        else:
            print(f"[FAIL] /reset returned status {response.status_code}")
    except Exception as e:
        print(f"[ERROR] /reset failed: {e}")

    # 3. Step Check
    try:
        # Sending a no-op step
        payload = {"action": None}
        response = requests.post(f"{base_url}/step", json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("[PASS] /step (no-op) successful.")
            print(f"       Reward: {data.get('reward')}, Done: {data.get('done')}")
        else:
            print(f"[FAIL] /step returned status {response.status_code}")
    except Exception as e:
        print(f"[ERROR] /step failed: {e}")

    print("\n--- Validation Complete ---")

if __name__ == "__main__":
    validate()
