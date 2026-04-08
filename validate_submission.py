import requests
import os

def validate():
    base_url = os.getenv(
        "ENV_BASE_URL",
        "https://ramcharan2905-hospital-resource-env.hf.space"
    )

    print(f"--- Validating OpenEnv Server at {base_url} ---")

    # HEALTH
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        print("[PASS]" if r.status_code == 200 else "[FAIL]", "/health")
    except Exception as e:
        print("[ERROR] /health:", e)
        return

    # RESET
    try:
        r = requests.post(
            f"{base_url}/reset",
            json={"task_id": "easy", "seed": 42},
            timeout=10
        )
        print("[PASS]" if r.status_code == 200 else "[FAIL]", "/reset")
    except Exception as e:
        print("[ERROR] /reset:", e)

    # STEP
    try:
        r = requests.post(
            f"{base_url}/step",
            json={"action": None},
            timeout=10
        )
        print("[PASS]" if r.status_code == 200 else "[FAIL]", "/step")
    except Exception as e:
        print("[ERROR] /step:", e)

    print("--- Done ---")


if __name__ == "__main__":
    validate()