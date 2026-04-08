import os
import requests


def validate() -> None:
    base_url = os.getenv(
        "ENV_BASE_URL",
        "https://ramcharan2905-hospital-resource-env.hf.space",
    ).rstrip("/")

    print(f"--- Validating OpenEnv Server at {base_url} ---")

    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        print("[PASS] /health" if r.status_code == 200 else f"[FAIL] /health ({r.status_code})")
    except Exception as exc:
        print(f"[ERROR] /health: {exc}")
        return

    try:
        r = requests.post(
            f"{base_url}/reset",
            json={"task_id": "easy", "seed": 42},
            timeout=10,
        )
        print("[PASS] /reset" if r.status_code == 200 else f"[FAIL] /reset ({r.status_code})")
    except Exception as exc:
        print(f"[ERROR] /reset: {exc}")

    try:
        r = requests.post(
            f"{base_url}/step",
            json={"action": None},
            timeout=10,
        )
        print("[PASS] /step" if r.status_code == 200 else f"[FAIL] /step ({r.status_code})")
    except Exception as exc:
        print(f"[ERROR] /step: {exc}")

    try:
        r = requests.get(f"{base_url}/state", timeout=10)
        print("[PASS] /state" if r.status_code == 200 else f"[FAIL] /state ({r.status_code})")
    except Exception as exc:
        print(f"[ERROR] /state: {exc}")

    print("--- Done ---")


if __name__ == "__main__":
    validate()