# server/app.py

import uvicorn
from server import app  # your existing FastAPI app


def main():
    """Entry point for OpenEnv validator"""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()