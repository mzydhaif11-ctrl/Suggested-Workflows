"""Local application entry point."""

import os

import uvicorn
from dotenv import load_dotenv


def main():
    load_dotenv()
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
