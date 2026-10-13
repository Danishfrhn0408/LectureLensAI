import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

key = os.getenv("LYZR_API_KEY", "").strip()

response = requests.get(
    "https://agent-prod.studio.lyzr.ai/v3/agents/",
    headers={"x-api-key": key},
    timeout=30,
)

print("HTTP Status:", response.status_code)
print("Response:", response.text[:500])