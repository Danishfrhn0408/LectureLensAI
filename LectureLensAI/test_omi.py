import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OMI_API_KEY")

if not api_key:
    raise ValueError("OMI_API_KEY not found in .env")

url = "https://api.omi.me/v1/dev/user/conversations"

response = requests.get(
    url,
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30
)

print("HTTP Status:", response.status_code)

if response.ok:
    data = response.json()
    print("Omi connection successful!")

    if isinstance(data, list):
        print("Conversations found:", len(data))
        for item in data[:5]:
            print({
                "id": item.get("id"),
                "created_at": item.get("created_at")
            })
    else:
        print(data)
else:
    print("Request failed:", response.text[:500])