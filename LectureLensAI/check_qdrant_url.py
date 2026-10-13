
import os
import socket
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("QDRANT_URL", "").strip()
parsed = urlparse(url)

print("URL exists:", bool(url))
print("HTTPS:", parsed.scheme == "https")
print("Hostname exists:", bool(parsed.hostname))

if parsed.hostname:
    print("Hostname:", parsed.hostname)
    try:
        socket.getaddrinfo(parsed.hostname, 443)
        print("DNS: OK")
    except socket.gaierror as e:
        print("DNS FAILED:", e)
else:
    print("Check QDRANT_URL in your .env file")
