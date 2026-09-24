"""Read-only probe: can the 2025 Structural Break data (allowed by the FAQ) be listed with our API key?"""
import logging
from pathlib import Path

from crunch.api import Client, auth
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
logging.disable(logging.CRITICAL)
env = dotenv_values(ROOT / ".env")
for label in ("CRUNCH_API_KEY_KRITTANUT", "CRUNCH_API_KEY_YODHARIN"):
    client = Client.from_env(auth=auth.ApiKeyAuth(env[label]), show_progress=False)
    for name in ("structural-break", "structural-break-open-benchmark"):
        try:
            comp = client.competitions.get(name)
            print(label[-9:], name, "->", {k: comp._attrs.get(k) for k in ("id", "name", "status", "format")})
            for attr in ("data_releases", "rounds", "phases"):
                if hasattr(comp, attr):
                    try:
                        items = getattr(comp, attr).list()
                        print("   ", attr, [getattr(i, "_attrs", {}).get("name") or getattr(i, "_attrs", {}).get("number") for i in items][:6])
                    except Exception as error:  # noqa: BLE001
                        print("   ", attr, "error:", str(error)[:120])
        except Exception as error:  # noqa: BLE001
            print(label[-9:], name, "error:", str(error)[:160])
