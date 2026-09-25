"""Discord channel talk-back for this project (replaces models/mailer.py from 2026-09-25).

The bot token lives in Other/.env as Discord_RNSA_BOT and is never printed.
The target channel id is cached in Other/discord_channel.txt.

  python models/discord_bot.py guilds                 list servers + text channels
  python models/discord_bot.py use CHANNEL_ID         remember the channel to post in
  python models/discord_bot.py send message.txt       post a file's text (split at 1,900 chars)
  python models/discord_bot.py say "one line"         post one short line
  python models/discord_bot.py inbox [limit]          show the channel's last messages
  python models/discord_bot.py authors [limit]        recent human authors + their user ids
  python models/discord_bot.py dm USER_ID             open a DM with that user and use it

A DM is just another channel id, so once `dm` has stored it, send / say / inbox all work in the DM
with no further changes. The bot CANNOT discover its own DMs (Discord does not let bots list them) and
cannot list server members without the GUILD_MEMBERS privileged intent, so the user id has to come from
somewhere once: `authors` reads it off any message the person posts in a channel the bot can see.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://discord.com/api/v10"
CHANNEL_FILE = ROOT / "Other/discord_channel.txt"
LIMIT = 1900          # Discord's hard limit is 2,000 characters per message


def token() -> str:
    for env in (ROOT / "Other/.env", ROOT / ".env"):
        if env.exists():
            for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip().startswith("Discord_RNSA_BOT") and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("Discord_RNSA_BOT not found in Other/.env")


def call(path: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Authorization": "Bot " + token(),
                                          "Content-Type": "application/json",
                                          "User-Agent": "DiscordBot (structural-break, 1.0)"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode()
                return json.loads(text) if text.strip() else {}
        except urllib.error.HTTPError as e:
            payload = e.read().decode()[:400]
            if e.code == 429 and attempt < 4:            # rate limited: honour retry_after
                try:
                    time.sleep(float(json.loads(payload).get("retry_after", 2)) + 0.5)
                except Exception:
                    time.sleep(3)
                continue
            raise SystemExit(f"{method} {path} -> HTTP {e.code}: {payload}")
    raise SystemExit("gave up after rate limits")


def channel() -> str:
    if not CHANNEL_FILE.exists():
        raise SystemExit("no channel yet: run `python models/discord_bot.py guilds` then `use CHANNEL_ID`")
    return CHANNEL_FILE.read_text(encoding="utf-8").strip()


def post(text: str) -> None:
    """Post `text`, split on line boundaries so blocks stay readable on a phone."""
    chunks, current = [], ""
    for line in text.splitlines(True):
        while len(line) > LIMIT:                      # a single very long line
            chunks.append(line[:LIMIT]); line = line[LIMIT:]
        if len(current) + len(line) > LIMIT:
            chunks.append(current); current = ""
        current += line
    if current.strip():
        chunks.append(current)
    for i, chunk in enumerate(chunks):
        call(f"/channels/{channel()}/messages", "POST", {"content": chunk})
        print(f"posted chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "guilds"
    if what == "guilds":
        me = call("/users/@me")
        print(f"bot: {me.get('username')} (id {me.get('id')})")
        for g in call("/users/@me/guilds"):
            print(f"server {g['name']} (id {g['id']})")
            try:
                for c in call(f"/guilds/{g['id']}/channels"):
                    if c.get("type") in (0, 5):        # text / announcement
                        print(f"   #{c['name']}  id {c['id']}")
            except SystemExit as e:
                print(f"   (cannot list channels: {e})")
        return
    if what == "use":
        CHANNEL_FILE.write_text(sys.argv[2].strip(), encoding="utf-8")
        info = call(f"/channels/{sys.argv[2].strip()}")
        print(f"will post in #{info.get('name')} (id {info.get('id')})")
        return
    if what == "send":
        post(Path(sys.argv[2]).read_text(encoding="utf-8"))
        return
    if what == "say":
        post(sys.argv[2])
        return
    if what == "authors":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        seen = {}
        for m in call(f"/channels/{channel()}/messages?limit={limit}"):
            a = m["author"]
            if not a.get("bot"):
                seen[a["id"]] = a.get("global_name") or a.get("username")
        if not seen:
            print("no human messages in the last", limit, "- post one line in the channel and retry")
        for uid, name in seen.items():
            print(f"{name}  user id {uid}")
        return
    if what == "dm":
        info = call("/users/@me/channels", "POST", {"recipient_id": sys.argv[2].strip()})
        CHANNEL_FILE.write_text(str(info["id"]), encoding="utf-8")
        print(f"DM channel {info['id']} opened and stored; send/say/inbox now use it")
        return
    if what == "inbox":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        for m in reversed(call(f"/channels/{channel()}/messages?limit={limit}")):
            who = m["author"].get("username")
            bot = " [bot]" if m["author"].get("bot") else ""
            print(f"--- {m['timestamp'][:16]} {who}{bot}\n{(m.get('content') or '').strip()[:800]}")
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
