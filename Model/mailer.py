"""Minimal mail helper: send a status email / read replies (credentials from email.txt).

  python models/mailer.py send "Subject" body.txt
  python models/mailer.py inbox [since_uid]
"""
from __future__ import annotations

import email
import email.utils
from email.header import decode_header, make_header
from email.message import EmailMessage
import imaplib
from pathlib import Path
import smtplib
import ssl
import sys

ROOT = Path(__file__).resolve().parents[1]
TO = "yodharinmonplub@actuarialtutor.org"   # AGENTS.md 2026-09-20 (was the gmail address)
OLD_TO = "yodharinmonplub@gmail.com"        # replies from the old address are still read


def credentials():
    fields = {}
    for line in (ROOT / "email.txt").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return fields["username"], fields["password"]


def send(subject: str, body: str) -> None:
    user, password = credentials()
    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = user, TO, subject
    message["Date"] = email.utils.formatdate(localtime=True)
    message.set_content(body)
    with smtplib.SMTP_SSL("smtp.purelymail.com", 465, context=ssl.create_default_context()) as smtp:
        smtp.login(user, password)
        smtp.send_message(message)
    print("sent:", subject)


def inbox(limit: int = 10, after_uid: int = 0) -> None:
    """Print the user's last `limit` messages; with after_uid, only those with a larger id."""
    user, password = credentials()
    with imaplib.IMAP4_SSL("imap.purelymail.com", 993) as imap:
        imap.login(user, password)
        imap.select("INBOX", readonly=True)
        _, data = imap.search(None, f'(OR FROM "{TO}" FROM "{OLD_TO}")')
        ids = [i for i in data[0].split() if int(i) > after_uid][-limit:]
        if not ids:
            print(f"no new mail after #{after_uid}")
        for i in ids:
            _, msg_data = imap.fetch(i, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            sender = str(make_header(decode_header(msg.get("From", ""))))
            subject = str(make_header(decode_header(msg.get("Subject", ""))))
            body = ""
            part = next((p for p in msg.walk() if p.get_content_type() == "text/plain"), None)
            if part is not None:
                body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
            print(f"--- #{i.decode()} {msg.get('Date')} | {sender} | {subject}")
            print(body.strip()[:800])


if __name__ == "__main__":
    if sys.argv[1] == "send":
        send(sys.argv[2], Path(sys.argv[3]).read_text(encoding="utf-8"))
    else:
        inbox(int(sys.argv[2]) if len(sys.argv) > 2 else 10, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
