from __future__ import annotations

import getpass
import hashlib
import secrets

from core.db import SessionLocal
from models.tables import User


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()


def main() -> None:
    username = input("Admin username: ").strip()
    if len(username) < 3:
        raise SystemExit("username must be at least 3 characters")
    password = getpass.getpass("Admin password: ")
    if len(password) < 6:
        raise SystemExit("password must be at least 6 characters")

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            raise SystemExit("username already exists")
        salt = secrets.token_hex(16)
        user = User(
            username=username,
            password_salt=salt,
            password_hash=_hash_password(password, salt),
            role="admin",
        )
        db.add(user)
        db.commit()
        print(f"Admin created: {username}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
