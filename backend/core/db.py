# proto/backend/core/db.py
import pymysql

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url, URL
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from core.config import settings


class Base(DeclarativeBase):
    pass


def _pymysql_connect(*, database: str | None = None):
    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=database,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=10,
    )


def ensure_database_exists() -> None:
    """
    Create the configured MySQL database if it does not exist yet.
    Requires valid MySQL host/user/password in settings.
    """
    url = make_url(settings.mysql_url)
    db_name = url.database
    if not db_name:
        raise RuntimeError("mysql_db is not configured")
    safe_db_name = db_name.replace("`", "``")

    # Connect to MySQL server without selecting a specific database.
    admin_engine = create_engine(
        settings.mysql_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
        creator=lambda: _pymysql_connect(),
    )

    with admin_engine.begin() as conn:
        conn.execute(
            text(f"CREATE DATABASE IF NOT EXISTS `{safe_db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        )

    admin_engine.dispose()


engine = create_engine(
    settings.mysql_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
    creator=lambda: _pymysql_connect(database=make_url(settings.mysql_url).database),
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Initializes the database and ensures it exists."""
    ensure_database_exists()
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
