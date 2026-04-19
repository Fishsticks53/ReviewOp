# proto/backend/services/batch_jobs.py
from __future__ import annotations

from datetime import datetime
import logging
import threading
from typing import Optional, List

import pandas as pd
from sqlalchemy.orm import Session

from core.db import SessionLocal
from models.tables import Job, JobItem, Review, Prediction, EvidenceSpan
from services.evidence import find_evidence_for_aspect
from services.seq2seq_infer import Seq2SeqEngine
from services.open_aspect import extract_open_aspects
from services.review_pipeline import refresh_corpus_graph

logger = logging.getLogger(__name__)


def _error_code(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return "invalid_input"
    if isinstance(exc, UnicodeError):
        return "decode_error"
    return type(exc).__name__


def _safe_extract_aspects(text: str, max_aspects: int = 8) -> list[str]:
    try:
        aspects = extract_open_aspects(text, max_aspects=max_aspects)
        if aspects:
            return aspects
    except Exception:
        pass
    return ["general"]


def detect_review_column(df: pd.DataFrame) -> str:
    candidates = ["reviews", "review", "text", "content", "sentence", "comment"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return df.columns[0]


def process_csv_sync(
    db: Session,
    engine: Seq2SeqEngine,
    job: Job,
    df: pd.DataFrame,
    domain: Optional[str] = None,
    product_id: Optional[str] = None,
) -> None:
    col = detect_review_column(df)
    total = int(len(df.index))

    job.total = total
    job.status = "running"
    job.processed = 0
    job.failed = 0
    job.updated_at = datetime.utcnow()
    db.add(job)
    db.commit()

    items: List[JobItem] = [JobItem(job_id=job.id, row_index=i, status="queued") for i in range(total)]
    db.add_all(items)
    db.commit()

    for i in range(total):
        item = db.query(JobItem).filter(JobItem.job_id == job.id, JobItem.row_index == i).first()
        try:
            text = str(df.iloc[i][col])
            if not text or text.strip().lower() in {"nan", "none"}:
                raise ValueError("Empty review text")

            r = Review(text=text, domain=domain, product_id=product_id)
            db.add(r)
            db.flush()

            # Phase 2 behavior: open-aspect extraction + per-aspect sentiment on evidence sentence
            aspects = _safe_extract_aspects(text, max_aspects=8)

            for aspect_raw in aspects:
                s, e, snippet = find_evidence_for_aspect(text, aspect_raw)
                sent, conf = engine.classify_sentiment_with_confidence(snippet, aspect_raw)

                pred = Prediction(
                    aspect_raw=aspect_raw,
                    aspect_cluster=aspect_raw,
                    sentiment=sent,
                    confidence=float(conf),
                    rationale=None,
                )
                pred.review = r
                pred.evidence_spans.append(
                    EvidenceSpan(
                        start_char=s,
                        end_char=e,
                        snippet=snippet,
                    )
                )
                db.add(pred)

            item.review_id = r.id
            item.status = "done"
            item.error = None

            job.processed += 1
            job.updated_at = datetime.utcnow()
            db.add(item)
            db.add(job)
            db.commit()

        except Exception as ex:
            db.rollback()
            item = db.query(JobItem).filter(JobItem.job_id == job.id, JobItem.row_index == i).first()
            job = db.query(Job).filter(Job.id == job.id).first() or job
            item.status = "failed"
            item.error = _error_code(ex)
            job.failed += 1
            job.updated_at = datetime.utcnow()
            db.add(item)
            db.add(job)
            db.commit()

    job.status = "done"
    job.updated_at = datetime.utcnow()
    try:
        refresh_corpus_graph(db, domain=domain)
    except Exception as ex:
        job.error = f"{job.error + ' | ' if job.error else ''}corpus_graph_refresh_failed"
    db.add(job)
    db.commit()


def schedule_csv_processing(
    *,
    engine: Seq2SeqEngine,
    job_id: str,
    df: pd.DataFrame,
    domain: Optional[str] = None,
    product_id: Optional[str] = None,
) -> threading.Thread:
    def _worker() -> None:
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                return
            process_csv_sync(db=db, engine=engine, job=job, df=df, domain=domain, product_id=product_id)
        except Exception as exc:
            logger.exception("CSV job worker crashed (job_id=%s)", job_id)
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = "failed"
                    job.error = _error_code(exc)
                    job.updated_at = datetime.utcnow()
                    db.add(job)
                    db.commit()
            except Exception:
                db.rollback()
        finally:
            db.close()

    thread = threading.Thread(target=_worker, name=f"csv-job-{job_id}", daemon=True)
    thread.start()
    return thread
