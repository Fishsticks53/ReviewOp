from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from services.analytics_alerts import alerts
from services.analytics_aspects import aspect_leaderboard, aspect_trends, emerging_aspects, evidence_drilldown
from services.analytics_kpis import dashboard_kpis
from services.analytics_segments import impact_matrix, segment_drilldown, weekly_summary
from services.analytics_user_reviews import user_reviews_list, user_reviews_summary
from datetime import datetime


def export_payload(db: Session, domain: Optional[str] = None, limit: int = 100, offset: int = 0) -> dict:
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "dashboard_kpis": dashboard_kpis(db, None, None, domain),
        "aspect_leaderboard": aspect_leaderboard(db, limit=25, domain=domain),
        "aspect_trends": aspect_trends(db, interval="day", domain=domain, limit=500),
        "emerging_aspects": emerging_aspects(db, interval="day", lookback_buckets=7, domain=domain),
        "evidence": evidence_drilldown(db, aspect=None, sentiment=None, limit=50, domain=domain),
        "alerts": alerts(db, domain=domain),
        "impact_matrix": impact_matrix(db, domain=domain, limit=20),
        "segments": segment_drilldown(db, domain=domain, limit=20),
        "weekly_summary": weekly_summary(db, domain=domain),
        "user_reviews_summary": user_reviews_summary(db, domain=domain),
        "user_reviews": user_reviews_list(db, domain=domain, limit=limit, offset=offset),
    }


def export_pdf_bytes(db: Session, domain: Optional[str] = None, limit: int = 100, offset: int = 0) -> bytes:
    payload = export_payload(db, domain=domain, limit=limit, offset=offset)
    lines = ["ReviewOp Admin Export Report", f"Generated At: {payload['generated_at']}", f"Domain Filter: {domain or 'all'}", "", "Dashboard KPIs"]
    kpis = payload["dashboard_kpis"]
    lines.extend([f"  Total Reviews: {kpis['total_reviews']}", f"  Total Aspects: {kpis['total_aspects']}", f"  Most Negative Aspect: {kpis.get('most_negative_aspect') or '-'}", f"  Negative Sentiment %: {kpis['negative_sentiment_pct']}", f"  Emerging Issues: {kpis['emerging_issues_count']}", "", "User Reviews Summary"])
    summary = payload["user_reviews_summary"]
    lines.extend([f"  Total User Reviews: {summary['total_user_reviews']}", f"  Unique Reviewers: {summary['unique_reviewers']}", f"  Average Rating: {summary['average_rating']}", f"  Recommendation Rate %: {summary['recommendation_rate']}", f"  Reviews Last 7 Days: {summary['reviews_last_7_days']}", "", "Top Alerts"])
    for item in payload["alerts"][:12]:
        lines.append(f"  [{item['severity']}] {item['aspect']}: {item['message']}")
    lines.append("")
    lines.append("Top User Reviews (latest)")
    for row in payload["user_reviews"]["rows"][:12]:
        lines.append(f"  {row['created_at'][:10]} | {row['username']} | {row['product_id']} | rating={row['rating']}")
    content_lines = []
    y = 800
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_lines.append(f"BT /F1 10 Tf 50 {y} Td ({escaped[:150]}) Tj ET")
        y -= 14
        if y < 40:
            break
    stream_data = "\n".join(content_lines).encode("latin-1", errors="ignore")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>", b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >>", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", f"<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1") + stream_data + b"\nendstream"]
    out = bytearray(b"%PDF-1.4\n")
    xref_positions = [0]
    for idx, obj in enumerate(objects, start=1):
        xref_positions.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("latin-1"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_start = len(out)
    out.extend(f"xref\n0 {len(xref_positions)}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for pos in xref_positions[1:]:
        out.extend(f"{pos:010d} 00000 n \n".encode("latin-1"))
    out.extend(f"trailer\n<< /Size {len(xref_positions)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode("latin-1"))
    return bytes(out)
