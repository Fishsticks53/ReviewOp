from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import case, func, text
from sqlalchemy.orm import Session

from models.tables import ProductCatalog, User, UserProductReview


def user_reviews_summary(db: Session, domain: Optional[str] = None) -> dict:
    q = db.query(UserProductReview).filter(UserProductReview.deleted_at.is_(None))
    if domain:
        q = q.join(ProductCatalog, ProductCatalog.product_id == UserProductReview.product_id).filter(ProductCatalog.category == domain)
    total_reviews = int(q.count())
    if total_reviews == 0:
        return {"total_user_reviews": 0, "unique_reviewers": 0, "average_rating": 0.0, "recommendation_rate": 0.0, "reviews_last_7_days": 0, "top_products": []}
    unique_reviewers = int(q.with_entities(func.count(func.distinct(UserProductReview.user_id))).scalar() or 0)
    average_rating = float(q.with_entities(func.avg(UserProductReview.rating)).scalar() or 0.0)
    rec_yes = int(q.with_entities(func.sum(case((UserProductReview.recommendation.is_(True), 1), else_=0))).scalar() or 0)
    rec_total = int(q.with_entities(func.sum(case((UserProductReview.recommendation.isnot(None), 1), else_=0))).scalar() or 0)
    recommendation_rate = (rec_yes / rec_total) * 100 if rec_total else 0.0
    last_7_days = datetime.utcnow() - timedelta(days=7)
    reviews_last_7_days = int(q.filter(UserProductReview.created_at >= last_7_days).count())
    top_q = db.query(UserProductReview.product_id.label("product_id"), func.count(UserProductReview.id).label("count"), func.avg(UserProductReview.rating).label("avg_rating"), ProductCatalog.name.label("name")).outerjoin(ProductCatalog, ProductCatalog.product_id == UserProductReview.product_id).filter(UserProductReview.deleted_at.is_(None))
    if domain:
        top_q = top_q.filter(ProductCatalog.category == domain)
    top_rows = top_q.group_by(UserProductReview.product_id, ProductCatalog.name).order_by(text("count DESC"), UserProductReview.product_id.asc()).limit(5).all()
    top_products = [{"product_id": row.product_id, "product_name": row.name or f"Product {row.product_id}", "review_count": int(row.count or 0), "average_rating": round(float(row.avg_rating or 0.0), 2)} for row in top_rows]
    return {"total_user_reviews": total_reviews, "unique_reviewers": unique_reviewers, "average_rating": round(average_rating, 2), "recommendation_rate": round(recommendation_rate, 2), "reviews_last_7_days": reviews_last_7_days, "top_products": top_products}


def user_reviews_list(db: Session, domain: Optional[str] = None, product_id: Optional[str] = None, username: Optional[str] = None, min_rating: Optional[int] = None, max_rating: Optional[int] = None, limit: int = 50, offset: int = 0) -> dict:
    q = db.query(UserProductReview.id.label("review_id"), UserProductReview.product_id.label("product_id"), UserProductReview.rating.label("rating"), UserProductReview.recommendation.label("recommendation"), UserProductReview.helpful_count.label("helpful_count"), UserProductReview.title.label("review_title"), UserProductReview.review_text.label("review_text"), UserProductReview.created_at.label("created_at"), User.username.label("username"), ProductCatalog.name.label("product_name"), ProductCatalog.category.label("product_category")).join(User, User.id == UserProductReview.user_id).outerjoin(ProductCatalog, ProductCatalog.product_id == UserProductReview.product_id).filter(UserProductReview.deleted_at.is_(None))
    if domain:
        q = q.filter(ProductCatalog.category == domain)
    if product_id:
        q = q.filter(UserProductReview.product_id == product_id)
    if username:
        q = q.filter(User.username.ilike(f"%{username.strip()}%"))
    if min_rating is not None:
        q = q.filter(UserProductReview.rating >= min_rating)
    if max_rating is not None:
        q = q.filter(UserProductReview.rating <= max_rating)
    total = int(q.count())
    rows = q.order_by(UserProductReview.created_at.desc(), UserProductReview.id.desc()).offset(max(0, offset)).limit(max(1, min(limit, 200))).all()
    return {"total": total, "limit": max(1, min(limit, 200)), "offset": max(0, offset), "rows": [{"review_id": row.review_id, "product_id": row.product_id, "product_name": row.product_name, "username": row.username, "rating": int(row.rating), "recommendation": row.recommendation, "helpful_count": int(row.helpful_count or 0), "review_title": row.review_title, "review_text": row.review_text, "created_at": row.created_at.isoformat()} for row in rows]}
