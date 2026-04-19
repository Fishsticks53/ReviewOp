from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class UserPortalCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("MYSQL_HOST", "localhost")
        os.environ.setdefault("MYSQL_PORT", "3306")
        os.environ.setdefault("MYSQL_USER", "user")
        os.environ.setdefault("MYSQL_PASSWORD", "pass")
        os.environ.setdefault("MYSQL_DB", "reviewop")
        sys.path.insert(0, "backend")

        from core.db import Base
        from models.tables import ProductCatalog, User, UserProductReview
        from routes.user_portal import _recompute_product_cache

        cls.Base = Base
        cls.ProductCatalog = ProductCatalog
        cls.User = User
        cls.UserProductReview = UserProductReview
        cls._recompute_product_cache = _recompute_product_cache

    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        self.Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_recompute_product_cache_matches_current_reviews(self) -> None:
        db = self.SessionLocal()
        try:
            user = self.User(username="alice", password_salt="salt", password_hash="hash", role="user")
            product = self.ProductCatalog(product_id="p1", name="Product 1", category="General", summary="")
            db.add_all([user, product])
            db.flush()
            db.add_all(
                [
                    self.UserProductReview(
                        user_id=user.id,
                        product_id="p1",
                        rating=2,
                        title="bad",
                        review_text="bad",
                        helpful_count=1,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    ),
                    self.UserProductReview(
                        user_id=user.id,
                        product_id="p1",
                        rating=4,
                        title="good",
                        review_text="good",
                        helpful_count=3,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    ),
                ]
            )
            db.flush()

            type(self)._recompute_product_cache(db, product)

            self.assertEqual(product.cached_review_count, 2)
            self.assertAlmostEqual(product.cached_average_rating, 3.0)
            self.assertEqual(product.cached_helpful_count, 4)
        finally:
            db.close()

    def test_recompute_product_cache_ignores_soft_deleted_reviews(self) -> None:
        db = self.SessionLocal()
        try:
            user = self.User(username="bob", password_salt="salt", password_hash="hash", role="user")
            product = self.ProductCatalog(product_id="p2", name="Product 2", category="General", summary="")
            db.add_all([user, product])
            db.flush()
            db.add_all(
                [
                    self.UserProductReview(
                        user_id=user.id,
                        product_id="p2",
                        rating=5,
                        title="ok",
                        review_text="ok",
                        helpful_count=2,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    ),
                    self.UserProductReview(
                        user_id=user.id,
                        product_id="p2",
                        rating=1,
                        title="old",
                        review_text="old",
                        helpful_count=9,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        deleted_at=datetime.utcnow(),
                    ),
                ]
            )
            db.flush()

            type(self)._recompute_product_cache(db, product)

            self.assertEqual(product.cached_review_count, 1)
            self.assertAlmostEqual(product.cached_average_rating, 5.0)
            self.assertEqual(product.cached_helpful_count, 2)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
