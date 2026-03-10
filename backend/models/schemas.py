# proto/backend/models/schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field


class EvidenceSpanOut(BaseModel):
    start_char: int
    end_char: int
    snippet: str


class PredictionOut(BaseModel):
    aspect_raw: str
    aspect_cluster: str
    sentiment: str  # positive|neutral|negative
    confidence: float

    # NEW:
    aspect_weight: Optional[float] = None
    aspect_score: Optional[float] = None

    evidence_spans: List[EvidenceSpanOut] = Field(default_factory=list)
    rationale: Optional[str] = None


class InferReviewIn(BaseModel):
    text: str
    domain: Optional[str] = None
    product_id: Optional[str] = None


class InferReviewOut(BaseModel):
    review_id: int
    domain: Optional[str] = None
    product_id: Optional[str] = None
    predictions: List[PredictionOut]

    # NEW:
    overall_sentiment: Optional[str] = None
    overall_score: Optional[float] = None
    overall_confidence: Optional[float] = None


class JobCreateOut(BaseModel):
    job_id: str
    status: str
    total: int


class JobStatusOut(BaseModel):
    job_id: str
    status: str
    total: int
    processed: int
    failed: int
    error: Optional[str] = None


class OverviewOut(BaseModel):
    total_reviews: int
    total_aspect_mentions: int
    unique_aspects_raw: int
    avg_confidence: float
    sentiment_counts: dict


class TopAspectOut(BaseModel):
    aspect: str
    count: int


class AspectSentimentDistOut(BaseModel):
    aspect: str
    positive: int
    neutral: int
    negative: int


class TrendPointOut(BaseModel):
    bucket: str
    mentions: int
    negative_pct: float
    sentiment_score: float


# Optional: KG payloads (for next step endpoints)
class AspectNodeOut(BaseModel):
    aspect_cluster: str
    domain: Optional[str] = None
    df: Optional[int] = None
    idf: Optional[float] = None
    centrality: Optional[float] = None


class AspectEdgeOut(BaseModel):
    src_aspect: str
    dst_aspect: str
    edge_type: str
    weight: float
    domain: Optional[str] = None

class CentralityOut(BaseModel):
    aspect: str
    centrality: float
    df: int = 0
    idf: float = 0.0


class CommunityOut(BaseModel):
    community_id: int
    aspects: List[str]


class EdgeOut(BaseModel):
    src: str
    dst: str
    edge_type: str
    weight: float


class GraphNodeOut(BaseModel):
    id: str
    label: str
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    frequency: Optional[int] = None
    avg_sentiment: Optional[float] = None
    dominant_sentiment: Optional[str] = None
    negative_ratio: Optional[float] = None
    explicit_count: int = 0
    implicit_count: int = 0
    evidence: Optional[str] = None
    evidence_start: Optional[int] = None
    evidence_end: Optional[int] = None
    origin: Optional[str] = None


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    type: str
    weight: float = 1.0
    directional: bool = False
    pair_count: Optional[int] = None
    polarity_hint: Optional[str] = None


class GraphResponseOut(BaseModel):
    scope: str
    review_id: Optional[int] = None
    generated_at: Optional[str] = None
    filters: dict = Field(default_factory=dict)
    nodes: List[GraphNodeOut] = Field(default_factory=list)
    edges: List[GraphEdgeOut] = Field(default_factory=list)
