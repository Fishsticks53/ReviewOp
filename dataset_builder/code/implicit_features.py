from __future__ import annotations

from collections import Counter
import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from mappings import (
    CONTRASTIVE_CONJUNCTIONS,
    GENERIC_ASPECT_STOPWORDS,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    TEXT_STOPWORDS,
)
from llm_utils import build_llm_client
from utils import normalize_whitespace, split_sentences, token_count, tokenize


NEGATION_WORDS = {"not", "never", "no", "without", "hardly", "barely", "scarcely"}
INTENSIFIERS = {"very", "so", "too", "extremely", "really", "quite", "highly", "super", "incredibly", "especially"}
DIMINISHERS = {"barely", "hardly", "slightly", "somewhat", "little", "mildly"}
POSITIVE_SENTIMENT_WORDS = set(POSITIVE_WORDS) | {
    "excellent", "great", "amazing", "wonderful", "love", "loved", "fantastic", "perfect", "solid", "smooth", "helpful",
    "stunning", "impressive", "quality", "reliable", "superb", "brilliant", "delighted", "satisfied", "pleased", "outstanding",
    "friendly", "efficient", "recommend", "best", "top-notch", "worth", "enjoyed", "refreshing", "tasty", "delicious",
}
NEGATIVE_SENTIMENT_WORDS = set(NEGATIVE_WORDS) | {
    "awful", "horrible", "poor", "annoying", "dirty", "broken", "late", "slow", "unusable", "badly", "issues", "problem", "problems",
    "disappointed", "worst", "terrible", "waste", "useless", "broken", "expensive", "rude", "frustrating", "faulty", "failed",
    "crash", "defect", "laggy", "slow", "clunky", "unhelpful", "noisy", "overpriced", "dirty", "old", "tiny", "small",
}


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy
    except Exception:
        return None

    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model_name)
        except Exception:
            continue
    try:
        return spacy.blank("en")
    except Exception:
        return None


def _normalize_aspect(text: str) -> str:
    text = normalize_whitespace(text).lower()
    text = re.sub(r"[^a-z0-9\s_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonicalize_aspect(aspect: str, *, seed_vocab: set[str]) -> str:
    aspect = _normalize_aspect(aspect)
    tokens = aspect.split()
    if not tokens:
        return aspect
    if aspect in seed_vocab:
        return aspect
    for token in tokens:
        if token in seed_vocab:
            return token
    return aspect


def _is_valid_aspect(aspect: str, *, seed_vocab: set[str] | None = None) -> bool:
    return _aspect_rejection_reason(aspect, seed_vocab=seed_vocab) is None


def _aspect_rejection_reason(aspect: str, *, seed_vocab: set[str] | None = None) -> str | None:
    if not aspect or len(aspect) < 3:
        return "too_short"
    if aspect in GENERIC_ASPECT_STOPWORDS:
        return "generic_stopword"
    if aspect in TEXT_STOPWORDS:
        return "text_stopword"
    tokens = aspect.split()
    if not tokens:
        return "empty"
    seeds = seed_vocab if seed_vocab is not None else set()
    if any(token in GENERIC_ASPECT_STOPWORDS or token in TEXT_STOPWORDS for token in tokens):
        return "contains_stopword_token"
    if all(token in TEXT_STOPWORDS for token in tokens):
        return "all_stopwords"
    if len(tokens) == 1:
        # Strict for one-word aspects: must NOT be polar words and should ideally 
        # relate to the domain (if seed_vocab is provided)
        if aspect in POSITIVE_WORDS or aspect in NEGATIVE_WORDS:
            return "polar_word"
        # If we have a seed vocab, one-word candidates not in it are suspicious 
        # unless they are very frequent (handled by discovery ranking later)
        return None
    if len(tokens) > 3:
        return "too_long"
    return None


def _aspect_score_token_set(text: str) -> set[str]:
    return set(tokenize(text))


def _lemmatize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def infer_sentiment(text: str) -> str:
    tokens = tokenize(text)
    pos, neg, _, _ = _sentiment_counts(tokens)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _sentiment_counts(tokens: List[str]) -> tuple[float, float, int, int]:
    pos = 0.0
    neg = 0.0
    negation_hits = 0
    sentiment_hits = 0
    for idx, token in enumerate(tokens):
        base = 0.0
        if token in POSITIVE_SENTIMENT_WORDS:
            base = 1.0
            sentiment_hits += 1
        elif token in NEGATIVE_SENTIMENT_WORDS:
            base = -1.0
            sentiment_hits += 1
        if base == 0.0:
            continue
        window = tokens[max(0, idx - 3):idx]
        negation_count = sum(1 for prev in window if prev in NEGATION_WORDS)
        if negation_count % 2 != 0:
            negation_hits += 1
            base *= -1.0
        if any(prev in INTENSIFIERS for prev in window):
            base *= 1.25
        if any(prev in DIMINISHERS for prev in window):
            base *= 0.75
        if base > 0:
            pos += base
        else:
            neg += abs(base)
    return pos, neg, negation_hits, sentiment_hits


def _extract_phrases_with_spacy(text: str) -> List[str]:
    nlp = _load_spacy()
    if nlp is None:
        return []
    doc = nlp(text)
    phrases: List[str] = []
    try:
        for chunk in getattr(doc, "noun_chunks", []):
            phrase = _normalize_aspect(chunk.text)
            if phrase:
                phrases.append(phrase)
    except Exception:
        pass
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            lemma = _normalize_aspect(token.lemma_ or token.text)
            if lemma:
                phrases.append(lemma)
    return phrases


def _extract_phrases_fallback(text: str, *, seed_vocab: set[str]) -> List[str]:
    phrases: List[str] = []
    tokens = tokenize(text)
    for idx, token in enumerate(tokens):
        if token in TEXT_STOPWORDS or token in GENERIC_ASPECT_STOPWORDS:
            continue
        if len(token) < 3:
            continue
        if token in POSITIVE_WORDS or token in NEGATIVE_WORDS:
            continue
        lemma = _lemmatize_token(token)
        if not seed_vocab or lemma in seed_vocab:
            phrases.append(lemma)
        elif not seed_vocab:
            phrases.append(lemma)
        if idx + 1 < len(tokens):
            nxt = tokens[idx + 1]
            if nxt not in TEXT_STOPWORDS and nxt not in GENERIC_ASPECT_STOPWORDS and nxt not in POSITIVE_WORDS and nxt not in NEGATIVE_WORDS:
                phrase = f"{token} {nxt}"
                phrases.append(_normalize_aspect(phrase))
    return phrases


def _extract_candidate_phrases(text: str, *, seed_vocab: set[str]) -> List[str]:
    phrases = _extract_phrases_with_spacy(text) or _extract_phrases_fallback(text, seed_vocab=seed_vocab)
    cleaned: List[str] = []
    for phrase in phrases:
        phrase = _canonicalize_aspect(phrase, seed_vocab=seed_vocab)
        if not _is_valid_aspect(phrase, seed_vocab=seed_vocab):
            continue
        cleaned.append(phrase)
    return cleaned


def _candidate_priority(phrase: str, *, doc_count: int, total_docs: int, seed_vocab: set[str], sentiment_context_boost: float = 0.0) -> float:
    tokens = phrase.split()
    if not tokens:
        return 0.0
    seed_hits = sum(1 for token in tokens if token in seed_vocab)
    support = doc_count / max(1, total_docs)
    exact_bonus = 1.5 if phrase in seed_vocab else 0.0
    length_bonus = 0.35 if len(tokens) == 1 else 0.6 if len(tokens) == 2 else 0.3
    support_bonus = min(1.5, support * 6.0)
    seed_bonus = seed_hits * 1.8
    return exact_bonus + length_bonus + support_bonus + seed_bonus + sentiment_context_boost




def learn_aspect_seed_vocab(
    train_rows: List[Dict[str, Any]],
    *,
    text_column: str,
    vocab_size: int,
    seed_vocab: set[str] | None = None,
) -> Dict[str, Any]:
    learned: List[str] = discover_aspects(train_rows, text_column=text_column, vocab_size=vocab_size, seed_vocab=seed_vocab)
    support = Counter()
    purity = Counter()
    total_docs = 0
    for row in train_rows:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text:
            continue
        total_docs += 1
        tokens = set(tokenize(text))
        for aspect in learned:
            if aspect in tokens:
                support[aspect] += 1
                purity[aspect] += 1
    filtered = [aspect for aspect in learned if support[aspect] >= 2]
    if not filtered:
        filtered = learned[: max(1, min(vocab_size, len(learned)))]
    return {
        "learned_seed_vocab": filtered,
        "learned_seed_support": {aspect: support[aspect] for aspect in filtered},
        "learned_seed_total_docs": total_docs,
    }


def discover_aspects(
    train_rows: List[Dict[str, Any]],
    *,
    text_column: str,
    vocab_size: int,
    seed_vocab: set[str] | None = None,
) -> List[str]:
    seed_vocab = set(seed_vocab or set())
    doc_count: Counter[str] = Counter()
    term_score: Counter[str] = Counter()
    total_docs = 0

    for row in train_rows:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text:
            continue
        total_docs += 1
        phrases = _extract_candidate_phrases(text, seed_vocab=seed_vocab)
        if not phrases:
            continue
        seen = set()
        for phrase in phrases:
            if phrase not in seen:
                doc_count[phrase] += 1
                seen.add(phrase)
        for phrase in phrases:
            tf = phrases.count(phrase)
            idf = math.log((1 + total_docs) / (1 + doc_count[phrase])) + 1.0
            term_score[phrase] += tf * idf

    # Calculate semantic proximity boost
    sentiment_context: Counter[str] = Counter()
    for row in train_rows:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text: continue
        tokens = tokenize(text)
        phrases = _extract_candidate_phrases(text, seed_vocab=seed_vocab)
        for phrase in phrases:
            phrase_tokens = set(phrase.lower().split())
            for i, token in enumerate(tokens):
                if token.lower() in phrase_tokens:
                    window = tokens[max(0, i-3):min(len(tokens), i+4)]
                    if any(w.lower() in POSITIVE_SENTIMENT_WORDS or w.lower() in NEGATIVE_SENTIMENT_WORDS for w in window):
                        sentiment_context[phrase] += 1
                        break

    boosted = Counter()
    for phrase, score in term_score.items():
        context_ratio = sentiment_context[phrase] / max(1, doc_count[phrase])
        sentiment_boost = min(2.5, context_ratio * 4.0)
        
        priority = _candidate_priority(
            phrase, 
            doc_count=doc_count.get(phrase, 0), 
            total_docs=total_docs, 
            seed_vocab=seed_vocab,
            sentiment_context_boost=sentiment_boost
        )
        boosted[phrase] = score + priority
    
    for aspect in seed_vocab:
        boosted[aspect] += 4
    ranked = [
        phrase
        for phrase, _ in boosted.most_common()
        if _is_valid_aspect(phrase, seed_vocab=seed_vocab) and (phrase in seed_vocab or doc_count[phrase] >= 2)
    ]

    canonicalized: List[str] = []
    seen = set()
    for phrase in ranked:
        root = _canonicalize_aspect(phrase, seed_vocab=seed_vocab)
        if not root or root in seen:
            continue
        if len(root.split()) > 1:
            continue
        seen.add(root)
        canonicalized.append(root)
        if len(canonicalized) >= vocab_size:
            break

    if not canonicalized:
        canonicalized = [aspect for aspect in list(seed_vocab)[:vocab_size]]
    return canonicalized


def _split_contrastive(sentence: str) -> List[str]:
    """Split a sentence into clauses based on contrastive conjunctions or strong punctuation."""
    lowered = sentence.lower()
    
    # Priority conjunctions for splitting
    delimiters = [rf"\s+{re.escape(conj)}\s+" for conj in CONTRASTIVE_CONJUNCTIONS]
    # Add punctuation and other clausal markers
    delimiters.append(r"\s*;\s*")
    delimiters.append(r",\s+but\s+")
    delimiters.append(r",\s+although\s+")
    delimiters.append(r",\s+while\s+")
    
    combined_pattern = "|".join(delimiters)
    parts = [part.strip(" ,;:-") for part in re.split(combined_pattern, sentence, flags=re.IGNORECASE)]
    
    # Filter out empty or too-short fragments (often noise)
    valid_parts = [part for part in parts if token_count(part) >= 3]
    return valid_parts if valid_parts else [sentence]


def _aspect_lexical_score(text: str, aspect: str, *, seed_vocab: set[str], symptom_map: dict[str, str]) -> float:
    if not _is_valid_aspect(aspect, seed_vocab=seed_vocab):
        return 0.0
    tokens = tokenize(text.lower())
    aspect_tokens = set(aspect.lower().split())
    score = 0.0
    
    # Hard Separation: If the aspect word itself is present, it's EXPLICIT, not implicit.
    # We penalize this heavily to ensure the implicit model learns from indirect signals.
    is_present = aspect.lower() in tokens
    if is_present:
        return -10.0 # Strict penalty for explicit mentions
        
    # Symptom Mapping: Reward indirect signals
    for symptom, target_aspect in symptom_map.items():
        if target_aspect == aspect and symptom in tokens:
            score += 5.0
            
    # Overlap with aspect components (e.g. "battery life" -> "life")
    # also penalized if it's the core aspect word
    token_set = set(tokens)
    overlap = len(token_set.intersection(aspect_tokens))
    if overlap > 0:
        return -5.0 # Partial overlap also suggests explicit mention
        
    # Seed bonus if we have symptom evidence
    if score > 0 and seed_vocab and aspect in seed_vocab:
        score += 2.0
        
    return max(0.0, score)


def _aspect_window_support(clause_tokens: List[str], aspect: str) -> float:
    aspect_tokens = aspect.split()
    if not aspect_tokens:
        return 0.0
    joined = " ".join(clause_tokens)
    if aspect in joined:
        return 1.5
    token_set = set(clause_tokens)
    hits = len(token_set.intersection(aspect_tokens))
    if hits == 0:
        return 0.0
    return 0.8 + 0.4 * hits


def _sentiment_evidence(clause_tokens: List[str]) -> tuple[float, int, float]:
    pos, neg, negation_hits, sentiment_hits = _sentiment_counts(clause_tokens)
    evidence = abs(pos - neg)
    return evidence, negation_hits, sentiment_hits


def _softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _sentence_confidence(text: str, top_prob: float, margin: float, evidence_count: int, sentiment_evidence: float) -> float:
    length_boost = min(0.08, token_count(text) * 0.003)
    evidence_boost = min(0.15, evidence_count * 0.04)
    margin_boost = min(0.15, margin * 0.6)
    sentiment_boost = min(0.10, sentiment_evidence * 0.05)
    
    # Base confidence is now slightly more conservative
    confidence = 0.35 + 0.30 * top_prob + length_boost + evidence_boost + margin_boost + sentiment_boost
    return round(min(0.99, confidence), 4)


def _contains_any_aspect_seed(tokens: List[str], seed_vocab: set[str]) -> bool:
    """True if ANY of the canonical aspect seeds appear in the token list."""
    return any(seed.lower() in tokens for seed in seed_vocab)


def is_explicit_clause(clause: str, seed_vocab: set[str]) -> bool:
    """
    Returns True if the clause contains a direct mention of any aspect in the seed_vocab.
    This is used for the clausal routing pipeline.
    """
    tokens = [t.split("'")[0] for t in tokenize(clause.lower())]
    return _contains_any_aspect_seed(tokens, seed_vocab)


def extract_explicit_aspect(clause: str, seed_vocab: set[str]) -> tuple[str | None, str]:
    """
    Extracts explicit aspect and sentiment from a clause.
    Used for the 'Explicit Branch' of the clausal pipeline.
    """
    tokens = [t.split("'")[0] for t in tokenize(clause.lower())]
    sentiment = infer_sentiment(clause)
    
    # Priority matching: find the first seed that appears in the tokens
    for seed in seed_vocab:
        if seed.lower() in tokens:
            return seed, sentiment
            
    return None, sentiment



def _score_clause(
    clause: str,
    candidate_aspects: List[str],
    *,
    seed_vocab: set[str],
    llm_client: Any | None,
    confidence_threshold: float,
    symptom_map: dict[str, str],
) -> tuple[str | None, str, float, bool]:
    if not candidate_aspects:
        return None, infer_sentiment(clause), 0.0, False

    clause_tokens = tokenize(clause)
    
    # Note: Global Explicit Filter moved to build_implicit_row for routing.
    # We no longer reject here, as routing ensures this function only sees implicit candidates.

    sentiment_evidence, negation_hits, sentiment_hits = _sentiment_evidence(clause_tokens)
    
    # Research-Grade Improvement: allow clauses with SYMPTOM hits to pass even without traditional sentiment hits.
    # This captures implicit signals like "pointer not moving" which are inherently negative but lack 'bad' keywords.
    has_symptom = any(s.lower() in clause_tokens for s in symptom_map)
    
    if (sentiment_hits == 0 or sentiment_evidence <= 0.0) and not has_symptom:
        return None, infer_sentiment(clause), 0.0, False

    scores = []
    for aspect in candidate_aspects:
        lexical = _aspect_lexical_score(clause, aspect, seed_vocab=seed_vocab, symptom_map=symptom_map)
        window_support = _aspect_window_support(clause_tokens, aspect)
        scores.append(lexical + window_support)
    probs = _softmax(scores)
    ranked = sorted(zip(candidate_aspects, scores, probs), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, infer_sentiment(clause), 0.0, False

    top_aspect, top_score, top_prob = ranked[0]
    second_prob = ranked[1][2] if len(ranked) > 1 else 0.0
    sentiment = infer_sentiment(clause)
    evidence_count = max(0, len([x for x in scores if x > 0]))
    
    confidence = _sentence_confidence(clause, top_prob, top_prob - second_prob, evidence_count, sentiment_evidence)
    
    if sentiment_evidence < 0.8:
        confidence = round(max(0.0, confidence - 0.20), 4)
    if top_score < 1.5:
        confidence = round(max(0.0, confidence - 0.15), 4)
    if negation_hits:
        confidence = round(min(0.99, confidence + 0.05), 4)
    if len(clause.split()) <= 4 and top_score < 2.0:
        confidence = round(max(0.0, confidence - 0.15), 4)

    # Plausible requires actual evidence (score > 0)
    # Research-Grade: has_symptom allows plausibility even if sentiment_evidence is low
    plausible = (top_score >= 1.0 or top_prob >= 0.65) and (sentiment_evidence >= 0.5 or has_symptom) and top_score > 0

    # Journal Worthiness: Escalate to LLM if rule-based approach is unsure or failed, 
    # but we still have clear sentiment evidence.
    should_call_llm = llm_client is not None and sentiment_evidence >= 0.5 and (not plausible or confidence < confidence_threshold)
    
    if should_call_llm:
        result = llm_client.infer(sentence=clause, candidate_aspects=candidate_aspects)
        if result is not None and result.aspect:
            return [(result.aspect, max(confidence, float(result.confidence)))], result.sentiment, max(confidence, float(result.confidence)), bool(result.is_novel_aspect)

    candidates = [(top_aspect, confidence)]
    if len(ranked) > 1 and ranked[1][1] > 0:
        # Add second candidate for fuzzy research
        candidates.append((ranked[1][0], round(confidence * (probs[1]/max(1e-9, probs[0])), 4)))
        
    if confidence >= confidence_threshold and plausible:
        return candidates[:1], sentiment, confidence, False

    if plausible:
        return candidates, sentiment, confidence, False
    return [], sentiment, confidence, False


def collect_implicit_diagnostics(
    train_rows: List[Dict[str, Any]],
    *,
    text_column: str,
    candidate_aspects: List[str],
    seed_vocab: set[str],
    confidence_threshold: float,
    learned_seed_vocab: List[str] | None = None,
    symptom_map: dict[str, str] = None,
) -> Dict[str, Any]:
    symptom_map = symptom_map or {}
    rejection_reasons: Counter[str] = Counter()
    rejected_examples: List[Dict[str, str]] = []
    aspect_counts: Counter[str] = Counter()
    fallback_count = 0
    scored_clauses = 0
    negation_hits = 0
    sentiment_hit_count = 0
    accepted_examples: List[Dict[str, str]] = []

    for row in train_rows[:1000]:
        text = normalize_whitespace(row.get(text_column, ""))
        if not text:
            continue
        for sentence in split_sentences(text):
            for clause in _split_contrastive(sentence):
                scored_clauses += 1
                clause_candidate_aspects = candidate_aspects or []
                if not clause_candidate_aspects:
                    continue
                candidates, _, confidence, _ = _score_clause(
                    clause,
                    clause_candidate_aspects,
                    seed_vocab=seed_vocab,
                    llm_client=None,
                    confidence_threshold=confidence_threshold,
                    symptom_map=symptom_map,
                )
                clause_tokens = tokenize(clause)
                _, clause_negations, clause_sentiment_hits = _sentiment_evidence(clause_tokens)
                negation_hits += clause_negations
                sentiment_hit_count += clause_sentiment_hits
                if candidates:
                    aspect = candidates[0][0]
                    confidence = candidates[0][1]
                    aspect_counts[aspect] += 1
                    if len(accepted_examples) < 20:
                        accepted_examples.append({"clause": clause, "aspect": aspect, "confidence": f"{confidence:.3f}"})
                else:
                    fallback_count += 1
                    if len(rejected_examples) < 20:
                        rejected_examples.append({"clause": clause, "reason": "no_plausible_aspect"})

        for phrase in _extract_candidate_phrases(text, seed_vocab=seed_vocab):
            reason = _aspect_rejection_reason(phrase, seed_vocab=seed_vocab)
            if reason is not None:
                rejection_reasons[reason] += 1
                if len(rejected_examples) < 20:
                    rejected_examples.append({"phrase": phrase, "reason": reason})

    return {
        "top_implicit_aspects": aspect_counts.most_common(20),
        "learned_seed_vocab": list(learned_seed_vocab or candidate_aspects[: min(len(candidate_aspects), 50)]),
        "sentiment_lexicon_coverage": {
            "sentiment_hit_count": sentiment_hit_count,
            "negation_hit_count": negation_hits,
            "candidate_aspect_count": len(candidate_aspects),
        },
        "candidate_rejection_reasons": rejection_reasons.most_common(),
        "false_positive_samples": accepted_examples,
        "false_negative_samples": rejected_examples,
        "accepted_examples": accepted_examples,
        "rejected_examples": rejected_examples,
        "fallback_clause_count_sample": fallback_count,
        "scored_clause_count_sample": scored_clauses,
    }


def build_implicit_row(
    row: Dict[str, Any],
    *,
    text_column: str,
    candidate_aspects: List[str],
    seed_vocab: set[str],
    confidence_threshold: float,
    llm_enabled: bool = False,
    llm_settings: Any | None = None,
    symptom_map: dict[str, str] = None,
) -> Dict[str, Any]:
    symptom_map = symptom_map or {}
    text = normalize_whitespace(row.get(text_column, ""))
    sentences = split_sentences(text)
    llm_client = build_llm_client(llm_settings, enabled=llm_enabled) if llm_settings is not None else None
    aspects: List[str] = []
    aspect_sentiments: Dict[str, str] = {}
    aspect_confidence: Dict[str, float] = {}
    novel_aspects: List[str] = []
    confidences: List[float] = []
    tier = 1
    if not sentences:
        tier = 3

    for sentence in (sentences or [text]):
        for clause in _split_contrastive(sentence):
            # 1. Routing: IF explicit -> explicit branch
            if is_explicit_clause(clause, seed_vocab):
                aspect, sentiment = extract_explicit_aspect(clause, seed_vocab)
                if aspect:
                    if aspect not in aspects:
                        aspects.append(aspect)
                        aspect_sentiments[aspect] = sentiment
                        aspect_confidence[aspect] = 0.99  # Explicit is high confidence by definition
                    confidences.append(0.99)
                continue

            # 2. Routing: ELSE -> implicit branch
            candidates, sentiment, confidence, is_novel = _score_clause(
                clause,
                candidate_aspects,
                seed_vocab=seed_vocab,
                llm_client=llm_client,
                confidence_threshold=confidence_threshold,
                symptom_map=symptom_map,
            )
            if not candidates:
                tier = 3
                continue
            
            # Confidence-based tiering (primary result)
            primary_aspect, primary_conf = candidates[0]
            
            if primary_conf < confidence_threshold:
                tier = 3
            elif primary_conf < 0.85 and tier < 3:
                tier = 2

            for aspect, conf in candidates:
                if aspect not in aspects:
                    aspects.append(aspect)
                    aspect_sentiments[aspect] = sentiment
                    aspect_confidence[aspect] = conf
                
            confidences.append(primary_conf)
            if is_novel and primary_aspect not in candidate_aspects:
                novel_aspects.append(primary_aspect)
                novel_aspects.append(aspect)

    if not aspects:
        tier = 3

    dominant_sentiment = infer_sentiment(text)
    avg_confidence = round(sum(confidences) / max(1, len(confidences)), 4)
    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "source_text": text,
        "implicit": {
            "aspects": aspects,
            "dominant_sentiment": dominant_sentiment,
            "aspect_sentiments": aspect_sentiments,
            "aspect_confidence": aspect_confidence,
            "avg_confidence": avg_confidence,
            "extraction_tier": tier,
            "novel_aspects": novel_aspects,
            "sentence_count_processed": len(sentences) or 1,
        },
    }
