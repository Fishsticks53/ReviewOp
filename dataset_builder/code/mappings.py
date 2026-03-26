import json
from pathlib import Path
from typing import Any, Dict, Set


CONTRASTIVE_CONJUNCTIONS = ("but", "however", "although", "while", "whereas", "yet")

TEXT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "if", "in", "is", "it", "its", "my", "of", "on",
    "or", "our", "so", "that", "the", "their", "them", "there", "these",
    "they", "this", "to", "was", "we", "were", "with", "you", "your",
    "than", "which", "will", "all", "about", "what", "where", "when", "who", "whom",
    "how", "can", "could", "would", "should", "may", "might", "must", "shall",
    "am", "was", "were", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "me", "him", "her", "us", "them", "his", "her",
    "its", "our", "their", "this", "that", "these", "those", "each", "every",
    "any", "all", "some", "none", "no", "not", "only", "own", "other", "another",
    "very", "quite", "just", "now", "then", "there", "here", "again", "too",
    "also", "really", "even", "more", "most", "less", "least", "well", "away",
}

GENERIC_ASPECT_STOPWORDS = {
    "thing", "things", "people", "place", "item", "items", "product", "products",
    "way", "lot", "time", "stuff", "like", "one", "had", "not",
    "even", "very", "just", "would", "could", "should", "make", "made", "good",
    "great", "best", "more", "most", "some", "many", "much", "also", "really",
    "been", "being", "get", "got", "going", "keep", "kept", "take", "taken",
    "say", "said", "see", "seen", "use", "used", "using", "come", "came",
    "stay", "stayed", "staying", "give", "given", "giving",
    "computer", "laptop", "phone", "device", "restaurant", "pizza", "wine", "food",
    "sauce", "meal", "drink", "service", "program", "app", "application", "software",
    "bit", "part", "side", "area", "case", "end", "back", "front", "top", "bottom",
}

POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "clean", "comfortable", "delicious",
    "easy", "excellent", "fantastic", "fast", "friendly", "fresh", "good",
    "great", "helpful", "intuitive", "love", "loved", "nice", "perfect",
    "quick", "responsive", "smooth", "solid", "stunning", "tasty", "worth",
    "wonderful", "delighted", "superb", "brilliant", "outstanding",
}

NEGATIVE_WORDS = {
    "awful", "bad", "bland", "boring", "broken", "cheap", "confusing",
    "crash", "crashes", "crashed", "defect", "delay", "delayed", "difficult",
    "dirty", "drains", "expensive", "hate", "lag", "laggy", "late", "noisy",
    "overpriced", "poor", "rude", "slow", "terrible", "uncomfortable",
    "unhelpful", "worse", "worst", "horrible", "annoying", "frustrating",
}

TARGET_COLUMN_HINTS = ("label", "class", "category", "rating", "sentiment", "target")


def load_priors(priors_path: Path) -> tuple[set[str], dict[str, str]]:
    """
    Loads domain-specific priors from a JSON file.
    Raises FileNotFoundError if missing to prevent silent failures.
    """
    if not priors_path.exists():
        raise FileNotFoundError(f"CRITICAL: Domain priors not found at {priors_path.resolve()}. ")
        
    try:
        with open(priors_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            seeds = set(data.get("generic_review_aspect_seeds", []))
            symptoms = data.get("symptom_map", {})
            return seeds, symptoms
    except Exception as e:
        raise ValueError(f"CRITICAL: Failed to parse domain priors JSON: {e}")
