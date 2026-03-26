from __future__ import annotations


CONTRASTIVE_CONJUNCTIONS = ("but", "however", "although", "while", "whereas", "yet")

TEXT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "if", "in", "is", "it", "its", "my", "of", "on",
    "or", "our", "so", "that", "the", "their", "them", "there", "these",
    "they", "this", "to", "was", "we", "were", "with", "you", "your",
}

GENERIC_ASPECT_STOPWORDS = {
    "thing", "things", "people", "place", "item", "items", "product", "products",
    "way", "lot", "time", "stuff", "like", "one", "had", "not",
    "even", "very", "just", "would", "could", "should", "make", "made", "good",
    "great", "best", "more", "most", "some", "many", "much", "also", "really",
    "been", "being", "get", "got", "going", "keep", "kept", "take", "taken",
    "say", "said", "see", "seen", "use", "used", "using", "come", "came",
}

POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "clean", "comfortable", "delicious",
    "easy", "excellent", "fantastic", "fast", "friendly", "fresh", "good",
    "great", "helpful", "intuitive", "love", "loved", "nice", "perfect",
    "quick", "responsive", "smooth", "solid", "stunning", "tasty", "worth",
}

NEGATIVE_WORDS = {
    "awful", "bad", "bland", "boring", "broken", "cheap", "confusing",
    "crash", "crashes", "crashed", "defect", "delay", "delayed", "difficult",
    "dirty", "drains", "expensive", "hate", "lag", "laggy", "late", "noisy",
    "overpriced", "poor", "rude", "slow", "terrible", "uncomfortable",
    "unhelpful", "worse", "worst",
}

TARGET_COLUMN_HINTS = ("label", "class", "category", "rating", "sentiment", "target")

GENERIC_REVIEW_ASPECT_SEEDS = {
    "battery", "camera", "comfort", "delivery", "display", "food", "performance",
    "price", "quality", "service", "taste", "connectivity", "usability",
    "ambiance", "stability", "durability", "portability", "sound", "storage",
    "memory", "safety", "cleanliness", "value", "speed", "design", "size",
    "weight", "noise", "heat", "software", "hardware", "wifi", "bluetooth",
    "packaging", "warranty", "accessories", "staff", "wait_time", "atmosphere",
    "parking", "resolution", "efficiency", "reliability", "ergonomics"
}

# Symptom-to-Aspect mapping for indirect/implicit reasoning
SYMPTOM_MAP = {
    "waited": "service",
    "waiting": "service",
    "seating": "service",
    "seated": "service",
    "staff": "service",
    "waiter": "service",
    "host": "service",
    "manager": "service",
    "ordered": "service",
    "dropping": "connectivity",
    "dropped": "connectivity",
    "signal": "connectivity",
    "connection": "connectivity",
    "wifi": "connectivity",
    "lag": "performance",
    "laggy": "performance",
    "stutter": "performance",
    "pointer": "usability",
    "cursor": "usability",
    "navigation": "usability",
    "mousepad": "usability",
    "finger": "usability",
    "click": "usability",
    "minutes": "service",
    "slow": "performance",
    "bitter": "taste",
    "salty": "taste",
    "bland": "taste",
    "tasty": "food",
    "sour": "taste",
    "sweet": "taste",
    "spicy": "taste",
    "flavor": "taste",
    "pricey": "price",
    "expensive": "price",
    "affordable": "price",
    "stalling": "stability",
    "crash": "stability",
    "crashed": "stability",
    "quiet": "noise",
    "loud": "noise",
    "silent": "noise",
    "buzzing": "noise",
    "hot": "heat",
    "warm": "heat",
    "overheating": "heat",
    "blurred": "camera",
    "dim": "display",
    "bright": "display",
    "grainy": "display",
    "heavy": "weight",
    "light": "weight",
    "bugs": "software",
    "glitch": "software",
    "unresponsive": "performance",
    "reloads": "stability",
    "reloading": "stability",
    "cramped": "comfort",
    "spacious": "comfort",
    "seat": "comfort",
    "wooden": "comfort",
    "cheap": "price",
    "breaking": "durability",
    "bulk": "size",
    "roomy": "size",
    "small": "size",
    "tiny": "size",
    "burnt": "food",
    "overcooked": "food",
    "raw": "food",
    "undercooked": "food",
    "hour": "wait_time",
    "forever": "wait_time",
    "long": "wait_time",
    "shorting": "hardware",
    "broken": "hardware",
    "defective": "hardware",
    "webcam": "hardware",
    "mic": "hardware",
    "speaker": "hardware",
    "reboot": "stability",
}
