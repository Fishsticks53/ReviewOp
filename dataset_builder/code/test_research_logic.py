from implicit_features import _aspect_lexical_score
from mappings import GENERIC_REVIEW_ASPECT_SEEDS

def test_research_logic():
    seeds = GENERIC_REVIEW_ASPECT_SEEDS
    
    # Case 1: Symptom based (no direct mention)
    score_service = _aspect_lexical_score("I waited 20 minutes for my order", "service", seed_vocab=seeds)
    print(f"Symptom 'waited' -> 'service' score: {score_service}")
    assert score_service > 0
    
    # Case 2: Explicit mention penalty
    score_explicit = _aspect_lexical_score("The service was very slow", "service", seed_vocab=seeds)
    print(f"Explicit 'service' mentioned -> score: {score_explicit}")
    assert score_explicit <= 0
    
    # Case 3: Another symptom
    score_perf = _aspect_lexical_score("The screen reloads constantly and is very laggy", "performance", seed_vocab=seeds)
    print(f"Symptom 'laggy' -> 'performance' score: {score_perf}")
    assert score_perf > 0
    
    print("Logic Verification Passed!")

if __name__ == "__main__":
    test_research_logic()
