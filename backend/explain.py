def classify_risk(trust_score: float) -> str:
    if trust_score >= 85:
        return "High Trust"
    elif trust_score >= 70:
        return "Medium Trust"
    else:
        return "High Hallucination Risk"

def explain_trust(trust_score: float) -> str:
    if trust_score >= 85:
        return (
            "The answer is strongly supported by retrieved evidence with "
            "high semantic alignment and no detected contradictions."
        )
    elif trust_score >= 70:
        return (
            "The answer is mostly supported by evidence, but some parts "
            "lack strong grounding. Caution is advised."
        )
    else:
        return (
            "The answer shows weak evidence support and may contain "
            "hallucinated information."
        )