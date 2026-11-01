"""Turns one diagnose() result into the two answers the demo shows side by side.

diagnose() does the slow work once (retrieve, generate, verify) WITHOUT applying a threshold, so the
demo's threshold slider can change the outcome instantly without regenerating anything.
"""


def build_views(res, threshold: float, abstain_message: str) -> dict:
    refused = bool(res.abstained)                      # the generator itself said "I don't know"
    confidence = res.confidence or 0.0
    suppressed = (not refused) and confidence < threshold
    return {
        "baseline": {
            "text": abstain_message if refused else res.raw_answer,
            "abstained": refused,
        },
        "aware": {
            "text": abstain_message if (refused or suppressed) else res.raw_answer,
            "abstained": refused or suppressed,
            "suppressed_by_verifier": suppressed,
            "confidence": confidence,
            "threshold": threshold,
        },
    }
