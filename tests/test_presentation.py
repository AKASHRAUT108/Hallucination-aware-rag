from src.pipeline.presentation import build_views
from src.pipeline.result import RAGResult

MSG = "ABSTAIN"


def res(raw, conf, refused=False):
    return RAGResult("q", MSG if refused else raw, raw, [], [], abstained=refused, confidence=conf)


def test_confident_answer_is_shown_by_both():
    v = build_views(res("75 percent [1].", 0.95), 0.6, MSG)
    assert v["baseline"]["text"] == "75 percent [1]." and not v["baseline"]["abstained"]
    assert v["aware"]["text"] == "75 percent [1]." and not v["aware"]["abstained"]


def test_low_confidence_answer_is_withheld_only_by_the_verifier():
    v = build_views(res("200 rupees [1].", 0.19), 0.6, MSG)
    assert v["baseline"]["text"] == "200 rupees [1]."
    assert v["aware"]["text"] == MSG and v["aware"]["suppressed_by_verifier"]


def test_threshold_slider_changes_the_outcome_without_new_generation():
    r = res("Ten days [2].", 0.41)
    assert build_views(r, 0.3, MSG)["aware"]["abstained"] is False
    assert build_views(r, 0.6, MSG)["aware"]["abstained"] is True


def test_generator_refusal_is_not_credited_to_the_verifier():
    v = build_views(res("I don't know.", 0.0, refused=True), 0.6, MSG)
    assert v["baseline"]["abstained"] and v["aware"]["abstained"] and not v["aware"]["suppressed_by_verifier"]
