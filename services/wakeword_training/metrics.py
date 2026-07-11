from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    false_reject_rate: float
    false_accept_rate: float
    num_positive: int
    num_negative: int
    num_false_rejects: int
    num_false_accepts: int


def compute_metrics(
    positive_scores: list[float], negative_scores: list[float], threshold: float
) -> EvalResult:
    num_false_rejects = sum(1 for s in positive_scores if s < threshold)
    num_false_accepts = sum(1 for s in negative_scores if s >= threshold)
    return EvalResult(
        false_reject_rate=(num_false_rejects / len(positive_scores)) if positive_scores else 0.0,
        false_accept_rate=(num_false_accepts / len(negative_scores)) if negative_scores else 0.0,
        num_positive=len(positive_scores),
        num_negative=len(negative_scores),
        num_false_rejects=num_false_rejects,
        num_false_accepts=num_false_accepts,
    )
