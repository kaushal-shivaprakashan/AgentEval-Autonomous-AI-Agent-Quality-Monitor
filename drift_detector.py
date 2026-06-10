import statistics
from dataclasses import dataclass, field
from typing import List, Optional
from app.services.quality_scorer import QualityReport


@dataclass
class DriftEvent:
    dimension: str
    baseline_avg: float
    current_avg: float
    drop: float
    severity: str       # MINOR | MODERATE | SEVERE | CRITICAL
    triggered_at: int   # index in the time series


@dataclass
class DriftReport:
    drift_detected: bool = False
    overall_drift: float = 0.0       # overall score drop
    events: List[DriftEvent] = field(default_factory=list)
    baseline_scores: List[int] = field(default_factory=list)
    current_scores: List[int] = field(default_factory=list)
    trend: str = "STABLE"            # IMPROVING | STABLE | DEGRADING | CRITICAL
    alert_message: str = ""
    should_trigger_rca: bool = False


# Thresholds for drift severity
DRIFT_THRESHOLDS = {
    "overall": {"MINOR": 5, "MODERATE": 10, "SEVERE": 20, "CRITICAL": 30},
    "hallucination": {"MINOR": 10, "MODERATE": 20, "SEVERE": 35, "CRITICAL": 50},
}


def detect_drift(
    baseline_reports: List[QualityReport],
    current_reports: List[QualityReport],
) -> DriftReport:
    """Compare two sets of quality reports and detect degradation."""

    if not baseline_reports or not current_reports:
        return DriftReport()

    # Extract overall scores
    baseline_scores = [r.overall for r in baseline_reports]
    current_scores  = [r.overall for r in current_reports]

    baseline_avg = statistics.mean(baseline_scores)
    current_avg  = statistics.mean(current_scores)
    overall_drop = baseline_avg - current_avg

    report = DriftReport(
        baseline_scores=baseline_scores,
        current_scores=current_scores,
        overall_drift=round(overall_drop, 1),
    )

    # Overall drift classification
    if overall_drop >= DRIFT_THRESHOLDS["overall"]["CRITICAL"]:
        report.drift_detected = True
        report.trend = "CRITICAL"
        report.should_trigger_rca = True
        report.alert_message = f"CRITICAL degradation: {round(overall_drop, 1)} point drop in overall quality"
        report.events.append(DriftEvent(
            dimension="Overall Quality", baseline_avg=baseline_avg,
            current_avg=current_avg, drop=overall_drop,
            severity="CRITICAL", triggered_at=0,
        ))
    elif overall_drop >= DRIFT_THRESHOLDS["overall"]["SEVERE"]:
        report.drift_detected = True
        report.trend = "DEGRADING"
        report.should_trigger_rca = True
        report.alert_message = f"Severe degradation: {round(overall_drop, 1)} point drop detected"
        report.events.append(DriftEvent(
            dimension="Overall Quality", baseline_avg=baseline_avg,
            current_avg=current_avg, drop=overall_drop,
            severity="SEVERE", triggered_at=0,
        ))
    elif overall_drop >= DRIFT_THRESHOLDS["overall"]["MODERATE"]:
        report.drift_detected = True
        report.trend = "DEGRADING"
        report.alert_message = f"Moderate degradation: {round(overall_drop, 1)} point drop"
        report.events.append(DriftEvent(
            dimension="Overall Quality", baseline_avg=baseline_avg,
            current_avg=current_avg, drop=overall_drop,
            severity="MODERATE", triggered_at=0,
        ))
    elif overall_drop >= DRIFT_THRESHOLDS["overall"]["MINOR"]:
        report.drift_detected = True
        report.trend = "DEGRADING"
        report.alert_message = f"Minor quality drift: {round(overall_drop, 1)} point drop"
    elif overall_drop < -5:
        report.trend = "IMPROVING"
        report.alert_message = f"Quality improvement detected: +{round(-overall_drop, 1)} points"
    else:
        report.trend = "STABLE"

    # Per-dimension drift
    dims = ["accuracy", "relevance", "hallucination", "coherence", "safety"]
    for dim in dims:
        b_scores = [getattr(r, dim).score for r in baseline_reports]
        c_scores = [getattr(r, dim).score for r in current_reports]
        b_avg = statistics.mean(b_scores)
        c_avg = statistics.mean(c_scores)

        # For hallucination: increase is bad
        drop = (c_avg - b_avg) if dim == "hallucination" else (b_avg - c_avg)

        threshold = DRIFT_THRESHOLDS.get(dim, DRIFT_THRESHOLDS["overall"])
        if drop >= threshold.get("CRITICAL", 30):
            sev = "CRITICAL"
        elif drop >= threshold.get("SEVERE", 20):
            sev = "SEVERE"
        elif drop >= threshold.get("MODERATE", 10):
            sev = "MODERATE"
        elif drop >= threshold.get("MINOR", 5):
            sev = "MINOR"
        else:
            continue  # no notable drift on this dimension

        report.events.append(DriftEvent(
            dimension=dim.capitalize(),
            baseline_avg=round(b_avg, 1),
            current_avg=round(c_avg, 1),
            drop=round(drop, 1),
            severity=sev,
            triggered_at=0,
        ))

    return report


def detect_trend(scores: List[int], window: int = 5) -> str:
    """Detect trend in a single time series of scores."""
    if len(scores) < 3:
        return "INSUFFICIENT_DATA"

    recent = scores[-window:]
    if len(recent) < 2:
        return "STABLE"

    # Simple linear regression slope
    n = len(recent)
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n
    numerator   = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return "STABLE"

    slope = numerator / denominator

    if slope > 1.5:    return "IMPROVING"
    if slope < -1.5:   return "DEGRADING"
    return "STABLE"
