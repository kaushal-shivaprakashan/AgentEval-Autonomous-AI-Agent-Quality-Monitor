import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services.quality_scorer import score_log_entry
from app.services.drift_detector import detect_drift
from app.services.rca_engine import run_rca

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────
class LogEntry(BaseModel):
    id: int
    timestamp: str
    input: str
    output: str


class EvalRequest(BaseModel):
    logs: List[LogEntry]
    agent_name: str = "AI Agent"
    agent_context: str = "You are a helpful AI assistant."


class CompareRequest(BaseModel):
    baseline: EvalRequest
    current: EvalRequest
    run_rca: bool = True


class SingleEvalRequest(BaseModel):
    input: str
    output: str
    agent_context: str = "You are a helpful AI assistant."


# ── Helpers ───────────────────────────────────────────────────────────────────
def report_to_dict(r):
    return {
        "log_id":      r.log_id,
        "overall":     r.overall,
        "risk_level":  r.risk_level,
        "red_flags":   r.red_flags,
        "dimensions": {
            "accuracy":      {"score": r.accuracy.score,      "rationale": r.accuracy.rationale},
            "relevance":     {"score": r.relevance.score,     "rationale": r.relevance.rationale},
            "hallucination": {"score": r.hallucination.score, "rationale": r.hallucination.rationale, "issues": r.hallucination.issues},
            "coherence":     {"score": r.coherence.score,     "rationale": r.coherence.rationale},
            "safety":        {"score": r.safety.score,        "rationale": r.safety.rationale},
        },
    }


def drift_to_dict(d):
    return {
        "drift_detected":     d.drift_detected,
        "overall_drift":      d.overall_drift,
        "trend":              d.trend,
        "alert_message":      d.alert_message,
        "should_trigger_rca": d.should_trigger_rca,
        "baseline_scores":    d.baseline_scores,
        "current_scores":     d.current_scores,
        "events": [
            {
                "dimension":    ev.dimension,
                "baseline_avg": ev.baseline_avg,
                "current_avg":  ev.current_avg,
                "drop":         ev.drop,
                "severity":     ev.severity,
            }
            for ev in d.events
        ],
    }


def rca_to_dict(r):
    return {
        "root_cause":           r.root_cause,
        "evidence":             r.evidence,
        "contributing_factors": r.contributing_factors,
        "severity":             r.severity,
        "confidence":           r.confidence,
        "post_mortem_draft":    r.post_mortem_draft,
        "fixes": [
            {
                "action":   f.action,
                "impact":   f.impact,
                "priority": f.priority,
                "effort":   f.effort,
            }
            for f in r.fixes
        ],
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/eval/single")
async def evaluate_single(req: SingleEvalRequest):
    """Evaluate a single agent output across 5 quality dimensions."""
    report = await score_log_entry(req.input, req.output, req.agent_context)
    return report_to_dict(report)


@router.post("/eval/batch")
async def evaluate_batch(req: EvalRequest):
    """Evaluate a batch of agent logs."""
    tasks = [
        score_log_entry(
            log.input, log.output, req.agent_context, log.id
        )
        for log in req.logs
    ]
    reports = await asyncio.gather(*tasks)
    overalls = [r.overall for r in reports]
    avg = round(sum(overalls) / len(overalls)) if overalls else 0

    return {
        "agent_name":    req.agent_name,
        "log_count":     len(reports),
        "average_score": avg,
        "risk_level":    reports[0].risk_level if reports else "UNKNOWN",
        "scores":        [report_to_dict(r) for r in reports],
    }


@router.post("/eval/compare")
async def compare_agents(req: CompareRequest):
    """
    Compare baseline vs current agent.
    Runs quality scoring, drift detection, and optional RCA.
    This is the core Foundry IQ multi-step reasoning pipeline.
    """
    # Step 1 + 2: Score both agents concurrently
    baseline_tasks = [
        score_log_entry(log.input, log.output, req.baseline.agent_context, log.id)
        for log in req.baseline.logs
    ]
    current_tasks = [
        score_log_entry(log.input, log.output, req.current.agent_context, log.id)
        for log in req.current.logs
    ]

    baseline_reports, current_reports = await asyncio.gather(
        asyncio.gather(*baseline_tasks),
        asyncio.gather(*current_tasks),
    )

    # Step 3: Drift detection
    drift = detect_drift(list(baseline_reports), list(current_reports))

    # Step 4: RCA (if triggered)
    rca = None
    if req.run_rca and (drift.should_trigger_rca or drift.drift_detected):
        baseline_logs_raw = [{"input": l.input, "output": l.output} for l in req.baseline.logs]
        current_logs_raw  = [{"input": l.input, "output": l.output} for l in req.current.logs]
        rca_report = await run_rca(
            baseline_logs_raw, current_logs_raw, drift,
            agent_name=req.current.agent_name,
        )
        rca = rca_to_dict(rca_report)

    baseline_avg = round(sum(r.overall for r in baseline_reports) / len(baseline_reports))
    current_avg  = round(sum(r.overall for r in current_reports)  / len(current_reports))

    return {
        "baseline": {
            "agent_name":    req.baseline.agent_name,
            "average_score": baseline_avg,
            "scores":        [report_to_dict(r) for r in baseline_reports],
        },
        "current": {
            "agent_name":   req.current.agent_name,
            "average_score": current_avg,
            "scores":        [report_to_dict(r) for r in current_reports],
        },
        "drift":    drift_to_dict(drift),
        "rca":      rca,
    }


@router.get("/dimensions")
def get_dimensions():
    return {
        "dimensions": [
            {"key": "accuracy",      "label": "Accuracy",        "description": "Is the information factually correct?",       "weight": 0.25},
            {"key": "relevance",     "label": "Relevance",       "description": "Does it address the actual query?",           "weight": 0.20},
            {"key": "hallucination", "label": "Hallucination Risk", "description": "Does it contain fabricated facts? (lower = better)", "weight": 0.25},
            {"key": "coherence",     "label": "Reasoning",       "description": "Is the reasoning logical?",                  "weight": 0.15},
            {"key": "safety",        "label": "Safety",          "description": "Does it follow safe, ethical guidelines?",   "weight": 0.15},
        ]
    }
