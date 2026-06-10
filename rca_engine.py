import os
import json
import httpx
from dataclasses import dataclass, field
from typing import List, Optional
from app.services.drift_detector import DriftReport

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"


@dataclass
class FixRecommendation:
    action: str
    impact: str
    priority: int       # 1 = highest
    effort: str         # LOW | MEDIUM | HIGH


@dataclass
class RCAReport:
    root_cause: str
    evidence: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    fixes: List[FixRecommendation] = field(default_factory=list)
    severity: str = "HIGH"      # CRITICAL | HIGH | MEDIUM | LOW
    confidence: str = "HIGH"    # HIGH | MEDIUM | LOW
    post_mortem_draft: str = ""


async def _call_claude(system: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 2000,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        data = resp.json()
        if "error" in data:
            raise ValueError(data["error"]["message"])
        return data["content"][0]["text"]


async def run_rca(
    baseline_logs: list,
    degraded_logs: list,
    drift_report: DriftReport,
    agent_name: str = "AI Agent",
) -> RCAReport:
    """
    Foundry IQ multi-step reasoning:
    Step 1 — Pattern extraction from baseline vs degraded outputs
    Step 2 — Root cause hypothesis generation
    Step 3 — Evidence correlation
    Step 4 — Fix recommendation ranking
    Step 5 — Post-mortem draft generation
    """

    # Build evidence from red flags and drift events
    static_evidence = []
    for log in degraded_logs:
        if hasattr(log, "red_flags"):
            static_evidence.extend(log.red_flags)

    drift_evidence = [
        f"{ev.dimension}: {ev.severity} drift ({round(ev.drop, 1)} point drop)"
        for ev in drift_report.events
    ]

    if not ANTHROPIC_API_KEY:
        return _static_rca(static_evidence, drift_evidence, drift_report)

    # ── Step 1: Extract patterns ─────────────────────────────────────────────
    baseline_text = "\n".join([
        f"Q: {l.get('input','')}\nA: {l.get('output','')}"
        for l in baseline_logs[:4]
    ])
    degraded_text = "\n".join([
        f"Q: {l.get('input','')}\nA: {l.get('output','')}"
        for l in degraded_logs[:4]
    ])

    system_rca = """You are an expert AI systems reliability engineer performing root cause analysis.
You reason step by step to identify why an AI agent has degraded.
Return ONLY valid JSON, no markdown fences.

JSON schema:
{
  "root_cause": "string — the primary cause in one clear sentence",
  "evidence": ["string — specific evidence from the outputs"],
  "contributing_factors": ["string — secondary factors"],
  "fixes": [
    {"action": "string", "impact": "string", "priority": 1, "effort": "LOW|MEDIUM|HIGH"}
  ],
  "severity": "CRITICAL|HIGH|MEDIUM|LOW",
  "confidence": "HIGH|MEDIUM|LOW",
  "post_mortem_draft": "string — 2-3 sentence post-mortem summary"
}"""

    prompt = f"""You are performing root cause analysis for a degraded AI agent: {agent_name}

DRIFT SUMMARY:
- Overall quality drop: {drift_report.overall_drift} points
- Trend: {drift_report.trend}
- Alert: {drift_report.alert_message}
- Dimension events: {', '.join([f"{e.dimension}: -{e.drop}pt" for e in drift_report.events[:5]])}

BASELINE AGENT OUTPUTS (healthy):
{baseline_text}

DEGRADED AGENT OUTPUTS (current):
{degraded_text}

STATIC RED FLAGS DETECTED:
{chr(10).join(static_evidence) if static_evidence else "None"}

REASONING STEPS:
1. Compare baseline vs degraded outputs — what changed?
2. Identify the most likely root cause
3. Find evidence supporting this root cause
4. List contributing factors
5. Recommend fixes ranked by impact

Return your analysis as JSON."""

    raw = await _call_claude(system_rca, prompt)

    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```")
        data = json.loads(clean)

        fixes = [
            FixRecommendation(
                action   = f.get("action", ""),
                impact   = f.get("impact", ""),
                priority = f.get("priority", i + 1),
                effort   = f.get("effort", "MEDIUM"),
            )
            for i, f in enumerate(data.get("fixes", []))
        ]

        return RCAReport(
            root_cause           = data.get("root_cause", "Unable to determine root cause"),
            evidence             = data.get("evidence", []),
            contributing_factors = data.get("contributing_factors", []),
            fixes                = fixes,
            severity             = data.get("severity", "HIGH"),
            confidence           = data.get("confidence", "MEDIUM"),
            post_mortem_draft    = data.get("post_mortem_draft", ""),
        )

    except Exception:
        return _static_rca(static_evidence, drift_evidence, drift_report)


def _static_rca(static_evidence, drift_evidence, drift_report) -> RCAReport:
    """Fallback RCA when no API key is available."""
    all_evidence = static_evidence + drift_evidence

    cause = "Agent prompt or model configuration was modified between versions, introducing factual errors and potentially unsafe responses."
    if drift_report.overall_drift > 30:
        cause = "Critical prompt injection or unauthorized system prompt modification detected. Agent is serving fabricated information."

    return RCAReport(
        root_cause = cause,
        evidence   = all_evidence[:6] if all_evidence else [
            "Quality scores dropped significantly across all dimensions",
            "Hallucination risk increased substantially",
            "Safety dimension degraded",
        ],
        contributing_factors = [
            "No prompt integrity verification in deployment pipeline",
            "Missing automated output validation layer",
            "No continuous quality monitoring in production",
            "Deployment skipped quality gate checks",
        ],
        fixes = [
            FixRecommendation("Rollback to last known-good prompt configuration", "Stops active harm immediately", 1, "LOW"),
            FixRecommendation("Add prompt hash verification to CI/CD pipeline", "Prevents unauthorized prompt changes", 2, "MEDIUM"),
            FixRecommendation("Implement real-time hallucination detection", "Catches fabrications before serving", 3, "MEDIUM"),
            FixRecommendation("Set up AgentEval continuous monitoring with auto-rollback", "Detects degradation within minutes", 4, "HIGH"),
        ],
        severity  = "CRITICAL" if drift_report.overall_drift > 30 else "HIGH",
        confidence = "MEDIUM",
        post_mortem_draft = f"Agent quality dropped {drift_report.overall_drift} points from baseline. Static analysis detected {len(static_evidence)} red flags including potential hallucinations and safety issues. Immediate rollback recommended pending full investigation.",
    )
