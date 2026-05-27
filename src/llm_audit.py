"""
LLM Audit Module -- Production-grade GenAI observability.

Tracks every Ollama API call with:
  - Token counts (prompt, completion, total)
  - Latency (total duration, eval duration)
  - Raw request/response JSON
  - Validation results (Pydantic pass/fail, fallback reason)
  - Quality checks (content length, disclaimer presence, JSON validity)

Output: JSON audit logs stored in data/output/audit_logs/
"""
import os
import json
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class LLMCallRecord:
    """Complete audit record for a single LLM API call."""
    # Identifiers
    call_id: str
    timestamp: str
    claim_id: str
    model: str

    # Input
    prompt_length_chars: int
    prompt_hash: str
    denial_probability: float
    risk_tier: str
    risk_factors: List[str]

    # Response
    response_length_chars: int
    response_text: str
    is_from_api: bool
    fallback_reason: Optional[str] = None

    # Token counts (from Ollama response)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Latency
    total_duration_ms: Optional[int] = None
    eval_duration_ms: Optional[int] = None

    # Validation
    json_parse_success: bool = False
    pydantic_validation_success: bool = False
    validation_errors: List[str] = field(default_factory=list)

    # Quality checks
    has_disclaimer: bool = False
    has_action: bool = False
    min_length_ok: bool = False
    quality_warnings: List[str] = field(default_factory=list)

    # Error
    api_error: Optional[str] = None


class LLMAuditLogger:
    """Accumulates audit records and flushes to disk.

    Usage:
        logger = LLMAuditLogger(output_dir='data/output/audit_logs')
        logger.log_call(record)
        logger.flush()
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records: List[LLMCallRecord] = []
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

    def log_call(self, record: LLMCallRecord):
        self.records.append(record)

    def flush(self):
        """Write all accumulated records to a structured JSON file."""
        path = os.path.join(self.output_dir, f'audit_{self.run_id}.json')
        payload = {
            'run_id': self.run_id,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_calls': len(self.records),
            'api_calls': sum(1 for r in self.records if r.is_from_api),
            'fallbacks': sum(1 for r in self.records if r.fallback_reason),
            'token_summary': self.token_summary,
            'quality_summary': self.quality_summary,
            'records': [asdict(r) for r in self.records],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    @property
    def token_summary(self) -> Dict[str, int]:
        """Aggregate token usage across all calls."""
        total_prompt = sum(r.prompt_tokens or 0 for r in self.records)
        total_completion = sum(r.completion_tokens or 0 for r in self.records)
        total_all = sum(r.total_tokens or 0 for r in self.records)
        return {
            'total_prompt_tokens': total_prompt,
            'total_completion_tokens': total_completion,
            'total_tokens': total_all,
        }

    @property
    def quality_summary(self) -> Dict[str, Any]:
        """Aggregate quality metrics."""
        if not self.records:
            return {'error': 'No records'}
        api_records = [r for r in self.records if r.is_from_api]
        return {
            'total_calls': len(self.records),
            'api_success_rate': len(api_records) / len(self.records) if self.records else 0,
            'json_parse_rate': sum(1 for r in api_records if r.json_parse_success) / len(api_records) if api_records else 0,
            'pydantic_pass_rate': sum(1 for r in api_records if r.pydantic_validation_success) / len(api_records) if api_records else 0,
            'disclaimer_rate': sum(1 for r in self.records if r.has_disclaimer) / len(self.records),
            'avg_latency_ms': sum(r.total_duration_ms or 0 for r in api_records) / len(api_records) if api_records else 0,
        }


# =======================================================
# Token counting (from Ollama response object)
# =======================================================

def extract_token_counts(response_obj) -> Dict[str, Optional[int]]:
    """Extract token counts from an ollama ChatResponse object.

    Args:
        response_obj: Instance of ollama._types.ChatResponse

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens
    """
    prompt = getattr(response_obj, 'prompt_eval_count', None)
    completion = getattr(response_obj, 'eval_count', None)
    total = (prompt or 0) + (completion or 0) if (prompt is not None and completion is not None) else None
    return {
        'prompt_tokens': prompt,
        'completion_tokens': completion,
        'total_tokens': total,
    }


def extract_latency(response_obj) -> Dict[str, Optional[int]]:
    """Extract latency metrics from the response (nanoseconds -> milliseconds)."""
    total_ns = getattr(response_obj, 'total_duration', None)
    eval_ns = getattr(response_obj, 'eval_duration', None)
    return {
        'total_duration_ms': int(total_ns / 1_000_000) if total_ns else None,
        'eval_duration_ms': int(eval_ns / 1_000_000) if eval_ns else None,
    }


# =======================================================
# Response validation
# =======================================================

def validate_response_quality(text: str, denial_prob: float) -> Dict[str, Any]:
    """Run quality checks on an LLM-generated explanation.

    Returns:
        Dict with pass/fail results and any warnings
    """
    checks = {
        'has_disclaimer': False,
        'has_action': False,
        'min_length_ok': False,
        'quality_warnings': [],
    }

    if not text:
        checks['quality_warnings'].append('Empty response text')
        return checks

    text_lower = text.lower()

    # Disclaimer check -- must contain uncertainty language
    uncertainty_keywords = ['estimate', 'statistical', 'not guaranteed', 'not a guarantee']
    checks['has_disclaimer'] = any(kw in text_lower for kw in uncertainty_keywords)
    if not checks['has_disclaimer']:
        checks['quality_warnings'].append('No uncertainty disclaimer found')

    # Action check -- must suggest a corrective action
    action_keywords = ['recommend', 'verify', 'obtain', 'review', 'attached', 'confirm',
                        'resolve', 'correct', 'submit', 'ensure', 'check']
    checks['has_action'] = any(kw in text_lower for kw in action_keywords)
    if not checks['has_action']:
        checks['quality_warnings'].append('No corrective action suggestion found')

    # Minimum length check
    checks['min_length_ok'] = len(text) >= 50
    if not checks['min_length_ok']:
        checks['quality_warnings'].append(f'Response too short: {len(text)} chars')

    # Additional quality: check for hallucination markers
    hallucination_markers = ['ICD-', 'CPT', 'diagnosis code', 'procedure code',
                              'patient name', 'SSN', 'DOB', 'date of birth']
    for marker in hallucination_markers:
        if marker.lower() in text_lower:
            checks['quality_warnings'].append(f'Potential PII/hallucination: "{marker}"')

    return checks


# =======================================================
# Full audit call wrapper
# =======================================================

def build_audit_record(
    call_id: str,
    claim_id: str,
    model: str,
    prompt_text: str,
    denial_prob: float,
    risk_tier: str,
    risk_factors: List[str],
    response_text: Optional[str] = None,
    is_from_api: bool = False,
    fallback_reason: Optional[str] = None,
    ollama_response_obj=None,
    json_parse_success: bool = False,
    pydantic_validation_success: bool = False,
    validation_errors: Optional[List[str]] = None,
    api_error: Optional[str] = None,
) -> LLMCallRecord:
    """Build a complete audit record from an LLM call.

    Call this after every explanation generation (API or fallback).
    """

    # Token counts
    tokens = {}
    latency = {}
    if ollama_response_obj is not None:
        tokens = extract_token_counts(ollama_response_obj)
        latency = extract_latency(ollama_response_obj)

    # Quality
    quality = validate_response_quality(response_text or '', denial_prob)

    return LLMCallRecord(
        call_id=call_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        claim_id=claim_id,
        model=model,
        prompt_length_chars=len(prompt_text),
        prompt_hash=hashlib.md5(prompt_text.encode()).hexdigest()[:12],
        denial_probability=denial_prob,
        risk_tier=risk_tier,
        risk_factors=risk_factors,
        response_length_chars=len(response_text or ''),
        response_text=(response_text or '')[:500],  # Truncate for audit
        is_from_api=is_from_api,
        fallback_reason=fallback_reason,
        prompt_tokens=tokens.get('prompt_tokens'),
        completion_tokens=tokens.get('completion_tokens'),
        total_tokens=tokens.get('total_tokens'),
        total_duration_ms=latency.get('total_duration_ms'),
        eval_duration_ms=latency.get('eval_duration_ms'),
        json_parse_success=json_parse_success,
        pydantic_validation_success=pydantic_validation_success,
        validation_errors=validation_errors or [],
        has_disclaimer=quality['has_disclaimer'],
        has_action=quality['has_action'],
        min_length_ok=quality['min_length_ok'],
        quality_warnings=quality.get('quality_warnings', []),
        api_error=api_error,
    )
