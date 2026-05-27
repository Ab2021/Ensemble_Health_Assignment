"""
GenAI Explanation Engine using Ollama Cloud (gemma4:31b-cloud).

Architecture:
  1. Build ExplanationRequest (Pydantic) from claim data + risk factors
  2. Render structured prompt via build_explanation_prompt()
  3. Call ollama.chat() with gemma4:31b-cloud
  4. Parse JSON response -> validate with ExplanationResponse Pydantic model
  5. Fall back to deterministic template if API fails or validation fails

Audit & Observability:
  - Every call logged via LLMAuditLogger (token counts, latency, quality)
  - Raw API responses stored with full response object metadata
  - Quality validation: disclaimer, action, length, no hallucinations
"""
import os
import json
import time
import uuid
import ollama
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Any

# Load credentials from .env (one level up from src/)
_ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)

# Set ollama cloud host (the library reads OLLAMA_HOST from env)
os.environ.setdefault('OLLAMA_HOST', 'https://api.ollama.com')

from src.prompts.templates import (
    ExplanationRequest, ExplanationResponse,
    RiskFactorItem, build_explanation_prompt, build_fallback_response,
)
from src.llm_audit import (
    LLMAuditLogger, LLMCallRecord, build_audit_record,
    extract_token_counts, extract_latency, validate_response_quality,
)

# -- Configuration ------------------------------------
MODEL = os.environ.get('OLLAMA_MODEL', 'gemma4:31b-cloud')
API_KEY = os.environ.get('OLLAMA_API_KEY')

# -- Risk Factor -> Action Mapping --------------------
ACTION_MAP = {
    'Missing Required Prior Authorization':
        'Confirm whether the required prior authorization can be obtained and documented before submission.',
    'Missing Required Referral':
        'Obtain the required referral from the referring provider and attach it before submission.',
    'Missing Supporting Documentation':
        'Review and attach all required supporting clinical documentation before submission.',
    'Patient Eligibility Not Verified':
        'Run an eligibility verification check with the payer before submission.',
    'Provider Not in Payer Network':
        'Verify network status and confirm whether a network exception or prior authorization exists.',
    'Late Claim Submission (>30 days)':
        'Confirm the payer timely-filing deadline and gather any proof of original timely submission.',
    'Extended Days to Submit':
        'Expedite preparation and submission to avoid approaching timely-filing limits.',
    'High Claim Billed Amount':
        'Ensure all billed services are well documented and appropriately coded.',
    'Low Expected Payment Ratio':
        'Review the contract rate and verify coding accuracy for the expected payment level.',
    'High Claim Complexity':
        'Carefully review procedure-diagnosis alignment and ensure comprehensive documentation.',
    'Multiple Administrative Gaps':
        'Systematically resolve each administrative gap (authorization, documentation, eligibility, referral) before submission.',
    'Out-of-Network + Missing Authorization':
        'Verify network status simultaneously with confirming authorization availability before submission.',
    'Out-of-Network + Missing Referral':
        'Confirm network status and obtain the required referral before submission.',
    'Compound deficit: missing auth + documentation':
        'Resolve the missing authorization and attach all required documentation before submission.',
    'High-risk payer & visit combination':
        'Perform an enhanced pre-submission review due to the high-risk payer and visit type combination.',
}


def _map_action(factor: str) -> str:
    """Map a risk factor string to its recommended corrective action."""
    return ACTION_MAP.get(
        factor,
        'Review the identified risk factors and take appropriate corrective action before submission.'
    )


# =======================================================
# Prompt Building
# =======================================================

def build_payload(
    claim_row_dict: dict,
    denial_prob: float,
    risk_factor_list: List[str]
) -> ExplanationRequest:
    """Build a validated ExplanationRequest from claim data and model output.

    Args:
        claim_row_dict: Single claim row as dict (from DataFrame)
        denial_prob: Model-estimated denial probability [0, 1]
        risk_factor_list: List of risk factor strings

    Returns:
        Pydantic-validated ExplanationRequest
    """
    factors_payload = []
    for factor in risk_factor_list:
        factors_payload.append(RiskFactorItem(
            factor=factor,
            fact=factor,
            permitted_action=_map_action(factor)
        ))

    risk_label = (
        'High' if denial_prob >= 0.5
        else 'Medium' if denial_prob >= 0.25
        else 'Low'
    )

    return ExplanationRequest(
        claim_id=str(claim_row_dict.get('claim_id', 'UNKNOWN')),
        denial_probability=round(float(denial_prob), 3),
        risk_estimate_label=risk_label,
        top_risk_factors=factors_payload,
    )


# =======================================================
# API Call
# =======================================================

def _call_ollama(prompt_text: str):
    """Call Ollama Cloud chat API with the given prompt.

    Uses ollama.chat() per the user-specified API pattern.
    Authentication via OLLAMA_API_KEY environment variable.

    Returns:
        (raw_response_text, ollama_response_object) tuple.
        text is None on failure; response_obj is the ChatResponse for token/latency extraction.
    """
    if not API_KEY:
        print("[Ollama] No API key found in environment; skipping API call")
        return None, None

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt_text}],
        )
        text = (response.message.content or '').strip()
        return (text if text else None), response
    except Exception as e:
        print(f"[Ollama API warning] {type(e).__name__}: {e}")
        return None, None


def generate_explanation_api(payload: ExplanationRequest, audit_logger: Optional[LLMAuditLogger] = None):
    """Call Ollama Cloud and parse/validate the JSON response.

    Pipeline:
      prompt -> ollama.chat() -> JSON parse -> Pydantic validation -> plain text

    Returns:
        (explanation_text, audit_record) tuple.
        explanation_text is None if all steps fail.
    """
    prompt_text = build_explanation_prompt(payload)
    raw, response_obj = _call_ollama(prompt_text)

    # Build audit record for this call
    call_id = f"{payload.claim_id}_{uuid.uuid4().hex[:8]}"
    factors_str = [f.factor for f in payload.top_risk_factors]

    if not raw:
        record = build_audit_record(
            call_id=call_id, claim_id=payload.claim_id, model=MODEL,
            prompt_text=prompt_text, denial_prob=payload.denial_probability,
            risk_tier=payload.risk_estimate_label, risk_factors=factors_str,
            response_text=None, is_from_api=(response_obj is not None),
            fallback_reason='API returned empty or failed',
            ollama_response_obj=response_obj,
            api_error='Empty API response' if response_obj else 'API call failed',
        )
        if audit_logger:
            audit_logger.log_call(record)
        return None, record

    # -- Try JSON parse + Pydantic validation ------
    json_ok = False
    pydantic_ok = False
    val_errors = []
    
    try:
        # Strip markdown code fences if present (common with gemma models)
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            lines = lines[1:] if len(lines) > 1 and lines[0].startswith('```') else lines
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned = '\n'.join(lines).strip()
        parsed = json.loads(cleaned)
        json_ok = True
        validated = ExplanationResponse.model_validate(parsed)
        pydantic_ok = True
        text = validated.to_plain_text()
        
        record = build_audit_record(
            call_id=call_id, claim_id=payload.claim_id, model=MODEL,
            prompt_text=prompt_text, denial_prob=payload.denial_probability,
            risk_tier=payload.risk_estimate_label, risk_factors=factors_str,
            response_text=text, is_from_api=True,
            ollama_response_obj=response_obj,
            json_parse_success=True, pydantic_validation_success=True,
        )
        if audit_logger:
            audit_logger.log_call(record)
        return text, record
        
    except (json.JSONDecodeError, ValueError) as e:
        val_errors.append(str(e)[:200])
        
        # JSON failed; try extracting JSON from within the raw text
        import re
        json_match = re.search(r'\{[^{}]*"claim_id"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                json_ok = True
                validated = ExplanationResponse.model_validate(parsed)
                pydantic_ok = True
                text = validated.to_plain_text()
                
                record = build_audit_record(
                    call_id=call_id, claim_id=payload.claim_id, model=MODEL,
                    prompt_text=prompt_text, denial_prob=payload.denial_probability,
                    risk_tier=payload.risk_estimate_label, risk_factors=factors_str,
                    response_text=text, is_from_api=True,
                    ollama_response_obj=response_obj,
                    json_parse_success=True, pydantic_validation_success=True,
                    validation_errors=['Regex extraction after JSON parse failure'],
                )
                if audit_logger:
                    audit_logger.log_call(record)
                return text, record
            except (json.JSONDecodeError, ValueError) as e2:
                val_errors.append(f'Regex fallback: {str(e2)[:200]}')
        
        # Last resort: use deterministic fallback
        fallback = build_fallback_response(payload)
        
        record = build_audit_record(
            call_id=call_id, claim_id=payload.claim_id, model=MODEL,
            prompt_text=prompt_text, denial_prob=payload.denial_probability,
            risk_tier=payload.risk_estimate_label, risk_factors=factors_str,
            response_text=fallback.to_plain_text(), is_from_api=True,
            fallback_reason=f'JSON/Pydantic validation failed: {"; ".join(val_errors)}',
            ollama_response_obj=response_obj,
            json_parse_success=json_ok, pydantic_validation_success=pydantic_ok,
            validation_errors=val_errors,
        )
        if audit_logger:
            audit_logger.log_call(record)
        return fallback.to_plain_text(), record


# =======================================================
# Unified Generation
# =======================================================

def generate_explanation(
    claim_row_dict: dict,
    denial_prob: float,
    risk_factor_list: List[str],
    use_api: bool = True,
    audit_logger: Optional[LLMAuditLogger] = None,
) -> Tuple[str, ExplanationRequest]:
    """Unified explanation generation -- API with fallback to deterministic template.

    Decision logic:
      - Low-risk claims (prob < 0.25 or 'No actionable' factors) -> offline template
      - High-risk claims -> try API first, fall back to template on failure

    Args:
        claim_row_dict: Single claim as dict
        denial_prob: Denial probability [0, 1]
        risk_factor_list: List of human-readable risk factors
        use_api: If False, skip API and use template directly
        audit_logger: Optional auditor (logged even for bypassed/low-risk calls)

    Returns:
        (explanation_text, payload) tuple
    """
    payload = build_payload(claim_row_dict, denial_prob, risk_factor_list)

    # Bypass API for low-risk / no-flags claims
    has_no_flags = (
        not risk_factor_list
        or any('No actionable' in f or 'No major' in f for f in risk_factor_list)
    )
    if has_no_flags or denial_prob < 0.25:
        fallback = build_fallback_response(payload)
        # Audit the bypass too
        if audit_logger:
            prompt_text = build_explanation_prompt(payload)
            factors_str = [f.factor for f in payload.top_risk_factors]
            record = build_audit_record(
                call_id=f"{payload.claim_id}_bypass_{uuid.uuid4().hex[:8]}",
                claim_id=payload.claim_id, model=MODEL,
                prompt_text=prompt_text, denial_prob=denial_prob,
                risk_tier=payload.risk_estimate_label, risk_factors=factors_str,
                response_text=fallback.to_plain_text(), is_from_api=False,
                fallback_reason='Low-risk bypass (prob < 0.25 or no actionable flags)',
            )
            audit_logger.log_call(record)
        return fallback.to_plain_text(), payload

    text = None
    if use_api:
        result = generate_explanation_api(payload, audit_logger=audit_logger)
        text, _ = result

    if not text:
        fallback = build_fallback_response(payload)
        text = fallback.to_plain_text()
        # Audit the template fallback for non-API path (Medium tier, etc.)
        if audit_logger and not use_api:
            prompt_text = build_explanation_prompt(payload)
            factors_str = [f.factor for f in payload.top_risk_factors]
            record = build_audit_record(
                call_id=f"{payload.claim_id}_template_{uuid.uuid4().hex[:8]}",
                claim_id=payload.claim_id, model=MODEL,
                prompt_text=prompt_text, denial_prob=denial_prob,
                risk_tier=payload.risk_estimate_label, risk_factors=factors_str,
                response_text=text, is_from_api=False,
                fallback_reason='Deterministic template (use_api=False, prob >= 0.25)',
            )
            audit_logger.log_call(record)

    return text, payload
