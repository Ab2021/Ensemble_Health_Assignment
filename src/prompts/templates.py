"""
Structured prompt templates with Pydantic validation
for the GenAI explanation engine.

Prompt flow:
  1. Build ExplanationRequest (Pydantic model) from claim data + risk factors
  2. Render prompt string via build_explanation_prompt()
  3. LLM returns JSON matching ExplanationResponse schema
  4. Validate with ExplanationResponse.model_validate_json()
  5. Render final text via ExplanationResponse.to_plain_text()
"""
from pydantic import BaseModel, Field, field_validator
from typing import List


# =======================================================
# Pydantic Models
# =======================================================

class RiskFactorItem(BaseModel):
    """A single risk factor with its factual description and recommended action."""
    factor: str = Field(
        ..., min_length=1,
        description="Human-readable risk factor name (e.g. 'Missing Required Prior Authorization')"
    )
    fact: str = Field(
        ..., min_length=1,
        description="Concise factual statement about this risk"
    )
    permitted_action: str = Field(
        ..., min_length=1,
        description="Recommended corrective action before claim submission"
    )


class ExplanationRequest(BaseModel):
    """Structured input payload sent to the LLM.

    This is built from a claim row + model scores before the API call.
    Every field is validated before the prompt string is generated.
    """
    claim_id: str = Field(..., description="Claim identifier")
    denial_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model-estimated probability of denial (0-1)"
    )
    risk_estimate_label: str = Field(
        ..., pattern=r'^(High|Medium|Low)$',
        description="Risk tier label"
    )
    top_risk_factors: List[RiskFactorItem] = Field(
        ..., min_length=0, max_length=5,
        description="Top contributing risk factors (0 = no actionable flags)"
    )


class ExplanationResponse(BaseModel):
    """Structured JSON response expected from the LLM.

    After the LLM returns JSON, we validate with this model.
    If validation fails, we fall back to the deterministic template.
    """
    claim_id: str = Field(..., description="Must match the input claim_id")
    disclaimer: str = Field(
        ...,
        min_length=10,
        description="One sentence: states this is a statistical estimate, NOT a guaranteed denial"
    )
    risk_description: str = Field(
        ...,
        min_length=5,
        description="ONE concise sentence describing the primary pre-submission risk"
    )
    recommended_action: str = Field(
        ...,
        min_length=5,
        description="ONE concise sentence with the single most important corrective action"
    )

    @field_validator('disclaimer')
    @classmethod
    def must_qualify_uncertainty(cls, v: str) -> str:
        """Ensure the disclaimer includes uncertainty-qualifying language."""
        keywords = ['estimate', 'not a guarantee', 'statistical', 'may not',
                     'does not guarantee', 'not guaranteed', 'probability']
        if not any(kw.lower() in v.lower() for kw in keywords):
            raise ValueError(
                f'Disclaimer must include uncertainty qualifier (e.g. "statistical estimate"). '
                f'Got: "{v[:80]}..."'
            )
        return v.strip()

    @field_validator('claim_id')
    @classmethod
    def claim_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('claim_id must not be empty')
        return v.strip()

    def to_plain_text(self) -> str:
        """Render the validated response as a single plain-English paragraph.

        Returns a single string suitable for the CSV 'explanation' column.
        """
        parts = [p for p in [self.disclaimer, self.risk_description, self.recommended_action] if p]
        return ' '.join(parts)


# =======================================================
# Prompt Building
# =======================================================

def build_explanation_prompt(request: ExplanationRequest) -> str:
    """Build the full LLM prompt from a validated ExplanationRequest.

    The prompt instructs the LLM to return ONLY a JSON object matching
    the ExplanationResponse schema. This allows us to validate the output
    with Pydantic before using it.

    Args:
        request: Pydantic-validated ExplanationRequest

    Returns:
        Full prompt string ready to send to the LLM
    """
    # Build risk factors section
    if request.top_risk_factors:
        factors_lines = []
        for i, f in enumerate(request.top_risk_factors, 1):
            factors_lines.append(f"  {i}. {f.factor}")
            factors_lines.append(f"     Fact: {f.fact}")
            factors_lines.append(f"     Recommended action: {f.permitted_action}")
        factors_text = "\n".join(factors_lines)
    else:
        factors_text = "  (No actionable pre-submission risk flags detected)"

    prompt = f"""You are a revenue-cycle billing analyst writing pre-submission review notes.

Write a plain-English explanation for the following claim.
Return ONLY a valid JSON object. No other text before or after the JSON.
The JSON must have exactly these keys (all strings):

- "claim_id": "{request.claim_id}"
- "disclaimer": One sentence stating this is a statistical estimate, NOT a guaranteed denial outcome.
- "risk_description": One concise sentence describing the primary risk.
- "recommended_action": One concise sentence giving the single most actionable corrective step.

RULES:
- Use ONLY the risk facts and actions listed below. Do not invent new risks.
- Do NOT mention ICD/CPT codes, dollar amounts, or patient identifiers.
- Do NOT claim any action guarantees payment.
- The disclaimer MUST include qualifying language like "estimated" or "statistical".

Claim ID: {request.claim_id}
Estimated Denial Probability: {request.denial_probability}
Risk Tier: {request.risk_estimate_label}
Top Risk Factors:
{factors_text}

JSON response:"""
    return prompt


# =======================================================
# Deterministic Offline Fallback
# =======================================================

def build_fallback_response(request: ExplanationRequest) -> ExplanationResponse:
    """Build a deterministic ExplanationResponse when the API is unavailable
    or when the LLM response fails Pydantic validation.

    This ensures 100% reproducibility even in offline/CI environments.
    """
    prob_pct = f"{request.denial_probability:.0%}"

    disclaimer = (
        f"This claim has an estimated denial risk of {prob_pct} - "
        f"this is a statistical estimate, not a guaranteed outcome."
    )

    if not request.top_risk_factors or any(
        'no actionable' in (f.factor or f.fact).lower()
        for f in request.top_risk_factors
    ):
        risk_desc = (
            "No actionable pre-submission risk flags are evident "
            "based on available data."
        )
        action = "Routine submission with standard verification is recommended."
    else:
        top = request.top_risk_factors[0]
        risk_desc = f"A key concern is that {top.fact.lower().rstrip('.')}."
        action = f"Recommended action: {top.permitted_action}"

    return ExplanationResponse(
        claim_id=request.claim_id,
        disclaimer=disclaimer,
        risk_description=risk_desc,
        recommended_action=action,
    )
