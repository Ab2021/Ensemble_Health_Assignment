"""
Unit tests for src/prompts/templates.py (Pydantic models + prompt building)
and src/explanations.py (Ollama API integration).

Tests both:
  - Pydantic model validation (ExplanationRequest, ExplanationResponse)
  - Prompt string generation
  - Offline fallback behavior
  - Payload building and action mapping
"""
import pytest
import json
from pydantic import ValidationError
from src.prompts.templates import (
    RiskFactorItem,
    ExplanationRequest,
    ExplanationResponse,
    build_explanation_prompt,
    build_fallback_response,
)


# =======================================================
# RiskFactorItem
# =======================================================

class TestRiskFactorItem:
    def test_valid_item(self):
        item = RiskFactorItem(
            factor='Missing Auth',
            fact='Prior authorization is missing',
            permitted_action='Obtain authorization before submission',
        )
        assert item.factor == 'Missing Auth'

    def test_empty_factor_fails(self):
        with pytest.raises(ValidationError):
            RiskFactorItem(factor='', fact='test', permitted_action='act')

    def test_missing_field_fails(self):
        with pytest.raises(ValidationError):
            RiskFactorItem(factor='test', fact='test')
        # permitted_action is required


# =======================================================
# ExplanationRequest
# =======================================================

class TestExplanationRequest:
    def test_valid_request(self):
        req = ExplanationRequest(
            claim_id='CCLM-001',
            denial_probability=0.85,
            risk_estimate_label='High',
            top_risk_factors=[
                RiskFactorItem(factor='Missing Auth', fact='No auth on file',
                               permitted_action='Get auth before submission'),
            ],
        )
        assert req.claim_id == 'CCLM-001'
        assert req.denial_probability == 0.85

    def test_probability_out_of_range_fails(self):
        with pytest.raises(ValidationError):
            ExplanationRequest(
                claim_id='X', denial_probability=1.5,
                risk_estimate_label='High', top_risk_factors=[],
            )

    def test_probability_negative_fails(self):
        with pytest.raises(ValidationError):
            ExplanationRequest(
                claim_id='X', denial_probability=-0.1,
                risk_estimate_label='High', top_risk_factors=[],
            )

    def test_invalid_risk_label_fails(self):
        with pytest.raises(ValidationError):
            ExplanationRequest(
                claim_id='X', denial_probability=0.5,
                risk_estimate_label='Critical',  # not High/Medium/Low
                top_risk_factors=[],
            )

    def test_empty_factors_allowed(self):
        """No factors = low-risk claim -- should be valid."""
        req = ExplanationRequest(
            claim_id='CCLM-LOW', denial_probability=0.05,
            risk_estimate_label='Low', top_risk_factors=[],
        )
        assert len(req.top_risk_factors) == 0

    def test_max_five_factors(self):
        factors = [
            RiskFactorItem(factor=f'Risk {i}', fact=f'Fact {i}',
                           permitted_action=f'Action {i}')
            for i in range(6)
        ]
        with pytest.raises(ValidationError):
            ExplanationRequest(
                claim_id='X', denial_probability=0.5,
                risk_estimate_label='Medium', top_risk_factors=factors,
            )


# =======================================================
# ExplanationResponse
# =======================================================

class TestExplanationResponse:
    def test_valid_response(self):
        resp = ExplanationResponse(
            claim_id='CCLM-001',
            disclaimer='This is a statistical estimate, not a guaranteed denial.',
            risk_description='Missing prior authorization is the primary concern.',
            recommended_action='Obtain authorization before resubmitting.',
        )
        assert resp.disclaimer is not None

    def test_missing_disclaimer_keywords_fails(self):
        with pytest.raises(ValidationError) as exc:
            ExplanationResponse(
                claim_id='CCLM-001',
                disclaimer='The claim will be denied.',  # No uncertainty qualifier
                risk_description='Missing auth.',
                recommended_action='Get auth.',
            )
        assert 'Disclaimer must include' in str(exc.value)

    def test_disclaimer_with_estimate_passes(self):
        resp = ExplanationResponse(
            claim_id='X', disclaimer='This is an estimated denial risk.',
            risk_description='Risk.', recommended_action='Action.',
        )
        assert 'estimate' in resp.disclaimer

    def test_disclaimer_with_statistical_passes(self):
        resp = ExplanationResponse(
            claim_id='X', disclaimer='Statistical probability of denial.',
            risk_description='Risk.', recommended_action='Action.',
        )
        assert 'statistical' in resp.disclaimer.lower()

    def test_empty_claim_id_fails(self):
        with pytest.raises(ValidationError):
            ExplanationResponse(
                claim_id='   ', disclaimer='This is an estimate.',
                risk_description='Risk.', recommended_action='Action.',
            )

    def test_to_plain_text(self):
        resp = ExplanationResponse(
            claim_id='C001',
            disclaimer='This is a statistical estimate.',
            risk_description='Missing authorization.',
            recommended_action='Get authorization before submission.',
        )
        text = resp.to_plain_text()
        assert 'statistical estimate' in text
        assert 'Missing authorization' in text
        assert 'Get authorization' in text

    def test_json_roundtrip(self):
        """JSON serialization and deserialization should preserve data."""
        original = ExplanationResponse(
            claim_id='CCLM-001',
            disclaimer='This is a statistical estimate, not a guaranteed denial.',
            risk_description='Missing prior authorization is the primary risk.',
            recommended_action='Obtain prior authorization before submitting.',
        )
        json_str = original.model_dump_json()
        parsed = json.loads(json_str)
        restored = ExplanationResponse.model_validate(parsed)
        assert restored.claim_id == original.claim_id
        assert restored.disclaimer == original.disclaimer


# =======================================================
# Prompt Building
# =======================================================

class TestBuildExplanationPrompt:
    def test_returns_string(self):
        req = ExplanationRequest(
            claim_id='CCLM-TEST', denial_probability=0.75,
            risk_estimate_label='High',
            top_risk_factors=[
                RiskFactorItem(factor='Missing Auth', fact='No auth',
                               permitted_action='Get auth'),
            ],
        )
        prompt = build_explanation_prompt(req)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_includes_claim_id(self):
        req = ExplanationRequest(
            claim_id='CCLM-ABC', denial_probability=0.5,
            risk_estimate_label='Medium',
            top_risk_factors=[],
        )
        prompt = build_explanation_prompt(req)
        assert 'CCLM-ABC' in prompt

    def test_includes_risk_factors(self):
        req = ExplanationRequest(
            claim_id='X', denial_probability=0.9,
            risk_estimate_label='High',
            top_risk_factors=[
                RiskFactorItem(factor='Missing Auth', fact='No auth on file',
                               permitted_action='Obtain auth'),
            ],
        )
        prompt = build_explanation_prompt(req)
        assert 'Missing Auth' in prompt
        assert 'Obtain auth' in prompt

    def test_asks_for_json_response(self):
        req = ExplanationRequest(
            claim_id='X', denial_probability=0.5,
            risk_estimate_label='Medium',
            top_risk_factors=[],
        )
        prompt = build_explanation_prompt(req)
        assert 'JSON' in prompt
        assert 'claim_id' in prompt
        assert 'disclaimer' in prompt

    def test_empty_factors_shows_no_flags_message(self):
        req = ExplanationRequest(
            claim_id='X', denial_probability=0.1,
            risk_estimate_label='Low',
            top_risk_factors=[],
        )
        prompt = build_explanation_prompt(req)
        assert 'No actionable' in prompt


# =======================================================
# Offline Fallback
# =======================================================

class TestBuildFallbackResponse:
    def test_returns_explanation_response(self):
        req = ExplanationRequest(
            claim_id='CCLM-001', denial_probability=0.85,
            risk_estimate_label='High',
            top_risk_factors=[
                RiskFactorItem(factor='Missing Auth',
                               fact='Prior authorization is missing',
                               permitted_action='Get auth before submission'),
            ],
        )
        resp = build_fallback_response(req)
        assert isinstance(resp, ExplanationResponse)
        assert 'estimated denial risk' in resp.disclaimer.lower()
        assert '85%' in resp.disclaimer

    def test_no_flags_returns_routine_message(self):
        req = ExplanationRequest(
            claim_id='CCLM-LOW', denial_probability=0.05,
            risk_estimate_label='Low',
            top_risk_factors=[],
        )
        resp = build_fallback_response(req)
        assert 'No actionable' in resp.risk_description
        assert 'Routine submission' in resp.recommended_action

    def test_no_actionable_flag_in_factor(self):
        req = ExplanationRequest(
            claim_id='X', denial_probability=0.1,
            risk_estimate_label='Low',
            top_risk_factors=[
                RiskFactorItem(
                    factor='No actionable pre-submission risk flags detected',
                    fact='No actionable pre-submission risk flags detected',
                    permitted_action='Review before submission.',
                ),
            ],
        )
        resp = build_fallback_response(req)
        assert 'No actionable' in resp.risk_description

    def test_output_validates_as_explanation_response(self):
        """Fallback response must always pass Pydantic validation."""
        req = ExplanationRequest(
            claim_id='TEST', denial_probability=0.7,
            risk_estimate_label='High',
            top_risk_factors=[
                RiskFactorItem(factor='Risk', fact='Risk exists',
                               permitted_action='Fix risk'),
            ],
        )
        resp = build_fallback_response(req)
        # This should NOT raise -- the fallback is our code, it must validate
        ExplanationResponse.model_validate(resp.model_dump())


# =======================================================
# explanations.py integration (without API calls)
# =======================================================

class TestExplanationOffline:
    """Test the offline/template pathway without hitting Ollama API."""

    def test_low_probability_bypasses_api(self):
        from src.explanations import generate_explanation
        claim = {'claim_id': 'CCLM-LOW'}
        text, payload = generate_explanation(claim, 0.10, ['Some factor'], use_api=True)
        assert 'No actionable' in text or 'estimated' in text.lower()
        assert isinstance(payload, ExplanationRequest)

    def test_no_actionable_factors_bypasses_api(self):
        from src.explanations import generate_explanation
        claim = {'claim_id': 'CCLM-NOF'}
        text, payload = generate_explanation(
            claim, 0.60,  # high prob, but factor says no flags
            ['No actionable pre-submission risk flags detected'],
            use_api=True,
        )
        assert 'No actionable' in text

    def test_use_api_false_forces_template(self):
        from src.explanations import generate_explanation
        claim = {'claim_id': 'CCLM-TPL'}
        text, payload = generate_explanation(
            claim, 0.85,
            ['Missing Required Prior Authorization'],
            use_api=False,
        )
        assert 'estimated' in text.lower()
        assert 'Missing' in text or 'prior authorization' in text.lower()

    def test_payload_has_correct_structure(self):
        from src.explanations import build_payload
        claim = {'claim_id': 'CCLM-001'}
        payload = build_payload(claim, 0.75, ['Missing Auth', 'Late Filing'])
        assert payload.claim_id == 'CCLM-001'
        assert payload.denial_probability == 0.75
        assert len(payload.top_risk_factors) == 2
