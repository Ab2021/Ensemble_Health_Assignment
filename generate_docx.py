"""
Generate Project_documentation.docx -- Manager-facing business narrative.
Summarizes the denial prediction system for non-technical stakeholders.
"""
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import os

def set_cell_shading(cell, color_hex):
    """Apply background shading to a table cell."""
    shading_elm = cell._element.get_or_add_tcPr()
    shading = shading_elm.makeelement(qn('w:shd'), {
        qn('w:fill'): color_hex,
        qn('w:val'): 'clear',
    })
    shading_elm.append(shading)

def add_styled_table(doc, headers, rows, col_widths=None):
    """Add a formatted table with blue header row."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'

    # Header row
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, '2F5496')

    # Data rows
    for r_idx, row_data in enumerate(rows):
        for c_idx, value in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(value)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)

    doc.add_paragraph()
    return table


def generate_docx():
    doc = Document()

    # ---- Title ----
    title = doc.add_heading('Claim Denial Risk Prediction System', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run('Ensemble Health Partners -- AI Team Hiring Assessment')
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)

    doc.add_paragraph()

    # ---- 1. Executive Summary ----
    doc.add_heading('1. Executive Summary', level=1)
    doc.add_paragraph(
        'This project delivers a production-ready pre-bill denial prediction system that '
        'combines classical machine learning with a GenAI explanation engine. The system '
        'scores 500 current claims for denial risk, ranks them by priority, and generates '
        'plain-English corrective action recommendations for the highest-risk claims.'
    )
    doc.add_paragraph(
        'After evaluating 10 model variants across 5 algorithm families, a Calibrated Logistic '
        'Regression model was selected for production deployment. It captures 45.7% of all denials '
        'within the top 25% review window -- nearly double the 25% expected from random review -- '
        'and provides reliable probability estimates with a Brier score of 0.137.'
    )

    # ---- 2. Business Impact ----
    doc.add_heading('2. Business Impact', level=1)
    doc.add_paragraph(
        'Denied claims are one of the largest sources of revenue leakage in healthcare revenue '
        'cycle management. Each denied claim requires rework -- investigation, correction, '
        'resubmission, and follow-up -- costing $25-118 per claim in administrative overhead. '
        'Pre-bill denial prediction moves this intervention upstream: instead of reacting to '
        'denials after submission, billers can fix issues before the claim leaves the door.'
    )

    bullets = [
        '45.7% of denials captured in the top 25% highest-risk claims -- nearly 2x random review',
        '125 claims flagged as High risk for immediate biller attention',
        '10 claims receive AI-generated plain-English explanations with specific corrective actions',
        'GenAI explanations include uncertainty disclaimers to prevent over-reliance',
        'Full audit trail for every AI-generated explanation (tokens, latency, quality validation)',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    # ---- 3. Key Findings ----
    doc.add_heading('3. Key Findings', level=1)
    doc.add_paragraph(
        'The most predictive risk factors are administrative completeness checks: whether the '
        'claim has a required prior authorization, whether supporting documentation is attached, '
        'whether a required referral is present, and whether the provider is in-network. These '
        'are all actionable before submission, making the system operationally practical.'
    )
    doc.add_paragraph(
        'Ten model architectures were evaluated. Logistic Regression variants achieved the highest '
        'performance (49.51% capture on validation), while Gradient Boosting was the best non-linear '
        'alternative at 48.54%. The strong performance of linear models confirms that administrative '
        'gap flags have a direct, additive relationship with denial outcomes.'
    )

    # ---- 4. Model Selection Results ----
    doc.add_heading('4. Model Selection Results', level=1)
    doc.add_paragraph(
        'The table below shows the top 5 models ranked by denial capture rate within the '
        'top 25% review window. The Calibrated Logistic Regression model was selected for '
        'production because it matches the best capture rate while providing substantially '
        'more reliable probability estimates (34% improvement in calibration).'
    )

    model_headers = ['Model', 'Type', 'Capture@25%', 'ROC-AUC', 'Brier']
    model_rows = [
        ['Calibrated Logistic Regression', 'Linear', '49.51%', '0.710', '0.137 -- Best calibration'],
        ['Logistic Regression Baseline', 'Linear', '49.51%', '0.711', '0.209'],
        ['Gradient Boosting (GBM)', 'Tree Ensemble', '48.54%', '0.708', '0.165 -- Best non-linear'],
        ['Voting Ensemble (LR+RF+GBM)', 'Ensemble', '48.54%', '0.705', '0.172'],
        ['Random Forest', 'Tree Ensemble', '39.81%', '0.653', '0.155'],
    ]
    add_styled_table(doc, model_headers, model_rows)

    doc.add_heading('Final Test Set Performance', level=2)

    perf_headers = ['Metric', 'Value']
    perf_rows = [
        ['Model Deployed', 'Calibrated Logistic Regression'],
        ['Denial Capture at 25% Review', '45.7%'],
        ['Precision at 25% Review', '47.8%'],
        ['ROC-AUC', '0.691'],
        ['Brier Score (Calibrated)', '0.137'],
        ['High-Risk Claims Flagged', '125 of 500'],
        ['LLM Model', 'gemma4:31b-cloud (Ollama)'],
    ]
    add_styled_table(doc, perf_headers, perf_rows)

    # ---- 5. GenAI Explanation Engine ----
    doc.add_heading('5. GenAI Explanation Engine', level=1)
    doc.add_paragraph(
        'The system uses Ollama Cloud\'s gemma4:31b-cloud model to generate plain-English '
        'explanations for the 10 highest-risk claims. Each explanation contains three elements:'
    )
    doc.add_paragraph('A statistical disclaimer ("This is a statistical estimate and not a guaranteed outcome")', style='List Bullet')
    doc.add_paragraph('Identification of specific risk factors (missing authorization, missing documentation)', style='List Bullet')
    doc.add_paragraph('Actionable corrective steps (resolve, attach, verify, confirm)', style='List Bullet')

    doc.add_paragraph(
        'Every AI explanation is validated through a Pydantic schema before being accepted. '
        'The schema enforces disclaimer presence, minimum length, and structural completeness. '
        'If validation fails, a deterministic template is used as fallback. In the latest '
        'production run: 10 API calls, 0 fallbacks, 100% validation pass rate.'
    )

    # ---- 6. Safety & Auditability ----
    doc.add_heading('6. Safety & Auditability', level=1)
    doc.add_paragraph(
        'The system includes a production-grade LLM audit infrastructure that records every '
        'API call with full metadata: token usage (prompt + completion), latency, JSON parse '
        'success, Pydantic validation success, and quality flags (disclaimer presence, actionable '
        'content, hallucination markers). No patient data is sent to the LLM -- only engineered '
        'features and risk factor labels. All audit logs are stored locally in JSON format '
        'with no external telemetry.'
    )

    audit_headers = ['Capability', 'Details']
    audit_rows = [
        ['Token Tracking', 'Prompt tokens + completion tokens from Ollama ChatResponse'],
        ['Latency Logging', 'Total duration and evaluation duration per call'],
        ['Response Validation', 'JSON parse success, Pydantic schema validation'],
        ['Quality Checks', 'Disclaimer, action verbs, min length, PII/hallucination scan'],
        ['Storage', 'Local JSON audit logs in data/output/audit_logs/'],
        ['HIPAA Alignment', 'No external telemetry, no PII in prompts, response truncation'],
    ]
    add_styled_table(doc, audit_headers, audit_rows)

    # ---- 7. Limitations & Recommendations ----
    doc.add_heading('7. Limitations & Recommendations', level=1)

    doc.add_heading('Current Limitations', level=2)
    limits = [
        'Synthetic training data: Real claims would include ICD-10/CPT codes and 835 remittance data, significantly improving prediction accuracy.',
        'Single global model: Different payers have distinct denial patterns; payer-specific sub-models may improve performance.',
        'Small dataset: 3,200 claims is adequate for linear models but insufficient for deep learning approaches.',
        'Binary outcome only: Does not distinguish administrative denials from medical necessity or coding denials.',
    ]
    for l in limits:
        doc.add_paragraph(l, style='List Bullet')

    doc.add_heading('Production Recommendations', level=2)
    recs = [
        'Deploy the Calibrated Logistic Regression model immediately for the 25% review workflow.',
        'Retrain monthly as new claims and denial outcomes become available.',
        'Monitor Brier score as an early warning for model drift.',
        'A/B test the flagged review workflow against current process to measure actual denial reduction.',
        'Incorporate ICD-10/CPT code features when real claims data becomes available.',
        'Build payer-specific sub-models for the highest-volume payers.',
    ]
    for r in recs:
        doc.add_paragraph(r, style='List Bullet')

    # ---- Footer ----
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run('Confidential -- Ensemble Health Partners Hiring Assessment -- May 2025')
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'Project_documentation.docx')
    doc.save(output_path)
    print(f'Saved: {output_path}')
    return output_path


if __name__ == '__main__':
    generate_docx()
