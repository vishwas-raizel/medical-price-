import io
import json
from datetime import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_prediction_pdf(record_data: Dict[str, Any]) -> io.BytesIO:
    """
    Generates a PDF Medical Insurance Risk Underwriting & Price Prediction Report.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#4b5563'),
        spaceAfter=15
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#1f2937'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#374151'),
        leading=14
    )
    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#6b7280'),
        leading=10
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("🏥 Medical Insurance Predictive Risk Assessment", title_style))
    story.append(Paragraph(f"Official Underwriting Intelligence Report | Generated: {datetime.utcnow().strftime('%B %d, %Y - %H:%M UTC')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceAfter=15))

    # 2. Key Metrics Card Table
    pred_annual = record_data.get('predicted_charges', 0.0)
    pred_monthly = record_data.get('monthly_charges', pred_annual / 12.0)
    risk_tier = record_data.get('risk_tier', 'Moderate Risk')
    conf_lower = record_data.get('confidence_lower', pred_annual * 0.85)
    conf_upper = record_data.get('confidence_upper', pred_annual * 1.15)

    summary_data = [
        [
            Paragraph("<b>Estimated Annual Cost</b>", bold_body),
            Paragraph("<b>Estimated Monthly Cost</b>", bold_body),
            Paragraph("<b>Actuarial Risk Tier</b>", bold_body)
        ],
        [
            Paragraph(f"<font size='16' color='#1e3a8a'><b>${pred_annual:,.2f}</b></font>", body_style),
            Paragraph(f"<font size='14' color='#2563eb'><b>${pred_monthly:,.2f}</b></font>", body_style),
            Paragraph(f"<font size='12' color='#dc2626'><b>{risk_tier}</b></font>", body_style)
        ],
        [
            Paragraph(f"90% Conformal Interval: <b>${conf_lower:,.2f}</b> – <b>${conf_upper:,.2f}</b>", disclaimer_style),
            Paragraph("Empirical non-conformity calibration", disclaimer_style),
            Paragraph("Multi-factorial assessment", disclaimer_style)
        ]
    ]

    summary_table = Table(summary_data, colWidths=[180, 180, 180])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # 3. Applicant Profile
    story.append(Paragraph("Beneficiary Demographic & Clinical Profile", section_heading))
    applicant_data = [
        [
            Paragraph("<b>Age:</b>", bold_body), str(record_data.get('age', 'N/A')),
            Paragraph("<b>Sex:</b>", bold_body), str(record_data.get('sex', 'N/A')).capitalize()
        ],
        [
            Paragraph("<b>BMI:</b>", bold_body), f"{record_data.get('bmi', 0.0):.1f} kg/m²",
            Paragraph("<b>Dependents:</b>", bold_body), str(record_data.get('children', 0))
        ],
        [
            Paragraph("<b>Tobacco Smoker:</b>", bold_body), str(record_data.get('smoker', 'N/A')).capitalize(),
            Paragraph("<b>Residential Region:</b>", bold_body), str(record_data.get('region', 'N/A')).capitalize()
        ]
    ]
    app_table = Table(applicant_data, colWidths=[120, 150, 120, 150])
    app_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(app_table)
    story.append(Spacer(1, 15))

    # 4. SHAP Feature Attribution Table
    story.append(Paragraph("Explainable AI: Key Cost-Influencing Factors (SHAP Analysis)", section_heading))
    shap_raw = record_data.get('shap_contributions')
    if isinstance(shap_raw, str):
        try:
            shap_list = json.loads(shap_raw)
        except Exception:
            shap_list = []
    elif isinstance(shap_raw, list):
        shap_list = shap_raw
    else:
        shap_list = []

    shap_table_data = [
        [
            Paragraph("<b>Feature</b>", bold_body),
            Paragraph("<b>Observed Value</b>", bold_body),
            Paragraph("<b>Price Impact ($)</b>", bold_body),
            Paragraph("<b>Influence Direction</b>", bold_body)
        ]
    ]

    for item in shap_list[:6]:
        direction_color = '#dc2626' if item.get('impact_direction') == 'increases_cost' else '#16a34a'
        sign = "+" if item.get('impact_direction') == 'increases_cost' else ""
        shap_table_data.append([
            Paragraph(item.get('feature', 'N/A'), body_style),
            str(item.get('feature_value', 'N/A')),
            Paragraph(f"<font color='{direction_color}'><b>{sign}${item.get('contribution_amount', 0.0):,.2f}</b></font>", body_style),
            Paragraph(item.get('impact_direction', '').replace('_', ' ').capitalize(), body_style)
        ])

    if len(shap_table_data) > 1:
        shap_table = Table(shap_table_data, colWidths=[150, 110, 140, 140])
        shap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(shap_table)
    story.append(Spacer(1, 15))

    # 5. Preventative Guidance
    recs_raw = record_data.get('recommendations')
    if isinstance(recs_raw, str):
        try:
            recs_list = json.loads(recs_raw)
        except Exception:
            recs_list = []
    elif isinstance(recs_raw, list):
        recs_list = recs_raw
    else:
        recs_list = []

    if recs_list:
        story.append(Paragraph("Clinical Risk Mitigation & Preventative Guidance", section_heading))
        for rec in recs_list:
            story.append(Paragraph(f"• {rec}", body_style))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 10))

    # 6. Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#9ca3af'), spaceAfter=8))
    story.append(Paragraph(
        "<b>LEGAL & CLINICAL DISCLAIMER:</b> This predictive assessment is generated by machine learning statistical models "
        "trained on historical healthcare cost datasets. It is designed for educational analytics and actuarial reference, "
        "not as an official binding commercial insurance quote or clinical diagnostic evaluation.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer
