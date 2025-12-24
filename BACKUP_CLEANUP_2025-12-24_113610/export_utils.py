"""
Export utilities for generating PDF, Excel, and Word reports
Ported from Streamlit app with full feature parity
"""

import io
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

# Optional imports for export functionality
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# ============================================================================
# PDF PREVIEW UTILITIES
# ============================================================================

def render_pdf_to_images(pdf_bytes: bytes, max_pages: int = 10, dpi: int = 150) -> List[bytes]:
    """
    Render PDF pages as PNG images for web preview.
    Returns list of PNG image bytes (one per page).
    
    Args:
        pdf_bytes: PDF file as bytes
        max_pages: Maximum number of pages to render
        dpi: DPI for rendering (150 is good balance of quality/size)
    
    Returns:
        List of PNG image bytes, or empty list if PyMuPDF not available
    """
    if not PYMUPDF_AVAILABLE:
        return []
    
    try:
        images = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
            pages_to_render = min(total_pages, max_pages)
            
            for page_num in range(pages_to_render):
                page = doc[page_num]
                # Render at specified DPI (150 DPI = 150/72 scale factor)
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)
        
        return images
    except Exception as e:
        print(f"Error rendering PDF to images: {e}")
        return []


# ============================================================================
# EXCEL EXPORT - Coverage Matrix with Color Coding
# ============================================================================

def to_excel_coverage_matrix(
    coverage: pd.DataFrame, 
    cat_map: Dict[str, str], 
    hi: float = 0.75, 
    lo: float = 0.35
) -> Optional[bytes]:
    """
    Export coverage matrix to Excel with formatting and color coding.
    
    Creates a multi-sheet workbook:
    - Sheet 1: Summary (metrics + top 5 candidates)
    - Sheet 2: Coverage Matrix (transposed: criteria as rows, candidates as columns)
    - Sheet 3: Legend (score color coding explanation)
    
    Args:
        coverage: DataFrame with columns ['Candidate', 'Overall', ...criteria...]
        cat_map: Dict mapping criterion -> category
        hi: High threshold for color coding (green)
        lo: Low threshold for color coding (red)
    
    Returns:
        Bytes of Excel file or None if openpyxl not available
    """
    if not OPENPYXL_AVAILABLE:
        return None
    
    try:
        buf = io.BytesIO()
        
        # Create workbook with multiple sheets
        wb = openpyxl.Workbook()
        
        # Sheet 1: Summary
        ws_summary = wb.active
        ws_summary.title = "Summary"
        
        # Add title
        ws_summary['A1'] = "Candidate Analysis Summary"
        ws_summary['A1'].font = Font(size=16, bold=True, color="1F77B4")
        ws_summary.merge_cells('A1:D1')
        
        # Add metrics
        row = 3
        ws_summary[f'A{row}'] = "Total Candidates:"
        ws_summary[f'B{row}'] = len(coverage)
        ws_summary[f'A{row}'].font = Font(bold=True)
        
        row += 1
        crit_cols = [c for c in coverage.columns if c not in ('Candidate', 'Overall')]
        ws_summary[f'A{row}'] = "Total Criteria:"
        ws_summary[f'B{row}'] = len(crit_cols)
        ws_summary[f'A{row}'].font = Font(bold=True)
        
        row += 2
        ws_summary[f'A{row}'] = "Top 5 Candidates"
        ws_summary[f'A{row}'].font = Font(size=14, bold=True)
        row += 1
        
        # Top candidates table
        top5 = coverage[['Candidate', 'Overall']].head(5)
        ws_summary[f'A{row}'] = "Rank"
        ws_summary[f'B{row}'] = "Candidate"
        ws_summary[f'C{row}'] = "Overall Score"
        for col in ['A', 'B', 'C']:
            ws_summary[f'{col}{row}'].font = Font(bold=True)
            ws_summary[f'{col}{row}'].fill = PatternFill(start_color="E7F3FF", end_color="E7F3FF", fill_type="solid")
        
        for idx, (_, r) in enumerate(top5.iterrows(), 1):
            row += 1
            ws_summary[f'A{row}'] = idx
            ws_summary[f'B{row}'] = r['Candidate']
            ws_summary[f'C{row}'] = round(r['Overall'], 2)
            
            # Color code the score
            score = r['Overall']
            if score >= hi:
                ws_summary[f'C{row}'].fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
            elif score >= lo:
                ws_summary[f'C{row}'].fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
            else:
                ws_summary[f'C{row}'].fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        
        # Adjust column widths
        ws_summary.column_dimensions['A'].width = 8
        ws_summary.column_dimensions['B'].width = 30
        ws_summary.column_dimensions['C'].width = 15
        
        # Sheet 2: Coverage Matrix (TRANSPOSED)
        ws_matrix = wb.create_sheet("Coverage Matrix")
        
        # Get criteria columns (exclude Candidate and Overall)
        criteria_cols = [c for c in coverage.columns if c not in ('Candidate', 'Overall')]
        
        # Create transposed structure: Criteria as rows, Candidates as columns
        # Header row: "Criterion" then candidate names
        header_row = ['Criterion'] + coverage['Candidate'].tolist()
        
        # Write header
        for c_idx, value in enumerate(header_row, 1):
            cell = ws_matrix.cell(row=1, column=c_idx, value=value)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="1F77B4", end_color="1F77B4", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
        
        # Write data rows (one criterion per row)
        for r_idx, criterion in enumerate(criteria_cols, 2):
            # First column: criterion name
            cell = ws_matrix.cell(row=r_idx, column=1, value=criterion)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="E7F3FF", end_color="E7F3FF", fill_type="solid")
            cell.alignment = Alignment(horizontal="left")
            
            # Remaining columns: scores for each candidate
            for c_idx, (_, candidate_row) in enumerate(coverage.iterrows(), 2):
                score = candidate_row[criterion]
                cell = ws_matrix.cell(row=r_idx, column=c_idx, value=round(score, 2))
                
                # Color code the score
                if score >= hi:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif score >= lo:
                    cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                cell.alignment = Alignment(horizontal="center")
        
        # Auto-adjust column widths
        for column in ws_matrix.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws_matrix.column_dimensions[column_letter].width = adjusted_width
        
        # Sheet 3: Legend
        ws_legend = wb.create_sheet("Legend")
        ws_legend['A1'] = "Score Color Coding"
        ws_legend['A1'].font = Font(size=14, bold=True)
        
        ws_legend['A3'] = f"Strong (≥{hi:.2f})"
        ws_legend['A3'].fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        
        ws_legend['A4'] = f"Moderate ({lo:.2f}-{hi:.2f})"
        ws_legend['A4'].fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
        
        ws_legend['A5'] = f"Weak (<{lo:.2f})"
        ws_legend['A5'].fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        
        ws_legend.column_dimensions['A'].width = 25
        
        wb.save(buf)
        buf.seek(0)
        return buf.getvalue()
        
    except Exception as e:
        print(f"Excel export error: {e}")
        return None


# ============================================================================
# PDF EXPORT - Executive Summary
# ============================================================================

def to_executive_summary_pdf(
    coverage: pd.DataFrame,
    insights: Dict[str, Dict],
    jd_text: str,
    cat_map: Dict[str, str],
    hi: float = 0.75,
    lo: float = 0.35,
    jd_filename: str = "Job Description"
) -> Optional[bytes]:
    """
    Generate a professional executive summary PDF.
    
    Includes:
    - Analysis overview & key metrics
    - Top 5 candidates with scores
    - AI insights for candidates (if available)
    - Shortlist recommendation
    
    Args:
        coverage: DataFrame with candidate scores
        insights: Dict mapping candidate_name -> {top, gaps, notes}
        jd_text: Job description text
        cat_map: Criterion -> category mapping
        hi: High threshold
        lo: Low threshold
        jd_filename: Name of JD file for display
    
    Returns:
        Bytes of PDF file or None if reportlab not available
    """
    if not REPORTLAB_AVAILABLE:
        return None
    
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1F77B4'),
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1F77B4'),
            spaceAfter=10,
            spaceBefore=10
        )
        
        # Title
        story.append(Paragraph("Executive Summary - Candidate Analysis", title_style))
        story.append(Paragraph(f"Job Position: {jd_filename}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
        
        # Analysis Overview
        story.append(Paragraph("Analysis Overview", heading_style))
        overview_data = [
            ["Total Candidates Analyzed", str(len(coverage))],
            ["Evaluation Criteria", str(len([c for c in coverage.columns if c not in ('Candidate', 'Overall')]))],
            ["Highest Score Achieved", f"{coverage['Overall'].max():.2f}"],
            ["Average Score", f"{coverage['Overall'].mean():.2f}"]
        ]
        overview_table = Table(overview_data, colWidths=[10*cm, 6*cm])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E7F3FF')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 0.5*cm))
        
        # Top 5 Candidates
        story.append(Paragraph("Top 5 Candidates", heading_style))
        top5 = coverage[['Candidate', 'Overall']].head(5)
        
        top_data = [["Rank", "Candidate Name", "Overall Score", "Rating"]]
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            score = row['Overall']
            rating = "Strong Match" if score >= hi else ("Moderate Match" if score >= lo else "Weak Match")
            top_data.append([str(idx), row['Candidate'], f"{score:.2f}", rating])
        
        top_table = Table(top_data, colWidths=[2*cm, 8*cm, 3*cm, 3*cm])
        top_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F77B4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        # Color code ratings
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            score = row['Overall']
            if score >= hi:
                top_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), colors.HexColor('#C6EFCE'))]))
            elif score >= lo:
                top_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), colors.HexColor('#FFEB9C'))]))
            else:
                top_table.setStyle(TableStyle([('BACKGROUND', (3, i), (3, i), colors.HexColor('#FFC7CE'))]))
        
        story.append(top_table)
        story.append(Spacer(1, 0.5*cm))
        
        # AI Insights for Top 3 (if available)
        if insights:
            story.append(Paragraph("Key Insights for Top Candidates", heading_style))
            
            for idx, candidate_name in enumerate(top5['Candidate'].head(3), 1):
                cand_insights = insights.get(candidate_name, {})
                
                if cand_insights:
                    story.append(Paragraph(f"<b>{idx}. {candidate_name}</b>", styles['Heading3']))
                    
                    # Strengths
                    strengths = cand_insights.get('top', [])
                    if strengths:
                        story.append(Paragraph("<b>Strengths:</b>", styles['Normal']))
                        for s in strengths[:3]:  # Top 3 strengths only
                            story.append(Paragraph(f"• {s}", styles['Normal']))
                    
                    # Gaps
                    gaps = cand_insights.get('gaps', [])
                    if gaps:
                        story.append(Paragraph("<b>Development Areas:</b>", styles['Normal']))
                        for g in gaps[:3]:  # Top 3 gaps only
                            story.append(Paragraph(f"• {g}", styles['Normal']))
                    
                    # Notes
                    notes = cand_insights.get('notes', '')
                    if notes:
                        story.append(Paragraph(f"<i>{notes}</i>", styles['Normal']))
                    
                    story.append(Spacer(1, 0.3*cm))
                else:
                    story.append(Paragraph(f"<b>{idx}. {candidate_name}</b>", styles['Heading3']))
                    story.append(Paragraph("<i>No AI insights generated for this candidate</i>", styles['Normal']))
                    story.append(Spacer(1, 0.3*cm))
        
        # Recommendation (intelligent, context-aware - matching Streamlit)
        story.append(PageBreak())
        story.append(Paragraph("Recommendation", heading_style))
        
        top_candidate_name = coverage.iloc[0]['Candidate']
        top_score = coverage.iloc[0]['Overall']
        
        # Check if we have multiple strong candidates
        strong_candidates = coverage[coverage['Overall'] >= hi]
        moderate_candidates = coverage[(coverage['Overall'] >= lo) & (coverage['Overall'] < hi)]
        weak_candidates = coverage[coverage['Overall'] < lo]
        
        # Build intelligent recommendation based on actual data
        recommendation_parts = []
        
        if len(strong_candidates) == 0:
            # No strong candidates at all
            if len(moderate_candidates) > 0:
                recommendation_parts.append(
                    f"<b>Caution:</b> No candidates achieved a strong match score (≥{hi:.2f}). "
                    f"The highest score was <b>{top_score:.2f}</b> for <b>{top_candidate_name}</b>, "
                    f"indicating a <b>moderate match</b>."
                )
                recommendation_parts.append(
                    f"We recommend carefully reviewing the {len(moderate_candidates)} moderate-scoring candidate(s) "
                    f"to identify specific skill gaps, or consider expanding the candidate pool."
                )
            else:
                recommendation_parts.append(
                    f"<b>Warning:</b> All candidates scored below the moderate threshold ({lo:.2f}). "
                    f"The highest score was only <b>{top_score:.2f}</b> for <b>{top_candidate_name}</b>."
                )
                recommendation_parts.append(
                    "We recommend reconsidering the job requirements or sourcing additional candidates, "
                    "as the current pool shows weak alignment with the position criteria."
                )
        
        elif len(strong_candidates) == 1:
            # Clear winner
            recommendation_parts.append(
                f"<b>{top_candidate_name}</b> is the clear leading candidate with a strong overall score "
                f"of <b>{top_score:.2f}</b>, demonstrating excellent alignment with the position requirements."
            )
            
            if len(moderate_candidates) > 0:
                recommendation_parts.append(
                    f"Additionally, {len(moderate_candidates)} candidate(s) achieved moderate scores and could serve as backup options."
                )
            
            recommendation_parts.append(
                f"We recommend prioritizing <b>{top_candidate_name}</b> for the next stage of recruitment."
            )
        
        else:
            # Multiple strong candidates
            top_3_strong = strong_candidates.head(3)
            score_range = top_3_strong['Overall'].max() - top_3_strong['Overall'].min()
            
            if score_range < 0.10:  # Very close scores
                names = ", ".join([f"<b>{row['Candidate']}</b>" for _, row in top_3_strong.iterrows()])
                recommendation_parts.append(
                    f"We have <b>{len(strong_candidates)} strong candidates</b> with very similar scores "
                    f"(range: {top_3_strong['Overall'].min():.2f}–{top_3_strong['Overall'].max():.2f}). "
                    f"The top candidates are: {names}."
                )
                recommendation_parts.append(
                    "Given the close scoring, we recommend interviewing multiple candidates to assess "
                    "cultural fit, communication skills, and other qualitative factors."
                )
            else:
                recommendation_parts.append(
                    f"<b>{top_candidate_name}</b> is the leading candidate with a score of <b>{top_score:.2f}</b>, "
                    f"followed by {len(strong_candidates)-1} other strong candidate(s)."
                )
                recommendation_parts.append(
                    f"We recommend prioritizing <b>{top_candidate_name}</b>, while keeping other strong candidates "
                    f"as viable alternatives."
                )
        
        # Combine recommendation parts
        for part in recommendation_parts:
            story.append(Paragraph(part, styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
        
        story.append(Spacer(1, 0.3*cm))
        
        # Legend
        story.append(Paragraph("Score Interpretation", heading_style))
        legend_text = (
            f"• <b>Strong (≥{hi:.2f}):</b> Excellent alignment with requirements<br/>"
            f"• <b>Moderate ({lo:.2f}–{hi:.2f}):</b> Acceptable fit with some gaps<br/>"
            f"• <b>Weak (&lt;{lo:.2f}):</b> Significant gaps in required areas"
        )
        story.append(Paragraph(legend_text, styles['Normal']))
        story.append(Spacer(1, 0.2*cm))
        
        # Build PDF
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None


# Function continues in next part...
