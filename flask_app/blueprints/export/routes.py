from flask import render_template, redirect, url_for, flash, request, jsonify, send_file, session
from flask_login import login_required, current_user
from database import db, Analysis, CandidateFile, UserSettings, Draft
import json
import os
import re
import io
import base64
import pandas as pd
from io import StringIO

from blueprints.export import export_bp

# ============================================================================
# EXPORT ROUTES - Report generation and downloads
# ============================================================================

@export_bp.route('/exports/<int:analysis_id>')
@login_required
def exports(analysis_id):
    """Main export page with all export options"""
    # Check if admin is viewing another user's analysis
    is_admin = session.get('admin_logged_in', False)
    
    if is_admin:
        # Admin can view any analysis
        analysis = Analysis.query.filter_by(id=analysis_id).first()
    else:
        # Regular user can only view their own
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Check if this is the current draft's analysis
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    is_current_draft_analysis = False
    if draft and draft.created_at < analysis.created_at and (not draft.updated_at or draft.updated_at <= analysis.created_at):
        is_current_draft_analysis = True
    
    # Get user settings for default evidence preference
    user_settings = UserSettings.get_or_create(current_user.id)
    
    # Parse coverage data
    import pandas as pd
    from io import StringIO
    coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
    
    # Parse gpt_candidates list (candidates with AI insights)
    gpt_candidates_list = []
    if analysis.gpt_candidates:
        try:
            gpt_candidates_list = json.loads(analysis.gpt_candidates)
        except:
            gpt_candidates_list = []
    
    # Check if admin is viewing another user's analysis
    viewing_other_user = is_admin and analysis.user_id != current_user.id
    analysis_owner = User.query.get(analysis.user_id) if viewing_other_user else None
    analysis_owner_email = analysis_owner.email if analysis_owner else None
    
    return render_template('export.html', 
                         analysis=analysis, 
                         coverage=coverage_df,
                         user_settings=user_settings,
                         is_current_draft_analysis=is_current_draft_analysis,
                         gpt_candidates=gpt_candidates_list,
                         viewing_other_user=viewing_other_user,
                         analysis_owner_email=analysis_owner_email)

@export_bp.route('/export/<int:analysis_id>/preview-pdf')
@login_required
def preview_executive_pdf(analysis_id):
    """Preview executive summary PDF as images"""
    from export_utils import to_executive_summary_pdf, render_pdf_to_images
    import base64
    
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        return {'error': 'Analysis not found'}, 404
    
    # Parse data
    import pandas as pd
    from io import StringIO
    coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
    insights = json.loads(analysis.insights_data)
    criteria = json.loads(analysis.criteria_list)
    
    # Build category map
    category_map = {}
    for crit in criteria:
        # Handle multiple formats: string, dict with 'criterion', or dict with 'text'
        if isinstance(crit, dict):
            crit_text = crit.get('criterion') or crit.get('text') or str(crit)
        else:
            crit_text = crit
        
        crit_lower = crit_text.lower()
        if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
            category_map[crit_text] = 'Technical Skills'
        elif any(word in crit_lower for word in ['experience', 'years', 'background']):
            category_map[crit_text] = 'Experience'
        elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
            category_map[crit_text] = 'Qualifications'
        elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
            category_map[crit_text] = 'Soft Skills'
        else:
            category_map[crit_text] = 'Other Requirements'
    
    # Generate PDF
    pdf_bytes = to_executive_summary_pdf(
        coverage=coverage_df,
        insights=insights,
        jd_text=analysis.job_description_text,
        cat_map=category_map,
        hi=0.75,
        lo=0.35,
        jd_filename=analysis.job_title,
        job_number=analysis.id
    )
    
    if pdf_bytes is None:
        return render_template('pdf_preview_error.html', 
                             error='PDF generation not available (ReportLab not installed)')
    
    # Render to images
    images = render_pdf_to_images(pdf_bytes, max_pages=10)
    
    if not images:
        return render_template('pdf_preview_error.html',
                             error='PDF preview not available (PyMuPDF not installed). Click Download PDF to view.')
    
    # Convert images to base64 for embedding in HTML
    image_data = []
    for idx, img_bytes in enumerate(images, 1):
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        image_data.append({
            'page': idx,
            'total': len(images),
            'data': b64_img
        })
    
    return render_template('pdf_preview.html', 
                         images=image_data,
                         analysis=analysis)

@export_bp.route('/export/<int:analysis_id>/preview-pdf-inline')
@login_required
def preview_executive_pdf_inline(analysis_id):
    """Serve executive summary PDF inline for preview"""
    from flask import send_file
    from export_utils import to_executive_summary_pdf
    import io
    
    try:
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            return "Analysis not found", 404
        
        # Parse data - same as export_executive_pdf
        import pandas as pd
        from io import StringIO
        coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
        insights = json.loads(analysis.insights_data)
        criteria = json.loads(analysis.criteria_list)
        
        # Build category map
        category_map = {}
        for crit in criteria:
            # Handle multiple formats: string, dict with 'criterion', or dict with 'text'
            if isinstance(crit, dict):
                crit_text = crit.get('criterion') or crit.get('text') or str(crit)
            else:
                crit_text = crit
            
            crit_lower = crit_text.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit_text] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit_text] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit_text] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit_text] = 'Soft Skills'
            else:
                category_map[crit_text] = 'Other Requirements'
        
        # Generate PDF
        pdf_bytes = to_executive_summary_pdf(
            coverage=coverage_df,
            insights=insights,
            jd_text=analysis.job_description_text,
            cat_map=category_map,
            hi=0.75,
            lo=0.35,
            jd_filename=analysis.job_title,
            job_number=analysis.id
        )
        
        if pdf_bytes is None:
            return "PDF generation not available (ReportLab not installed)", 500
        
        # Return PDF with inline disposition (not as download)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=False,
            download_name=f'{analysis.job_title}_preview.pdf'
        )
    except Exception as e:
        import traceback
        app.logger.error(f"Error generating PDF preview: {str(e)}")
        app.logger.error(traceback.format_exc())
        return f"Error generating PDF: {str(e)}", 500

@export_bp.route('/export/<int:analysis_id>/executive-pdf')
@login_required
def export_executive_pdf(analysis_id):
    """Download executive summary PDF"""
    from flask import send_file
    from export_utils import to_executive_summary_pdf
    
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Parse data
    import pandas as pd
    from io import StringIO
    coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
    insights = json.loads(analysis.insights_data)
    criteria = json.loads(analysis.criteria_list)
    
    # Build category map
    category_map = {}
    for crit in criteria:
        # Handle multiple formats: string, dict with 'criterion', or dict with 'text'
        if isinstance(crit, dict):
            crit_text = crit.get('criterion') or crit.get('text') or str(crit)
        else:
            crit_text = crit
        
        crit_lower = crit_text.lower()
        if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
            category_map[crit_text] = 'Technical Skills'
        elif any(word in crit_lower for word in ['experience', 'years', 'background']):
            category_map[crit_text] = 'Experience'
        elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
            category_map[crit_text] = 'Qualifications'
        elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
            category_map[crit_text] = 'Soft Skills'
        else:
            category_map[crit_text] = 'Other Requirements'
    
    # Generate PDF
    pdf_bytes = to_executive_summary_pdf(
        coverage=coverage_df,
        insights=insights,
        jd_text=analysis.job_description_text,
        cat_map=category_map,
        hi=0.75,
        lo=0.35,
        jd_filename=analysis.job_title,
        job_number=analysis.id
    )
    
    if pdf_bytes is None:
        flash('PDF generation not available (ReportLab not installed)', 'error')
        return redirect(url_for('export.exports', analysis_id=analysis_id))
    
    # Return as download with sanitized filename and timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d')
    safe_title = re.sub(r'[^\w\s-]', '', analysis.job_title).strip().replace(' ', '_')[:50]
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'executive_summary_{safe_title}_{timestamp}.pdf'
    )

@export_bp.route('/export/<int:analysis_id>/executive-docx')
@login_required
def export_executive_docx(analysis_id):
    """Download executive summary Word document"""
    from flask import send_file
    from export_utils import to_executive_summary_word
    
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Parse data
    import pandas as pd
    from io import StringIO
    coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
    insights = json.loads(analysis.insights_data)
    criteria = json.loads(analysis.criteria_list)
    
    # Build category map
    category_map = {}
    for crit in criteria:
        # Handle multiple formats: string, dict with 'criterion', or dict with 'text'
        if isinstance(crit, dict):
            crit_text = crit.get('criterion') or crit.get('text') or str(crit)
        else:
            crit_text = crit
        
        crit_lower = crit_text.lower()
        if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
            category_map[crit_text] = 'Technical Skills'
        elif any(word in crit_lower for word in ['experience', 'years', 'background']):
            category_map[crit_text] = 'Experience'
        elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
            category_map[crit_text] = 'Qualifications'
        elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
            category_map[crit_text] = 'Soft Skills'
        else:
            category_map[crit_text] = 'Other Requirements'
    
    # Generate Word document
    docx_bytes = to_executive_summary_word(
        coverage=coverage_df,
        insights=insights,
        jd_text=analysis.job_description_text,
        cat_map=category_map,
        hi=0.75,
        lo=0.35,
        jd_filename=analysis.job_title,
        job_number=analysis.id
    )
    
    if docx_bytes is None:
        flash('Word generation not available (python-docx not installed)', 'error')
        return redirect(url_for('export.exports', analysis_id=analysis_id))
    
    # Return as download with sanitized filename and timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d')
    safe_title = re.sub(r'[^\w\s-]', '', analysis.job_title).strip().replace(' ', '_')[:50]
    return send_file(
        io.BytesIO(docx_bytes),
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        as_attachment=True,
        download_name=f'executive_summary_{safe_title}_{timestamp}.docx'
    )

@export_bp.route('/export/<int:analysis_id>/coverage-excel')
@login_required
def export_coverage_excel(analysis_id):
    """Download coverage matrix Excel file"""
    from flask import send_file
    from export_utils import to_excel_coverage_matrix
    
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Parse data
    import pandas as pd
    from io import StringIO
    coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
    criteria = json.loads(analysis.criteria_list)
    
    # Build category map
    category_map = {}
    for crit in criteria:
        # Handle multiple formats: string, dict with 'criterion', or dict with 'text'
        if isinstance(crit, dict):
            crit_text = crit.get('criterion') or crit.get('text') or str(crit)
        else:
            crit_text = crit
        
        crit_lower = crit_text.lower()
        if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
            category_map[crit_text] = 'Technical Skills'
        elif any(word in crit_lower for word in ['experience', 'years', 'background']):
            category_map[crit_text] = 'Experience'
        elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
            category_map[crit_text] = 'Qualifications'
        elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
            category_map[crit_text] = 'Soft Skills'
        else:
            category_map[crit_text] = 'Other Requirements'
    
    # Generate Excel
    excel_bytes = to_excel_coverage_matrix(
        coverage=coverage_df,
        cat_map=category_map,
        hi=0.75,
        lo=0.35,
        job_title=analysis.job_title,
        job_number=analysis.id
    )
    
    if excel_bytes is None:
        flash('Excel generation not available (openpyxl not installed)', 'error')
        return redirect(url_for('export.exports', analysis_id=analysis_id))
    
    # Return as download with sanitized filename
    safe_title = re.sub(r'[^\w\s-]', '', analysis.job_title).strip().replace(' ', '_')[:50]
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'coverage_matrix_{safe_title}.xlsx'
    )

@export_bp.route('/export/<int:analysis_id>/coverage-csv')
@login_required
def export_coverage_csv(analysis_id):
    """Download coverage matrix CSV (fallback)"""
    from flask import send_file
    
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Parse data
    import pandas as pd
    from io import StringIO
    coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
    
    # Transpose: Candidates as columns, Criteria as rows
    # Set 'Candidate' as index, transpose, then reset index
    coverage_df = coverage_df.set_index('Candidate').T
    
    # Round all numeric values to 2 decimal places for better readability
    coverage_df = coverage_df.round(2)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    coverage_df.to_csv(csv_buffer, index=True)  # Keep index (criteria names)
    csv_buffer.seek(0)
    
    # Return as download with sanitized filename
    safe_title = re.sub(r'[^\w\s-]', '', analysis.job_title).strip().replace(' ', '_')[:50]
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'coverage_matrix_{safe_title}.csv'
    )

@export_bp.route('/export/<int:analysis_id>/individual-pdf', methods=['POST'])
@login_required
def export_individual_pdf(analysis_id):
    """Generate and merge individual candidate PDFs"""
    try:
        from flask import send_file
        from export_candidate import to_individual_candidate_pdf
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Get request data
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        candidates = data.get('candidates', [])
        include_justifications = data.get('include_justifications', False)
        
        if not candidates:
            return jsonify({'error': 'No candidates selected'}), 400
        
        # Parse data
        import pandas as pd
        from io import StringIO
        coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
        insights = json.loads(analysis.insights_data)
        evidence_map_raw = json.loads(analysis.evidence_data)
        criteria = json.loads(analysis.criteria_list)
        gpt_candidates_raw = json.loads(analysis.gpt_candidates) if analysis.gpt_candidates else []
        
        # Convert evidence_map keys to tuples (handle both pipe and tuple formats)
        evidence_map = {}
        for key_str, value in evidence_map_raw.items():
            if '|||' in key_str:
                parts = key_str.split('|||', 1)
                if len(parts) == 2:
                    evidence_map[(parts[0], parts[1])] = value
            else:
                try:
                    key_tuple = eval(key_str)
                    if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                        evidence_map[key_tuple] = value
                except:
                    pass
        
        # Build category map
        category_map = {}
        for crit in criteria:
            # Handle multiple formats: string, dict with 'criterion', or dict with 'text'
            if isinstance(crit, dict):
                crit_text = crit.get('criterion') or crit.get('text') or str(crit)
            else:
                crit_text = crit
            
            crit_lower = crit_text.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit_text] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit_text] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit_text] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit_text] = 'Soft Skills'
            else:
                category_map[crit_text] = 'Other Requirements'
        
        # Generate PDFs for each candidate
        pdf_list = []
        for candidate in candidates:
            coverage_row = coverage_df[coverage_df['Candidate'] == candidate].iloc[0]
            cand_insights = insights.get(candidate, {})
            
            print(f"Generating PDF for candidate: {candidate}")
            pdf_bytes = to_individual_candidate_pdf(
                candidate_name=candidate,
                coverage_row=coverage_row,
                insights=cand_insights,
                evidence_map=evidence_map,
                cat_map=category_map,
                hi=0.75,
                lo=0.35,
                include_justifications=include_justifications,
                job_title=analysis.job_title,
                gpt_candidates=gpt_candidates_raw,
                job_number=analysis.id
            )
            
            if pdf_bytes:
                print(f"✓ PDF generated for {candidate}, size: {len(pdf_bytes)} bytes")
                pdf_list.append(pdf_bytes)
            else:
                print(f"✗ PDF generation failed for {candidate}")
        
        print(f"Total PDFs generated: {len(pdf_list)}")
        
        if not pdf_list:
            return jsonify({'error': 'PDF generation failed'}), 500
        
        # If single candidate, return directly
        if len(pdf_list) == 1:
            safe_name = re.sub(r'[^\w\s-]', '', candidates[0]).strip().replace(' ', '_')
            return send_file(
                io.BytesIO(pdf_list[0]),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'{safe_name}_report.pdf'
            )
        
        # Merge multiple PDFs
        try:
            from PyPDF2 import PdfMerger
            merger = PdfMerger()
            for pdf_bytes in pdf_list:
                merger.append(io.BytesIO(pdf_bytes))
            
            merged_buffer = io.BytesIO()
            merger.write(merged_buffer)
            merger.close()
            merged_buffer.seek(0)
            
            return send_file(
                merged_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'candidate_reports_{len(candidates)}_candidates.pdf'
            )
        except ImportError:
            # Fallback: return first PDF only if PyPDF2 not available
            flash('PyPDF2 not installed - returning first candidate only', 'warning')
            safe_name = re.sub(r'[^\w\s-]', '', candidates[0]).strip().replace(' ', '_')
            return send_file(
                io.BytesIO(pdf_list[0]),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'{safe_name}_report.pdf'
            )
        except Exception as e:
            print(f"PDF merge error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500
    except Exception as e:
        print(f"FATAL: PDF export error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@export_bp.route('/export/<int:analysis_id>/individual-docx', methods=['POST'])
@login_required
def export_individual_docx(analysis_id):
    """Generate individual candidate Word documents (ZIP if multiple)"""
    try:
        from flask import send_file
        from export_candidate import to_individual_candidate_docx
        import zipfile
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Get request data
        data = request.get_json()
        candidates = data.get('candidates', [])
        include_justifications = data.get('include_justifications', False)
        
        if not candidates:
            return jsonify({'error': 'No candidates selected'}), 400
        
        # Parse data
        import pandas as pd
        from io import StringIO
        coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
        insights = json.loads(analysis.insights_data)
        evidence_map_raw = json.loads(analysis.evidence_data)
        criteria = json.loads(analysis.criteria_list)
        gpt_candidates_raw = json.loads(analysis.gpt_candidates) if analysis.gpt_candidates else []
        
        # Convert evidence_map keys to tuples (handle both pipe and tuple formats)
        evidence_map = {}
        for key_str, value in evidence_map_raw.items():
            if '|||' in key_str:
                parts = key_str.split('|||', 1)
                if len(parts) == 2:
                    evidence_map[(parts[0], parts[1])] = value
            else:
                try:
                    key_tuple = eval(key_str)
                    if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                        evidence_map[key_tuple] = value
                except:
                    pass
        
        # Build category map
        category_map = {}
        for crit in criteria:
            # Handle multiple formats: string, dict with 'criterion', or dict with 'text'
            if isinstance(crit, dict):
                crit_text = crit.get('criterion') or crit.get('text') or str(crit)
            else:
                crit_text = crit
            
            crit_lower = crit_text.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit_text] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit_text] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit_text] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit_text] = 'Soft Skills'
            else:
                category_map[crit_text] = 'Other Requirements'
        
        # Generate Word docs for each candidate
        docx_list = []
        for candidate in candidates:
            coverage_row = coverage_df[coverage_df['Candidate'] == candidate].iloc[0]
            cand_insights = insights.get(candidate, {})
            
            docx_bytes = to_individual_candidate_docx(
                candidate_name=candidate,
                coverage_row=coverage_row,
                insights=cand_insights,
                evidence_map=evidence_map,
                cat_map=category_map,
                hi=0.75,
                lo=0.35,
                include_justifications=include_justifications,
                job_title=analysis.job_title,
                gpt_candidates=gpt_candidates_raw,
                job_number=analysis.id
            )
            
            if docx_bytes:
                docx_list.append((candidate, docx_bytes))
        
        if not docx_list:
            return jsonify({'error': 'Word generation failed'}), 500
        
        # If single candidate, return directly
        if len(docx_list) == 1:
            safe_name = re.sub(r'[^\w\s-]', '', docx_list[0][0]).strip().replace(' ', '_')
            return send_file(
                io.BytesIO(docx_list[0][1]),
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                as_attachment=True,
                download_name=f'{safe_name}_report.docx'
            )
        
        # Create ZIP for multiple candidates
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for candidate, docx_bytes in docx_list:
                safe_name = re.sub(r'[^\w\s-]', '', candidate).strip().replace(' ', '_')
                zip_file.writestr(f'{safe_name}_report.docx', docx_bytes)
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'candidate_reports_{len(candidates)}_candidates.zip'
        )
    except Exception as e:
        print(f"FATAL: DOCX export error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@export_bp.route('/export/<int:analysis_id>/candidates-csv')
@login_required
def export_candidates_csv(analysis_id):
    """Download simple candidate list CSV"""
    from flask import send_file
    
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Parse data
    import pandas as pd
    from io import StringIO
    coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
    
    # Create simple CSV with just Candidate and Overall
    simple_df = coverage_df[['Candidate', 'Overall']].copy()
    simple_df.columns = ['Candidate Name', 'Overall Score']
    
    # Round Overall Score to 2 decimal places
    simple_df['Overall Score'] = simple_df['Overall Score'].round(2)
    
    # Convert to CSV with Job# header
    csv_buffer = io.StringIO()
    # Write Job# and Job Title as header
    csv_buffer.write(f"Job #{analysis.id:04d}: {analysis.job_title}\n")
    simple_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Return as download with sanitized filename
    safe_title = re.sub(r'[^\w\s-]', '', analysis.job_title).strip().replace(' ', '_')[:50]
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'candidates_{safe_title}.csv'
    )

@export_bp.route('/export/<int:analysis_id>/criteria-csv')
@login_required
def export_criteria_csv(analysis_id):
    """Download criteria list CSV"""
    from flask import send_file
    
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Parse criteria
    criteria = json.loads(analysis.criteria_list)
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame({'Criterion': criteria})
    
    # Convert to CSV with Job# header
    csv_buffer = io.StringIO()
    # Write Job# and Job Title as header
    csv_buffer.write(f"Job #{analysis.id:04d}: {analysis.job_title}\n")
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Return as download with sanitized filename
    safe_title = re.sub(r'[^\w\s-]', '', analysis.job_title).strip().replace(' ', '_')[:50]
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'criteria_{safe_title}.csv'
    )

# ============================================================================
