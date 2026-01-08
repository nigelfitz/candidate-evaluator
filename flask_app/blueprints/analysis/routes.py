from flask import render_template, redirect, url_for, flash, request, jsonify, send_file, session, current_app
from flask_login import login_required, current_user
from database import db, User, Draft, DraftResume, Analysis, CandidateFile, Transaction, Feedback, UserSettings
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
import re
import hashlib
import io
import base64

from blueprints.analysis import analysis_bp

# Helper functions
def load_system_settings():
    """Load system settings from JSON file"""
    settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'system_settings.json')
    with open(settings_path, 'r') as f:
        return json.load(f)

def load_pricing_settings():
    """Load pricing settings from JSON file"""
    settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'pricing_settings.json')
    with open(settings_path, 'r') as f:
        return json.load(f)

@analysis_bp.route('/jd-length-warning')
@login_required
def jd_length_warning_route():
    """Show JD length warning page"""
    jd_length = request.args.get('jd_length', type=int)
    jd_limit = request.args.get('jd_limit', type=int)
    
    if not session.get('show_jd_length_warning'):
        flash('Invalid access to warning page', 'error')
        return redirect(url_for('analysis.analyze'))
    
    return render_template('jd_length_warning.html',
                         jd_length=jd_length,
                         jd_limit=jd_limit)

@analysis_bp.route('/resume-length-warning')
@login_required
def resume_length_warning_route():
    """Show resume length warning page"""
    truncated_count = request.args.get('truncated_count', type=int)
    resume_limit = request.args.get('resume_limit', type=int)
    
    if not session.get('show_resume_length_warning'):
        flash('Invalid access to warning page', 'error')
        return redirect(url_for('analysis.analyze', step='resumes'))
    
    # Get actual truncated resume details from database
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    truncated_resumes = []
    if draft:
        for resume in DraftResume.query.filter_by(draft_id=draft.id).all():
            if len(resume.extracted_text or '') > resume_limit:
                truncated_resumes.append({
                    'name': resume.candidate_name,
                    'length': len(resume.extracted_text)
                })
    
    return render_template('resume_length_warning.html',
                         truncated_resumes=truncated_resumes,
                         resume_limit=resume_limit)

@analysis_bp.route('/document-warnings')
@login_required
def document_warnings_route():
    """Show combined document warnings page (too short AND/OR too long)"""
    if not session.get('show_document_warnings'):
        flash('Invalid access to warning page', 'error')
        return redirect(url_for('analysis.analyze'))
    
    # Get warning details from session
    jd_length = session.get('jd_length')
    jd_limit = session.get('jd_limit')
    resume_limit = session.get('resume_limit')
    
    # Only show JD warnings if they haven't been confirmed yet
    jd_warnings_already_confirmed = session.get('warnings_confirmed', False)
    short_jd = session.get('short_jd', False) and not jd_warnings_already_confirmed
    long_jd = session.get('long_jd', False) and not jd_warnings_already_confirmed
    
    # Get resume details from database
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    short_resumes = []
    long_resumes = []
    
    if draft:
        for resume in DraftResume.query.filter_by(draft_id=draft.id).all():
            resume_length = len(resume.extracted_text or '')
            resume_text = resume.extracted_text or ''
            
            # Check for too short with multiple criteria
            is_too_short = False
            if resume_length < 50:
                is_too_short = True
            elif resume_length < 500:
                non_whitespace = len(resume_text.strip().replace('\n', '').replace(' ', ''))
                whitespace_ratio = 1.0 - (non_whitespace / max(resume_length, 1))
                if whitespace_ratio > 0.8:
                    is_too_short = True
            
            if is_too_short:
                short_resumes.append({
                    'name': resume.candidate_name,
                    'length': resume_length
                })
            elif resume_length > resume_limit:  # Too long threshold
                long_resumes.append({
                    'name': resume.candidate_name,
                    'length': resume_length
                })
    
    has_short_documents = short_jd or short_resumes
    has_long_documents = long_jd or long_resumes
    
    return render_template('document_warnings.html',
                         has_short_documents=has_short_documents,
                         has_long_documents=has_long_documents,
                         short_jd=short_jd,
                         long_jd=long_jd,
                         short_resumes=short_resumes,
                         long_resumes=long_resumes,
                         jd_length=jd_length,
                         jd_limit=jd_limit,
                         resume_limit=resume_limit)

@analysis_bp.route('/analyze', methods=['GET', 'POST'])

@login_required
def analyze():
    """Analysis page - Step 1: Upload JD, extract criteria OR Step 2: Upload resumes and run analysis"""
    if request.method == 'GET':
        # Get step parameter (jd, resumes, or default to auto-detect)
        step = request.args.get('step', 'auto')
        
        # If new=1, clear any existing draft to start fresh
        if request.args.get('new') == '1':
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if draft:
                # Explicitly delete all associated draft resumes first
                DraftResume.query.filter_by(draft_id=draft.id).delete()
                db.session.flush()  # Ensure resumes are deleted first
                # Then delete the draft
                db.session.delete(draft)
                db.session.commit()
                flash('Started new analysis. Previous draft cleared.', 'info')
            # Clear any warning-related session flags
            session.pop('truncation_confirmed', None)
            session.pop('show_jd_length_warning', None)
            session.pop('show_resume_length_warning', None)
            system_settings = load_system_settings()
            return render_template('analyze.html', user=current_user, jd_data=None, criteria_count=0, current_step='jd',
                                 in_workflow=False, has_unsaved_work=False, analysis_completed=False, system_settings=system_settings)
        
        # Get draft data if exists
        draft = Draft.query.filter_by(user_id=current_user.id).first()
        if draft:
            criteria_data = json.loads(draft.criteria_data) if draft.criteria_data else []
            jd_data = {'filename': draft.jd_filename, 'text': draft.jd_text[:500] if draft.jd_text else None}
            
            # Get uploaded resumes for this draft
            uploaded_resumes = DraftResume.query.filter_by(draft_id=draft.id).all()
            
            # Get most recent completed analysis for this user to enable Results box link
            latest_analysis = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).first()
            
            # Check if CURRENT draft has been analyzed and if it's been modified since
            analysis_completed = False
            draft_modified_after_analysis = False
            current_draft_analysis_id = None
            
            if latest_analysis and draft.created_at < latest_analysis.created_at:
                analysis_completed = True
                current_draft_analysis_id = latest_analysis.id
                # Check if draft was updated after the analysis
                if draft.updated_at > latest_analysis.created_at:
                    draft_modified_after_analysis = True
            
            # Auto-detect step if not specified
            if step == 'auto':
                step = 'resumes' if criteria_data else 'jd'
            
            has_unsaved_work = draft is not None
            system_settings = load_system_settings()
                
            return render_template('analyze.html', user=current_user, jd_data=jd_data, 
                                 criteria_count=len(criteria_data), current_step=step,
                                 latest_analysis_id=current_draft_analysis_id,
                                 uploaded_resumes=uploaded_resumes,
                                 in_workflow=True, has_unsaved_work=has_unsaved_work,
                                 analysis_completed=analysis_completed,
                                 draft_modified_after_analysis=draft_modified_after_analysis,
                                 system_settings=system_settings)
        else:
            system_settings = load_system_settings()
            return render_template('analyze.html', user=current_user, jd_data=None, 
                                 criteria_count=0, current_step='jd', latest_analysis_id=None,
                                 uploaded_resumes=[], in_workflow=False, has_unsaved_work=False,
                                 analysis_completed=False, system_settings=system_settings)
    
    # POST - handle JD upload, resume upload, or full analysis
    action = request.form.get('action')
    
    if action == 'upload_jd':
        # Step 1: Upload and extract criteria from JD
        try:
            from analysis import read_file_bytes, hash_bytes, extract_jd_sections_with_gpt, build_criteria_from_sections
            
            jd_file = request.files.get('job_description')
            jd_text = request.form.get('jd_text', '').strip()
            
            if not jd_file and not jd_text:
                flash('Please upload a JD file or paste JD text', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Read JD
            if jd_file:
                jd_bytes = jd_file.read()
                jd_text_content = read_file_bytes(jd_bytes, jd_file.filename)
                jd_filename = jd_file.filename
            else:
                jd_text_content = jd_text
                jd_filename = "pasted_text.txt"
                jd_bytes = jd_text.encode('utf-8')
            
            jd_hash = hash_bytes(jd_bytes)
            
            # Check JD length and show warning if needed
            system_settings = load_system_settings()
            warnings_enabled = system_settings.get('enable_document_length_warnings', True)
            
            if warnings_enabled:
                try:
                    from analysis import load_gpt_settings
                    gpt_settings = load_gpt_settings()
                    jd_limit = gpt_settings.get('jd_text_chars', 5000)
                    jd_length = len(jd_text_content)
                    
                    # Check for too short (likely scanned image/corrupted)
                    # Multiple checks for robustness:
                    # 1. Extremely short (< 50 chars)
                    # 2. High whitespace ratio (> 80% whitespace suggests garbage extraction)
                    jd_too_short = False
                    if jd_length < 50:
                        jd_too_short = True
                    elif jd_length < 300:  # Only check ratio if suspiciously short
                        non_whitespace = len(jd_text_content.strip().replace('\n', '').replace(' ', ''))
                        whitespace_ratio = 1.0 - (non_whitespace / max(jd_length, 1))
                        if whitespace_ratio > 0.8:
                            jd_too_short = True
                    
                    # Check for too long (will be truncated)
                    jd_too_long = jd_length > jd_limit
                    
                    if jd_too_short or jd_too_long:
                        # Store JD data in draft (not session - too large for cookies)
                        draft = Draft.query.filter_by(user_id=current_user.id).first()
                        if not draft:
                            draft = Draft(user_id=current_user.id)
                            db.session.add(draft)
                        
                        draft.jd_filename = jd_filename
                        draft.jd_text = jd_text_content
                        draft.jd_hash = jd_hash
                        draft.jd_bytes = jd_bytes
                        db.session.commit()
                        
                        # Store warning details in session
                        session['jd_length'] = jd_length
                        session['jd_limit'] = jd_limit
                        session['short_jd'] = jd_too_short
                        session['long_jd'] = jd_too_long
                        
                        # Route to appropriate warning page
                        if jd_too_short and not jd_too_long:
                            # Only too short - use combined page (may have short resumes later)
                            session['show_document_warnings'] = True
                            session.modified = True
                            return redirect(url_for('analysis.document_warnings_route'))
                        elif jd_too_long and not jd_too_short:
                            # Only too long - use existing single-issue page
                            session['show_jd_length_warning'] = True
                            session.modified = True
                            return redirect(url_for('analysis.jd_length_warning_route', jd_length=jd_length, jd_limit=jd_limit))
                        else:
                            # Both issues - use combined page
                            session['show_document_warnings'] = True
                            session.modified = True
                            return redirect(url_for('analysis.document_warnings_route'))
                except Exception as e:
                    print(f"ERROR in JD length warning check: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue without warning if there's an error
            
            # Extract criteria (no limit - let user uncheck unwanted ones)
            sections = extract_jd_sections_with_gpt(jd_text_content)
            criteria, cat_map = build_criteria_from_sections(sections, per_section=999, cap_total=10000)
            
            if not criteria:
                flash('Could not extract criteria from job description', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Use AI-extracted job title from sections
            job_title = sections.job_title or "Position Not Specified"
            
            # Store in database
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if not draft:
                draft = Draft(user_id=current_user.id)
                db.session.add(draft)
            
            draft.jd_filename = jd_filename
            draft.jd_text = jd_text_content
            draft.jd_hash = jd_hash
            draft.jd_bytes = jd_bytes
            draft.job_title = job_title
            draft.criteria_data = json.dumps([
                {'criterion': crit, 'category': cat_map.get(crit, 'Other Requirements'), 'use': True}
                for crit in criteria
            ])
            
            db.session.commit()
            
            flash(f'✅ JD processed! Extracted {len(criteria)} criteria. Review them on the Job Criteria page.', 'success')
            return redirect(url_for('analysis.review_criteria'))
            
        except Exception as e:
            flash(f'JD processing failed: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('analysis.analyze'))
    
    elif action == 'confirm_jd_length':
        # User confirmed to proceed with long JD
        try:
            from analysis import read_file_bytes, hash_bytes, extract_jd_sections_with_gpt, build_criteria_from_sections
            
            # Retrieve JD data from draft (stored during warning display)
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if not draft or not draft.jd_text:
                flash('Session expired. Please upload your JD again.', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Clear session flag
            session.pop('show_jd_length_warning', None)
            
            jd_text_content = draft.jd_text
            jd_hash = draft.jd_hash
            
            # Extract criteria
            sections = extract_jd_sections_with_gpt(jd_text_content)
            criteria, cat_map = build_criteria_from_sections(sections, per_section=999, cap_total=10000)
            
            if not criteria:
                flash('Could not extract criteria from job description', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Use AI-extracted job title from sections
            job_title = sections.job_title or "Position Not Specified"
            
            # Draft already exists with JD data, just update criteria
            draft.job_title = job_title
            draft.criteria_data = json.dumps([
                {'criterion': crit, 'category': cat_map.get(crit, 'Other Requirements'), 'use': True}
                for crit in criteria
            ])
            
            db.session.commit()
            
            flash(f'✅ JD processed! Extracted {len(criteria)} criteria. Review them on the Job Criteria page.', 'success')
            return redirect(url_for('analysis.review_criteria'))
            
        except Exception as e:
            flash(f'JD processing failed: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('analysis.analyze'))
    
    elif action == 'upload_resumes':
        # Step 3: Upload resumes and store in draft (don't run analysis yet)
        try:
            from analysis import read_file_bytes, hash_bytes, infer_candidate_name
            
            # Check we have a draft with criteria
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if not draft or not draft.criteria_data:
                flash('Please upload a JD and review criteria first', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Get resume files
            resume_files = request.files.getlist('resumes')
            if not resume_files or not resume_files[0].filename:
                flash('Please select at least one resume file', 'error')
                return redirect(url_for('analysis.analyze', step='resumes'))
            
            # Process and store resumes temporarily
            resumes_added = 0
            processed_resumes = []  # Store for length checking
            locked_files = []  # Track files that couldn't be read
            
            for resume_file in resume_files:
                if not resume_file.filename:
                    continue
                
                # Wrap entire file processing in error handling
                try:
                    resume_bytes = resume_file.read()
                    resume_hash = hash_bytes(resume_bytes)
                    
                    # Check if already uploaded (by hash)
                    existing = DraftResume.query.filter_by(draft_id=draft.id, file_hash=resume_hash).first()
                    if existing:
                        continue  # Skip duplicates
                    
                    resume_text = read_file_bytes(resume_bytes, resume_file.filename)
                    
                    # Check if file is unreadable (very short text suggests corrupted/scanned image)
                    # Use obvious placeholder name instead of trying to extract
                    import os
                    if not resume_text or len(resume_text.strip()) < 100:
                        candidate_name = f"[UNREADABLE FILE - {os.path.basename(resume_file.filename)}]"
                    else:
                        # Extract candidate name using AI (with regex fallback)
                        from analysis import extract_candidate_name_with_gpt
                        candidate_name = extract_candidate_name_with_gpt(resume_text, resume_file.filename)
                    
                    # Store for potential length warning
                    processed_resumes.append({
                        'filename': resume_file.filename,
                        'bytes': resume_bytes,
                        'text': resume_text,
                        'name': candidate_name,
                        'hash': resume_hash
                    })
                    
                    resumes_added += 1
                    
                except Exception as e:
                    # File is locked, corrupted, or otherwise unreadable
                    # Log the error but continue processing other files
                    print(f"ERROR: Could not process {resume_file.filename}: {str(e)}")
                    locked_files.append(resume_file.filename)
                    continue  # Skip this file
            
            # Check resume lengths and show warning if needed
            system_settings = load_system_settings()
            warnings_enabled = system_settings.get('enable_document_length_warnings', True)
            
            if warnings_enabled and processed_resumes:
                from analysis import load_gpt_settings
                gpt_settings = load_gpt_settings()
                resume_limit = gpt_settings.get('candidate_text_chars', 12000)
                
                short_resumes = []  # Too short (likely scanned/corrupted)
                long_resumes = []   # Too long (will be truncated)
                
                for resume in processed_resumes:
                    resume_length = len(resume['text'])
                    resume_text = resume['text']
                    
                    # Check for too short with multiple criteria:
                    # 1. Extremely short (< 50 chars)
                    # 2. High whitespace ratio (> 80% whitespace suggests garbage extraction)
                    is_too_short = False
                    if resume_length < 50:
                        is_too_short = True
                    elif resume_length < 500:  # Only check ratio if suspiciously short
                        non_whitespace = len(resume_text.strip().replace('\n', '').replace(' ', ''))
                        whitespace_ratio = 1.0 - (non_whitespace / max(resume_length, 1))
                        if whitespace_ratio > 0.8:
                            is_too_short = True
                    
                    if is_too_short:
                        short_resumes.append({
                            'name': resume['name'],
                            'length': resume_length
                        })
                    elif resume_length > resume_limit:  # Too long threshold
                        long_resumes.append({
                            'name': resume['name'],
                            'length': resume_length
                        })
                
                if short_resumes or long_resumes:
                    # Store resumes in draft_resume table (not session - too large)
                    for resume in processed_resumes:
                        draft_resume = DraftResume(
                            draft_id=draft.id,
                            file_name=resume['filename'],
                            file_bytes=resume['bytes'],
                            extracted_text=resume['text'],
                            candidate_name=resume['name'],
                            file_hash=resume['hash']
                        )
                        db.session.add(draft_resume)
                    
                    db.session.commit()
                    
                    # Store warning details in session
                    session['resumes_added'] = resumes_added
                    session['resume_limit'] = resume_limit
                    
                    # Check if we already have JD warnings that haven't been confirmed yet
                    # Don't re-show JD warnings if user already confirmed them at JD stage
                    jd_warnings_already_confirmed = session.get('warnings_confirmed', False)
                    has_jd_warnings = (session.get('short_jd', False) or session.get('long_jd', False)) and not jd_warnings_already_confirmed
                    
                    # Route to appropriate warning page
                    if (short_resumes or long_resumes) and has_jd_warnings:
                        # Have both JD and resume warnings - use combined page
                        session['show_document_warnings'] = True
                        session.modified = True
                        return redirect(url_for('analysis.document_warnings_route'))
                    elif short_resumes and not long_resumes:
                        # Only too short - use combined page
                        session['show_document_warnings'] = True
                        session.modified = True
                        return redirect(url_for('analysis.document_warnings_route'))
                    elif long_resumes and not short_resumes:
                        # Only too long - use existing single-issue page
                        session['show_resume_length_warning'] = True
                        session.modified = True
                        return redirect(url_for('analysis.resume_length_warning_route',
                                              truncated_count=len(long_resumes),
                                              resume_limit=resume_limit))
                    else:
                        # Both short and long - use combined page
                        session['show_document_warnings'] = True
                        session.modified = True
                        return redirect(url_for('analysis.document_warnings_route'))
            
            # No warnings needed or warnings disabled - commit resumes
            for resume in processed_resumes:
                draft_resume = DraftResume(
                    draft_id=draft.id,
                    file_name=resume['filename'],
                    file_bytes=resume['bytes'],
                    extracted_text=resume['text'],
                    candidate_name=resume['name'],
                    file_hash=resume['hash']
                )
                db.session.add(draft_resume)
            
            db.session.commit()
            
            # Inform user about results
            if locked_files:
                locked_names = ', '.join(locked_files)
                flash(f'⚠️ Could not upload {len(locked_files)} file(s) because they are open in another program: {locked_names}. Please close these files and try again.', 'warning')
            
            if resumes_added == 0:
                if locked_files:
                    flash('No resumes were uploaded. Please close the locked files and retry.', 'error')
                else:
                    flash('No new resumes added (duplicates skipped)', 'info')
            else:
                flash(f'✅ {resumes_added} resume(s) uploaded successfully!', 'success')
            
            # Only proceed if at least one resume was uploaded
            if resumes_added > 0:
                # Redirect to Run Analysis page
                return redirect(url_for('analysis.run_analysis_route'))
            else:
                # No resumes uploaded - stay on upload page
                return redirect(url_for('analysis.analyze', step='resumes'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Resume upload failed: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('analysis.analyze', step='resumes'))
    
    elif action == 'confirm_resume_length':
        # User confirmed to proceed with long resumes (already saved in DB)
        try:
            # Check we have the session flag
            if not session.get('show_resume_length_warning'):
                flash('Session expired. Please upload your resumes again.', 'error')
                return redirect(url_for('analysis.analyze', step='resumes'))
            
            # Get draft
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if not draft:
                flash('Draft not found. Please start over.', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Resumes already in database, clear session flags
            resumes_added = session.get('resumes_added', 0)
            session.pop('show_resume_length_warning', None)
            session.pop('resumes_added', None)
            
            # Mark that user has already confirmed truncation for this draft
            # So run_analysis doesn't show the warning again
            session['truncation_confirmed'] = True
            session.modified = True
            
            flash(f'✅ {resumes_added} resume(s) added to analysis!', 'success')
            return redirect(url_for('analysis.run_analysis_route'))
            
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('analysis.analyze', step='resumes'))
    
    elif action == 'confirm_document_warnings':
        # User confirmed to proceed despite document warnings (too short and/or too long)
        try:
            # Check we have the session flag
            if not session.get('show_document_warnings'):
                flash('Session expired. Please start over.', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Get draft
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if not draft:
                flash('Draft not found. Please start over.', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Clear all warning flags
            short_jd = session.pop('short_jd', False)
            long_jd = session.pop('long_jd', False)
            session.pop('show_document_warnings', None)
            session.pop('jd_length', None)
            session.pop('jd_limit', None)
            session.pop('resume_limit', None)
            
            # Mark that user has confirmed warnings
            session['warnings_confirmed'] = True
            session.modified = True
            
            # Determine next step based on what was uploaded
            has_resumes = DraftResume.query.filter_by(draft_id=draft.id).count() > 0
            
            if short_jd or long_jd:
                # Had JD warnings - need to extract criteria if not done already
                if not draft.criteria_data:
                    # Extract criteria from JD
                    from analysis import extract_jd_sections_with_gpt, build_criteria_from_sections
                    
                    sections = extract_jd_sections_with_gpt(draft.jd_text)
                    criteria, cat_map = build_criteria_from_sections(sections, per_section=999, cap_total=10000)
                    
                    if not criteria:
                        flash('Could not extract criteria from job description', 'error')
                        return redirect(url_for('analysis.analyze'))
                    
                    # Use AI-extracted job title from sections
                    job_title = sections.job_title or "Position Not Specified"
                    
                    # Update draft with criteria
                    draft.job_title = job_title
                    draft.criteria_data = json.dumps([
                        {'criterion': crit, 'category': cat_map.get(crit, 'Other Requirements'), 'use': True}
                        for crit in criteria
                    ])
                    db.session.commit()
                    
                    flash(f'✅ JD processed! Extracted {len(criteria)} criteria. Review them on the Job Criteria page.', 'success')
                    return redirect(url_for('analysis.review_criteria'))
                else:
                    # Criteria already extracted, go to next step
                    if has_resumes:
                        # Have everything, go to run analysis
                        resumes_added = session.pop('resumes_added', 0)
                        if resumes_added:
                            flash(f'✅ {resumes_added} resume(s) added to analysis!', 'success')
                        return redirect(url_for('analysis.run_analysis_route'))
                    else:
                        # Need resumes
                        return redirect(url_for('analysis.analyze', step='resumes'))
            else:
                # Only had resume warnings
                resumes_added = session.pop('resumes_added', 0)
                if resumes_added:
                    flash(f'✅ {resumes_added} resume(s) added to analysis!', 'success')
                return redirect(url_for('analysis.run_analysis_route'))
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('analysis.analyze'))
            
            flash(f'✅ {resumes_added} resume(s) uploaded successfully!', 'success')
            return redirect(url_for('analysis.run_analysis_route'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Resume confirmation failed: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('analysis.analyze', step='resumes'))
            return redirect(url_for('analysis.analyze', step='resumes'))
    
    elif action == 'run_analysis':
        # Legacy handler - redirect to proper flow
        flash('Please upload resumes first', 'info')
        return redirect(url_for('analysis.analyze', step='resumes'))
    
    elif action == 'run_analysis':
        # Step 2: Run full analysis with resumes
        try:
            from analysis import (
                read_file_bytes, hash_bytes, Candidate,
                infer_candidate_name
            )
            
            # Check we have JD and criteria in database
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if not draft or not draft.criteria_data:
                flash('Please upload a JD first and review criteria', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Get resume files
            resume_files = request.files.getlist('resumes')
            if not resume_files:
                flash('Please upload at least one resume', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Get options
            insights_mode = request.form.get('insights_mode', 'top3')
            
            # Read JD from draft
            jd_text = draft.jd_text
            
            # Get active criteria from draft
            criteria_list = json.loads(draft.criteria_data)
            criteria = [c['criterion'] for c in criteria_list if c.get('use', True)]
            
            if not criteria:
                flash('No criteria selected. Please review criteria first.', 'error')
                return redirect(url_for('analysis.review_criteria'))
            
            # Read candidates
            candidates = []
            for resume_file in resume_files:
                resume_bytes = resume_file.read()
                resume_text = read_file_bytes(resume_bytes, resume_file.filename)
                resume_hash = hash_bytes(resume_bytes)
                candidate_name = infer_candidate_name(resume_file.filename, resume_text)
                candidates.append(Candidate(
                    name=candidate_name,
                    file_name=resume_file.filename,
                    text=resume_text,
                    hash=resume_hash,
                    raw_bytes=resume_bytes
                ))
            
            # Determine number of insights to generate
            num_candidates = len(candidates)
            if insights_mode == 'top3':
                num_insights = min(3, num_candidates)
            elif insights_mode == 'top5':
                num_insights = min(5, num_candidates)
            elif insights_mode == 'top10':
                num_insights = min(10, num_candidates)
            elif insights_mode == 'all':
                num_insights = num_candidates
            else:
                num_insights = 0
            
            # Calculate cost
            from config import Config
            pricing = Config.get_pricing()
            total_cost = pricing['BASE_ANALYSIS_PRICE']
            num_extra_insights = max(0, num_insights - 3)
            total_cost += num_extra_insights * pricing['EXTRA_INSIGHT_PRICE']
            total_cost = Decimal(str(total_cost))
            
            # Check funds
            if current_user.balance_usd < total_cost:
                flash(f'Insufficient funds. You need ${total_cost:.2f} but only have ${current_user.balance_usd:.2f}.', 'error')
                return redirect(url_for('analysis.analyze'))
            
            # Deduct funds (we'll link the analysis_id after creating the analysis)
            current_user.balance_usd -= total_cost
            
            # Create transaction (analysis_id will be linked later)
            transaction = Transaction(
                user_id=current_user.id,
                amount_usd=-total_cost,
                transaction_type='debit',
                description=f'Analysis: {len(candidates)} candidates, {len(criteria)} criteria'
            )
            db.session.add(transaction)
            
            # Infer job title using intelligent extraction
            jd_lines = [line.strip() for line in jd_text.split('\n') if line.strip()]
            job_title = "Position Not Specified"
            
            # Look for common job title patterns in first 5 lines
            for line in jd_lines[:5]:
                # Skip very short lines, URLs, dates
                if len(line) < 10 or 'http' in line.lower() or any(char.isdigit() for char in line[:4]):
                    continue
                # Look for title indicators
                if any(indicator in line.lower() for indicator in ['position:', 'role:', 'job title:', 'title:']):
                    job_title = line.split(':', 1)[1].strip() if ':' in line else line
                    break
                # Or just use first substantial line (likely the title)
                elif len(line) > 15 and len(line) < 100:
                    job_title = line
                    break
            
            # ============================================
            # NEW AI PIPELINE: Production-Grade Scoring
            # ============================================
            import asyncio
            from ai_service import run_global_ranking, run_deep_insights
            
            # Check document lengths and warn user if truncation will occur
            from analysis import load_gpt_settings
            gpt_settings = load_gpt_settings()
            jd_limit = gpt_settings['jd_text_chars']
            resume_limit = gpt_settings['candidate_text_chars']
            
            jd_length = len(jd_text)
            truncated_docs = []
            
            # Check JD length (but skip if user already saw warning at upload stage)
            jd_already_warned = session.get('show_jd_length_warning', False)
            if jd_length > jd_limit and not jd_already_warned:
                truncated_docs.append({
                    'name': 'Job Description',
                    'length': jd_length,
                    'limit': jd_limit
                })
            
            # Check resume lengths
            long_resumes = [(c.name, len(c.text)) for c in candidates if len(c.text) > resume_limit]
            for name, length in long_resumes:
                truncated_docs.append({
                    'name': f'Resume: {name}',
                    'length': length,
                    'limit': resume_limit
                })
            
            # If documents exceed limits, show informational banner
            # NOTE: This is legacy code - /run-analysis route is now the primary analysis execution path
            if truncated_docs:
                doc_list = ", ".join([f"{d['name']} ({d['length']:,} chars)" for d in truncated_docs])
                flash(f"ℹ️ Document length notice: {doc_list} exceed our limits and will be automatically trimmed to maintain optimal performance. Analysis quality remains high.", 'info')
            
            # Track start time for duration calculation
            analysis_start_time = datetime.utcnow()
            
            # Check if user exceeded resume limit (for analytics)
            exceeded_limit = len(candidates) > 200
            chose_override = request.form.get('override_limit') == 'true'
            
            # Calculate document size metrics for analytics
            jd_char_count = len(jd_text)
            resume_char_counts = [len(c.text) for c in candidates]
            avg_resume_chars = int(sum(resume_char_counts) / len(resume_char_counts)) if resume_char_counts else 0
            min_resume_chars = min(resume_char_counts) if resume_char_counts else 0
            max_resume_chars = max(resume_char_counts) if resume_char_counts else 0
            
            # Create Analysis record for progress tracking
            analysis = Analysis(
                user_id=current_user.id,
                job_title=job_title,
                job_description_text=jd_text[:5000],
                num_candidates=len(candidates),
                num_criteria=len(criteria),
                coverage_data='',  # Will be populated after Phase 1
                insights_data='',  # Will be populated after Phase 2
                evidence_data='',  # Will be populated after Phase 1
                criteria_list=json.dumps(criteria),
                cost_usd=total_cost,
                analysis_size='small' if len(candidates) <= 5 else ('medium' if len(candidates) <= 15 else 'large'),
                resumes_processed=0,
                exceeded_resume_limit=exceeded_limit,
                user_chose_override=chose_override,
                jd_character_count=jd_char_count,
                avg_resume_character_count=avg_resume_chars,
                min_resume_character_count=min_resume_chars,
                max_resume_character_count=max_resume_chars
            )
            db.session.add(analysis)
            db.session.flush()  # Get analysis.id
            
            # Link transaction to analysis
            transaction.analysis_id = analysis.id
            transaction.description = f"[Job #{analysis.id:04d}] - {job_title}"
            
            # Progress callback for live updates
            def update_progress(completed, total):
                """Update database with progress"""
                analysis.resumes_processed = completed
                db.session.commit()
                print(f"Progress: {completed}/{total} candidates scored")
            
            # PHASE 1: Global Ranking (All Candidates)
            print("Phase 1: AI scoring all candidates...")
            candidate_tuples = [(c.name, c.text) for c in candidates]
            
            evaluations = asyncio.run(
                run_global_ranking(
                    candidates=candidate_tuples,
                    jd_text=jd_text,
                    criteria=criteria_list,  # Pass full criteria with metadata
                    progress_callback=update_progress
                )
            )
            
            print(f"Phase 1 complete. Top candidate: {evaluations[0].candidate_name} ({evaluations[0].overall_score:.1f}/100)")
            
            # PHASE 2: Deep Insights (Top N Only)
            insights_data = {}
            gpt_candidates_list = []
            
            if num_insights > 0:
                print(f"Phase 2: Generating deep insights for top {num_insights}...")
                
                insights_data = asyncio.run(
                    run_deep_insights(
                        candidates=candidate_tuples,
                        jd_text=jd_text,
                        evaluations=evaluations,
                        top_n=num_insights
                    )
                )
                
                gpt_candidates_list = list(insights_data.keys())
                print(f"Phase 2 complete. Generated insights for {len(insights_data)} candidates")
                
                # CRITICAL: Overwrite draft justifications with INSIGHT_AGENT versions
                for candidate_name, insights in insights_data.items():
                    # Find evaluation for this candidate
                    eval_obj = next((e for e in evaluations if e.candidate_name == candidate_name), None)
                    if eval_obj and 'justifications' in insights:
                        # Replace Phase 1 draft justifications with Phase 2 premium versions
                        for criterion, refined_justification in insights['justifications'].items():
                            # Update the evaluation object
                            score_obj = next((s for s in eval_obj.criterion_scores if s.criterion == criterion), None)
                            if score_obj:
                                score_obj.justification = refined_justification
            
            # Convert evaluations to DataFrame format (for existing UI compatibility)
            import pandas as pd
            coverage_records = []
            evidence_map = {}
            
            for eval_obj in evaluations:
                row = {"Candidate": eval_obj.candidate_name}
                
                # Add criterion scores
                for score in eval_obj.criterion_scores:
                    row[score.criterion] = score.score / 100.0  # Convert to 0-1
                    
                    # Store justification as evidence (for UI compatibility)
                    evidence_map[(eval_obj.candidate_name, score.criterion)] = (
                        score.justification,  # Justification text
                        score.score / 100.0,  # Score
                        1  # Density count (not applicable in AI pipeline)
                    )
                
                row["Overall"] = eval_obj.overall_score / 100.0
                coverage_records.append(row)
            
            coverage = pd.DataFrame(coverage_records)
            
            # Build category map from criteria_list
            category_map = {c['criterion']: c.get('category', 'Other Requirements') for c in criteria_list if c.get('use', True)}
            
            # Update analysis with results
            analysis.coverage_data = coverage.to_json(orient='records')
            analysis.insights_data = json.dumps(insights_data)
            analysis.evidence_data = json.dumps({f"{k[0]}|||{k[1]}": v for k, v in evidence_map.items()})
            analysis.category_map = json.dumps(category_map)
            analysis.gpt_candidates = json.dumps(gpt_candidates_list)
            
            # Capture completion time and duration for analytics
            analysis.completed_at = datetime.utcnow()
            analysis.processing_duration_seconds = int((analysis.completed_at - analysis_start_time).total_seconds())
            
            db.session.add(analysis)
            
            # Flush to get analysis.id before linking to transaction
            db.session.flush()
            
            # Store candidate files for viewing
            from database import CandidateFile
            for candidate in candidates:
                candidate_file = CandidateFile(
                    analysis_id=analysis.id,
                    candidate_name=candidate.name,
                    file_name=candidate.file_name,
                    file_bytes=candidate.raw_bytes,
                    extracted_text=candidate.text
                )
                db.session.add(candidate_file)
            
            # Keep draft so users can navigate back to see uploaded files
            # Don't delete it - it will be overwritten on next analysis
            db.session.commit()
            
            flash(f'Analysis complete! Cost: ${total_cost:.2f}. Remaining balance: ${current_user.balance_usd:.2f}', 'success')
            return redirect(url_for('analysis.results', analysis_id=analysis.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Analysis failed: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('analysis.analyze'))
    
    else:
        flash('Invalid action', 'error')
        return redirect(url_for('analysis.analyze'))

@analysis_bp.route('/clear-session', methods=['POST'])
@login_required
def clear_session():
    """Clear JD and criteria from database"""
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    if draft:
        db.session.delete(draft)
        db.session.commit()
    flash('Draft cleared. You can upload a new JD.', 'info')
    return redirect(url_for('analysis.analyze'))

@analysis_bp.route('/delete-resume/<int:resume_id>', methods=['POST'])
@login_required
def delete_resume(resume_id):
    """Delete a specific resume from draft"""
    resume = DraftResume.query.get_or_404(resume_id)
    
    # Verify ownership through draft
    draft = Draft.query.filter_by(id=resume.draft_id, user_id=current_user.id).first()
    if not draft:
        flash('Resume not found', 'error')
        return redirect(url_for('analysis.analyze', step='resumes'))
    
    db.session.delete(resume)
    db.session.commit()
    flash(f'Removed resume: {resume.file_name}', 'success')
    return redirect(url_for('analysis.analyze', step='resumes'))

@analysis_bp.route('/load-analysis-to-draft/<int:analysis_id>', methods=['POST'])
@login_required
def load_analysis_to_draft(analysis_id):
    """Load a historical analysis into draft for editing and re-running"""
    # Get the analysis
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.job_history'))
    
    # Check if user already has a draft
    existing_draft = Draft.query.filter_by(user_id=current_user.id).first()
    if existing_draft:
        # Clear existing draft and resumes
        DraftResume.query.filter_by(draft_id=existing_draft.id).delete()
        db.session.delete(existing_draft)
        db.session.flush()
    
    # Create new draft from analysis data
    draft = Draft(user_id=current_user.id)
    draft.jd_filename = analysis.jd_filename or f"{analysis.job_title}.txt"
    draft.jd_text = analysis.jd_full_text or analysis.job_description_text
    draft.jd_bytes = analysis.jd_bytes  # Restore PDF bytes if available
    draft.jd_hash = hashlib.md5(draft.jd_text.encode()).hexdigest()
    draft.job_title = analysis.job_title
    
    # Restore criteria_data from analysis (preserves checked/unchecked state)
    criteria_list = json.loads(analysis.criteria_list)
    
    # Check if criteria_list is already in full format (with 'use' flags) or old format (just strings)
    if criteria_list and isinstance(criteria_list[0], dict):
        # New format - already has full structure, use as-is
        draft.criteria_data = analysis.criteria_list
    else:
        # Old format (just strings) - rebuild with category_map and default to checked
        category_map = json.loads(analysis.category_map) if analysis.category_map else {}
        criteria_data = []
        for criterion in criteria_list:
            criteria_data.append({
                'criterion': criterion,
                'category': category_map.get(criterion, 'Other Requirements'),
                'use': True
            })
        draft.criteria_data = json.dumps(criteria_data)
    
    db.session.add(draft)
    db.session.flush()
    
    # Load candidate files from analysis into DraftResume
    candidate_files = CandidateFile.query.filter_by(analysis_id=analysis_id).all()
    for cf in candidate_files:
        draft_resume = DraftResume(
            draft_id=draft.id,
            file_name=cf.file_name,
            file_bytes=cf.file_bytes,
            extracted_text=cf.extracted_text,
            candidate_name=cf.candidate_name,
            file_hash=hashlib.md5(cf.file_bytes).hexdigest()
        )
        db.session.add(draft_resume)
    
    db.session.commit()
    
    flash(f'✏️ Loaded "{analysis.job_title}" to draft. You can now edit and re-run the analysis.', 'info')
    return redirect(url_for('analysis.review_criteria'))

@analysis_bp.route('/run-analysis', methods=['GET', 'POST'])
@login_required
def run_analysis_route():
    """Configure and run analysis on uploaded resumes"""
    from flask import session
    import secrets
    
    if request.method == 'GET':
        # Check we have draft with JD, criteria, and resumes
        draft = Draft.query.filter_by(user_id=current_user.id).first()
        if not draft or not draft.criteria_data:
            flash('Please upload a JD and review criteria first', 'error')
            return redirect(url_for('analysis.analyze'))
        
        # Check if returning from payment with auto_submit flag
        auto_submit = request.args.get('auto_submit') == '1'
        saved_insights_mode = session.pop('pending_insights_mode', None)
        
        # Get uploaded resumes count
        resume_count = DraftResume.query.filter_by(draft_id=draft.id).count()
        if resume_count == 0:
            flash('Please upload at least one resume', 'error')
            return redirect(url_for('analysis.analyze', step='resumes'))
        
        # Get most recent analysis for workflow navigation
        latest_analysis = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).first()
        
        # Check if CURRENT draft has been analyzed (draft created before latest analysis)
        analysis_completed = False
        draft_modified_after_analysis = False
        current_draft_analysis_id = None
        
        if latest_analysis and draft.created_at < latest_analysis.created_at:
            analysis_completed = True
            current_draft_analysis_id = latest_analysis.id
            # Check if draft was updated after the analysis
            if draft.updated_at > latest_analysis.created_at:
                draft_modified_after_analysis = True
        
        # Generate a unique form token to prevent duplicate submissions
        # Only generate a NEW token if we don't already have one (prevents double-charge on payment return)
        existing_token = session.get('analysis_form_token')
        if not existing_token:
            form_token = secrets.token_hex(16)
            session['analysis_form_token'] = form_token
        else:
            form_token = existing_token
        
        from config import Config
        pricing = Config.get_pricing()
        
        # REMOVED: Truncation warning display code
        # Warnings now handled during upload only (lines 590-650)
        
        return render_template('run_analysis.html', 
                             user=current_user,
                             resume_count=resume_count,
                             latest_analysis_id=current_draft_analysis_id,
                             in_workflow=True, has_unsaved_work=True,
                             analysis_completed=analysis_completed,
                             draft_modified_after_analysis=draft_modified_after_analysis,
                             form_token=form_token,
                             pricing=pricing)
    
    # POST - Run the analysis
    try:
        from analysis import Candidate
        from config import Config
        
        # CRITICAL: Check form token to prevent duplicate submissions
        submitted_token = request.form.get('form_token')
        expected_token = session.get('analysis_form_token')
        
        if not submitted_token or submitted_token != expected_token:
            # Token missing or invalid - likely duplicate submission
            flash('This analysis has already been processed. Redirecting to results...', 'info')
            # Try to find the most recent analysis for this user
            recent_analysis = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).first()
            if recent_analysis:
                return redirect(url_for('analysis.results', analysis_id=recent_analysis.id))
            else:
                return redirect(url_for('dashboard.dashboard'))
        
        # NOTE: Token will be consumed AFTER document length check
        # (so it can be restored if we need to show truncation warning)
        
        # Get configuration
        insights_mode = request.form.get('insights_mode', 'top3')
        
        # Get draft and validate
        draft = Draft.query.filter_by(user_id=current_user.id).first()
        if not draft or not draft.criteria_data:
            flash('Please upload a JD and review criteria first', 'error')
            return redirect(url_for('analysis.analyze'))
        
        # Get resumes from DraftResume table
        draft_resumes = DraftResume.query.filter_by(draft_id=draft.id).all()
        if not draft_resumes:
            flash('No resumes found. Please upload resumes first.', 'error')
            return redirect(url_for('analysis.analyze', step='resumes'))
        
        # Convert DraftResumes to Candidate objects
        candidates = []
        for dr in draft_resumes:
            candidates.append(Candidate(
                name=dr.candidate_name,
                file_name=dr.file_name,
                text=dr.extracted_text,
                hash=dr.file_hash,
                raw_bytes=dr.file_bytes
            ))
        
        # Get active criteria
        jd_text = draft.jd_text
        criteria_list = json.loads(draft.criteria_data)
        criteria = [c['criterion'] for c in criteria_list if c.get('use', True)]
        
        if not criteria:
            flash('No criteria selected. Please review criteria first.', 'error')
            return redirect(url_for('analysis.review_criteria'))
        
        # CRITICAL: Calculate cost and check balance BEFORE running analysis
        num_candidates = len(candidates)
        
        # Load pricing from admin settings
        pricing = load_pricing_settings()
        standard_price = Decimal(str(pricing['standard_tier_price']['value']))
        deep_dive_price = Decimal(str(pricing['deep_dive_price']['value']))
        individual_insight_price = Decimal(str(pricing['individual_insight_price']['value']))
        
        # Initialize cost variable
        estimated_cost = standard_price  # Default to standard tier
        
        # NEW PRICING MODEL: Calculate cost based on tier
        if insights_mode == 'standard':
            # Standard: base price (includes Top 5 insights)
            estimated_cost = standard_price
            num_insights = min(5, num_candidates)
        elif insights_mode == 'deep_dive':
            # Deep Dive: standard + deep dive price (includes Top 15 insights)
            estimated_cost = standard_price + deep_dive_price
            num_insights = min(15, num_candidates)
        elif insights_mode == 'full_radar':
            # Full Radar: standard price + individual price per candidate beyond 5
            extra_candidates = max(0, num_candidates - 5)
            estimated_cost = standard_price + (individual_insight_price * extra_candidates)
            num_insights = num_candidates
        # Legacy support for old values (map to new tiers)
        elif insights_mode in ['top3', 'top5']:
            estimated_cost = standard_price
            num_insights = min(5, num_candidates)
        elif insights_mode == 'top10':
            estimated_cost = standard_price + deep_dive_price
            num_insights = min(15, num_candidates)
        elif insights_mode == 'all':
            extra_candidates = max(0, num_candidates - 5)
            estimated_cost = standard_price + (individual_insight_price * extra_candidates)
            num_insights = num_candidates
        else:
            estimated_cost = standard_price
            num_insights = 0
        
        # Check funds BEFORE running analysis
        if current_user.balance_usd < estimated_cost:
            # Save insights mode to session for restoration after payment
            from flask import session
            session['pending_insights_mode'] = insights_mode
            
            # Don't redirect - show modal to add funds and continue
            resume_count = len(candidates)
            from config import Config
            pricing = Config.get_pricing()
            return render_template('run_analysis.html',
                                 user=current_user,
                                 resume_count=resume_count,
                                 latest_analysis_id=None,
                                 in_workflow=True,
                                 has_unsaved_work=True,
                                 analysis_completed=False,
                                 insufficient_funds=True,
                                 required_amount=float(estimated_cost),
                                 current_balance=float(current_user.balance_usd),
                                 shortfall=float(estimated_cost - current_user.balance_usd),
                                 selected_insights_mode=insights_mode,
                                 pricing=pricing)
        
        # ============================================
        # NEW AI PIPELINE: Production-Grade Scoring
        # ============================================
        import asyncio
        from ai_service import run_global_ranking, run_deep_insights
        
        # Use job title from draft (already extracted/edited by user)
        job_title = draft.job_title or "Position Not Specified"
        
        # REMOVED: Redundant truncation warning check
        # Warnings are now handled comprehensively during upload (lines 590-650)
        # No need to check again at Run Analysis stage
        # User already confirmed any warnings before reaching this point
        # NOW consume the token
        session.pop('analysis_form_token', None)
        
        # Track start time for duration calculation
        analysis_start_time = datetime.utcnow()
        
        # Check if user exceeded resume limit (for analytics)
        exceeded_limit = len(candidates) > 200
        chose_override = request.form.get('override_limit') == 'true'
        
        # Calculate document size metrics for analytics
        jd_char_count = len(jd_text)
        resume_char_counts = [len(c.text) for c in candidates]
        avg_resume_chars = int(sum(resume_char_counts) / len(resume_char_counts)) if resume_char_counts else 0
        min_resume_chars = min(resume_char_counts) if resume_char_counts else 0
        max_resume_chars = max(resume_char_counts) if resume_char_counts else 0
        
        # Create Analysis record for progress tracking
        analysis = Analysis(
            user_id=current_user.id,
            job_title=job_title,
            job_description_text=jd_text[:5000],
            jd_full_text=jd_text,
            jd_filename=draft.jd_filename,
            jd_bytes=draft.jd_bytes,  # Save PDF bytes for preview in history
            num_candidates=len(candidates),
            num_criteria=len(criteria),
            coverage_data='',  # Will be populated after Phase 1
            insights_data='',  # Will be populated after Phase 2
            evidence_data='',  # Will be populated after Phase 1
            criteria_list=json.dumps(criteria_list),  # Store full criteria_data with use flags, not just filtered list
            cost_usd=estimated_cost,
            analysis_size='phase1',  # Track progress phase
            resumes_processed=0,
            exceeded_resume_limit=exceeded_limit,
            user_chose_override=chose_override,
            jd_character_count=jd_char_count,
            avg_resume_character_count=avg_resume_chars,
            min_resume_character_count=min_resume_chars,
            max_resume_character_count=max_resume_chars
        )
        db.session.add(analysis)
        db.session.flush()  # Get analysis.id
        db.session.commit()  # Commit so frontend polling can see progress
        # NOTE: This creates an "incomplete" analysis record. If the job fails,
        # it won't be rolled back, but we filter job history to only show
        # completed analyses (those with coverage_data populated).
        
        # Progress callback for live updates
        def update_progress(completed, total):
            """Update database with progress"""
            analysis.resumes_processed = completed
            db.session.commit()
            print(f"Progress: {completed}/{total} candidates scored")
        
        # PHASE 1: Global Ranking (All Candidates)
        print("Phase 1: AI scoring all candidates...")
        candidate_tuples = [(c.name, c.text) for c in candidates]
        
        # Load GPT settings to respect user's configured text limits
        from analysis import load_gpt_settings
        gpt_settings = load_gpt_settings()
        jd_text_limit = gpt_settings.get('jd_text_chars', 15000)
        
        # DIAGNOSTIC: Show what text GPT receives from each resume
        for cand_name, cand_text in candidate_tuples:
            text_len = len(cand_text) if cand_text else 0
            print(f"\n📄 Resume: {cand_name}")
            print(f"   Text length: {text_len} characters")
            if text_len > 0:
                preview = cand_text[:500].replace('\n', ' ')
                print(f"   First 500 chars: {preview}")
            else:
                print(f"   ⚠️  NO TEXT EXTRACTED")
        print(f"\n📋 JD text length: {len(jd_text)} characters (limit: {jd_text_limit})\n")
        
        # CRITICAL FIX: Filter criteria_list to only include used criteria before passing to AI
        # This prevents coverage DataFrame from having columns for unchecked criteria
        criteria_for_ai = [c for c in criteria_list if c.get('use', True)]
        
        evaluations = asyncio.run(
            run_global_ranking(
                candidates=candidate_tuples,
                jd_text=jd_text[:jd_text_limit],  # Respect admin setting, not hardcoded limit
                criteria=criteria_for_ai,  # Pass only used criteria
                progress_callback=update_progress
            )
        )
        
        print(f"Phase 1 complete. Top candidate: {evaluations[0].candidate_name} ({evaluations[0].overall_score:.1f}/100)")
        
        # PHASE 2: Deep Insights (Top N Only)
        insights_data = {}
        gpt_candidates_list = []
        
        if num_insights > 0:
            print(f"Phase 2: Generating deep insights for top {num_insights}...")
            
            # Mark Phase 2 start
            analysis.analysis_size = 'phase2'
            db.session.commit()
            
            insights_data = asyncio.run(
                run_deep_insights(
                    candidates=candidate_tuples,
                    jd_text=jd_text,
                    evaluations=evaluations,
                    top_n=num_insights
                )
            )
            
            gpt_candidates_list = list(insights_data.keys())
            print(f"Phase 2 complete. Generated insights for {len(insights_data)} candidates")
            
            # CRITICAL: Overwrite draft justifications with INSIGHT_AGENT versions
            for candidate_name, insights in insights_data.items():
                # Find evaluation for this candidate
                eval_obj = next((e for e in evaluations if e.candidate_name == candidate_name), None)
                if eval_obj and 'justifications' in insights:
                    # Replace Phase 1 draft justifications with Phase 2 premium versions
                    for criterion, refined_justification in insights['justifications'].items():
                        # Update the evaluation object
                        score_obj = next((s for s in eval_obj.criterion_scores if s.criterion == criterion), None)
                        if score_obj:
                            score_obj.justification = refined_justification
            
            # Mark Phase 2 complete
            analysis.analysis_size = 'complete'
            db.session.commit()
        else:
            # No Phase 2, mark as complete
            analysis.analysis_size = 'complete'
            db.session.commit()
        
        # Convert evaluations to DataFrame format (for existing UI compatibility)
        import pandas as pd
        coverage_records = []
        evidence_map = {}
        
        for eval_obj in evaluations:
            row = {"Candidate": eval_obj.candidate_name}
            
            # DIAGNOSTIC: Check if this is an unreadable file
            if 'UNREADABLE' in eval_obj.candidate_name:
                print(f"\n⚠️  Building coverage for UNREADABLE FILE: {eval_obj.candidate_name}")
                print(f"   Overall score: {eval_obj.overall_score}/100")
                print(f"   Number of criterion scores: {len(eval_obj.criterion_scores)}")
                print(f"   ALL CRITERION SCORES:")
                for idx, score in enumerate(eval_obj.criterion_scores):
                    print(f"     [{idx}] {score.criterion}: {score.score}/100")
                    if score.score > 0:
                        print(f"         Evidence: {score.raw_evidence[:150]}...")
                print(f"   ---")
            
            # Add criterion scores
            for score in eval_obj.criterion_scores:
                row[score.criterion] = score.score / 100.0  # Convert to 0-1
                
                # Store evidence in new structure: (raw_evidence, justification, score, density_count)
                # NOTE: justification already contains Phase 2 refined version if insights were generated
                evidence_map[(eval_obj.candidate_name, score.criterion)] = (
                    score.raw_evidence,     # [0] Verbatim quotes from resume ("Raw Source Text")
                    score.justification,    # [1] AI reasoning (Phase 2 refined for top N, Phase 1 for others)
                    score.score / 100.0,    # [2] Score
                    1                       # [3] Density count (deprecated but kept for compatibility)
                )
            
            row["Overall"] = eval_obj.overall_score / 100.0
            coverage_records.append(row)
        
        coverage = pd.DataFrame(coverage_records)
        
        # Build category map
        category_map = {c['criterion']: c.get('category', 'Other Requirements') for c in criteria_list if c.get('use', True)}
        
        # Update analysis with results
        analysis.coverage_data = coverage.to_json(orient='records')
        analysis.insights_data = json.dumps(insights_data)
        analysis.evidence_data = json.dumps({f"{k[0]}|||{k[1]}": v for k, v in evidence_map.items()})
        analysis.category_map = json.dumps(category_map)
        analysis.gpt_candidates = json.dumps(gpt_candidates_list)
        
        # Capture completion time and duration for analytics
        analysis.completed_at = datetime.utcnow()
        analysis.processing_duration_seconds = int((analysis.completed_at - analysis_start_time).total_seconds())
        
        # Store candidate files
        from database import CandidateFile
        for candidate in candidates:
            candidate_file = CandidateFile(
                analysis_id=analysis.id,
                candidate_name=candidate.name,
                file_name=candidate.file_name,
                file_bytes=candidate.raw_bytes,
                extracted_text=candidate.text
            )
            db.session.add(candidate_file)
        
        # CRITICAL: Charge user ONLY after all work succeeded
        # If anything above failed, we never reach this point (exception caught below)
        current_user.balance_usd -= estimated_cost
        
        # Create transaction record (linked to analysis)
        transaction = Transaction(
            user_id=current_user.id,
            amount_usd=-estimated_cost,
            transaction_type='debit',
            description=f'[Job #{analysis.id:04d}] - {job_title}',
            analysis_id=analysis.id
        )
        db.session.add(transaction)
        
        # Update user analytics tracking
        current_user.total_analyses_count += 1
        current_user.total_revenue_usd += estimated_cost
        
        # Commit everything atomically (if this fails, EVERYTHING rolls back including charge)
        db.session.commit()
        
        flash(f'✅ Analysis complete! Cost: ${estimated_cost:.2f}. Remaining balance: ${current_user.balance_usd:.2f}', 'success')
        
        # Return JSON redirect for fetch to handle cleanly
        return jsonify({'redirect': url_for('analysis.results', analysis_id=analysis.id)})
        
    except Exception as e:
        # Capture error details for admin diagnostics
        import traceback
        error_trace = traceback.format_exc()
        
        try:
            # Save error info to the analysis record before rollback
            analysis.error_message = f"{str(e)}\n\n{error_trace}"
            analysis.failed_at = datetime.utcnow()
            db.session.commit()  # Commit error details
            print(f"ERROR: Analysis #{analysis.id} failed. Error details saved.")
        except:
            print(f"ERROR: Could not save error details for analysis")
        
        # Rollback the main transaction (user not charged, no results saved)
        db.session.rollback()
        print(f"ERROR: Analysis failed, transaction rolled back. User NOT charged.")
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        
        # Try to extract candidate name from error message for actionable feedback
        error_msg = str(e)
        candidate_name = None
        
        # Check if error mentions a specific candidate
        if "for " in error_msg and ("candidate" in error_msg.lower() or "resume" in error_msg.lower()):
            # Try to extract candidate name from patterns like "...for John Smith" or "...candidate John Smith..."
            import re
            # Pattern: "for [name]" at end of sentence or before punctuation
            match = re.search(r'for\s+([A-Z][a-zA-Z\s]+?)(?:\.|$|,|\s+\()', error_msg)
            if match:
                candidate_name = match.group(1).strip()
        
        # Build user-friendly error message
        if candidate_name:
            flash(f'⚠️ Analysis failed while processing candidate "{candidate_name}". This resume may contain unusual formatting or characters. You have NOT been charged. Try: (1) Remove this candidate and re-run, or (2) Re-upload their resume with simpler formatting (plain text works best).', 'error')
        else:
            # Generic error message when we can't identify the problematic candidate
            flash('⚠️ Analysis failed due to a processing error. You have NOT been charged. Our team has been notified. Please try again with different resumes or a simplified job description, or contact support if the issue persists.', 'error')
        
        return redirect(url_for('analysis.run_analysis_route'))

@analysis_bp.route('/results/<int:analysis_id>')
@login_required
def results(analysis_id):
    """Display analysis results with enhanced coverage matrix"""
    # Allow admin to view any user's analysis, otherwise restrict to own analysis
    is_admin = session.get('admin_logged_in', False)
    if is_admin:
        analysis = Analysis.query.filter_by(id=analysis_id).first()
    else:
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Check if admin is viewing another user's analysis
    viewing_other_user = is_admin and analysis.user_id != current_user.id
    analysis_owner_email = None
    if viewing_other_user:
        from database import User
        owner = User.query.get(analysis.user_id)
        analysis_owner_email = owner.email if owner else "Unknown User"
    
    # Get user settings for threshold preferences
    user_settings = UserSettings.get_or_create(current_user.id)
    
    # Parse stored JSON data
    import pandas as pd
    from io import StringIO
    coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
    insights = json.loads(analysis.insights_data)
    criteria_raw = json.loads(analysis.criteria_list)
    
    # CRITICAL FIX: Filter to only criteria that were actually used (use=True)
    # The stored criteria_list includes ALL criteria with use flags, but we only want used ones
    criteria = []
    for c in criteria_raw:
        if isinstance(c, dict):
            # New format with 'use' flag - only include if use=True
            if c.get('use', True):
                criteria.append(c['criterion'])
        else:
            # Old format (plain string) - always include
            criteria.append(c)
    
    # Parse evidence_map and normalize to pipe format for template
    evidence_map_raw = json.loads(analysis.evidence_data)
    evidence_map = {}
    
    for key_str, value in evidence_map_raw.items():
        # Handle both formats: pipe "Name|||Criterion" or tuple "('Name', 'Criterion')"
        if '|||' in key_str:
            evidence_map[key_str] = value
        else:
            # Try parsing as tuple and convert to pipe format
            try:
                key_tuple = eval(key_str)
                if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                    evidence_map[f"{key_tuple[0]}|||{key_tuple[1]}"] = value
            except:
                pass
    
    # Build category map from criteria (extract from criterion names or use default categories)
    category_map = {}
    for crit in criteria:
        # After filtering, criteria is now a list of strings
        crit_lower = crit.lower()
        
        # Use crit as the key for category_map
        if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
            category_map[crit] = 'Technical Skills'
        elif any(word in crit_lower for word in ['experience', 'years', 'background']):
            category_map[crit] = 'Experience'
        elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
            category_map[crit] = 'Qualifications'
        elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
            category_map[crit] = 'Soft Skills'
        elif any(word in crit_lower for word in ['responsibility', 'duty', 'manage', 'lead']):
            category_map[crit] = 'Responsibilities'
        else:
            category_map[crit] = 'Other Requirements'
    
    # Convert coverage DataFrame to list of dicts for JSON serialization
    coverage_data = coverage_df.to_dict(orient='records')
    
    # Check if this analysis is for the current draft or a historical one
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    is_current_draft_analysis = False
    if draft and draft.created_at < analysis.created_at and (not draft.updated_at or draft.updated_at <= analysis.created_at):
        is_current_draft_analysis = True
    
    # Get pricing for unlock cost display
    from config import Config
    pricing = Config.get_pricing()
    
    # Get candidate file information for tooltips (file sizes, truncation warnings)
    from database import CandidateFile
    from analysis import load_gpt_settings
    gpt_settings = load_gpt_settings()
    resume_limit = gpt_settings.get('candidate_text_chars', 12000)
    
    candidate_files_info = {}
    candidate_files = CandidateFile.query.filter_by(analysis_id=analysis.id).all()
    for cf in candidate_files:
        text_length = len(cf.extracted_text or '')
        candidate_files_info[cf.candidate_name] = {
            'text_length': text_length,
            'was_truncated': text_length > resume_limit,
            'filename': cf.file_name
        }
    
    return render_template('results_enhanced.html',
                         analysis=analysis,
                         coverage_data=coverage_data,
                         category_map=category_map,
                         insights=insights,
                         criteria=criteria,
                         evidence_map=evidence_map,
                         user_settings=user_settings,
                         in_workflow=is_current_draft_analysis,
                         has_unsaved_work=False,
                         analysis_completed=True,
                         is_current_draft_analysis=is_current_draft_analysis,
                         pricing=pricing,
                         candidate_files_info=candidate_files_info,
                         resume_limit=resume_limit,
                         viewing_other_user=viewing_other_user,
                         analysis_owner_email=analysis_owner_email)

@analysis_bp.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    """Submit user feedback on analysis accuracy"""
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        vote = data.get('vote')  # 'up' or 'down'
        improvement_note = data.get('improvement_note', '')
        
        # Verify the analysis belongs to the user
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            return jsonify({'success': False, 'error': 'Analysis not found'}), 404
        
        # Check if feedback already exists for this analysis
        existing_feedback = Feedback.query.filter_by(
            analysis_id=analysis_id,
            user_id=current_user.id
        ).first()
        
        if existing_feedback:
            # Update existing feedback
            existing_feedback.vote = vote
            existing_feedback.improvement_note = improvement_note if improvement_note else None
            existing_feedback.created_at = datetime.utcnow()  # Update timestamp
        else:
            # Create new feedback
            feedback = Feedback(
                analysis_id=analysis_id,
                user_id=current_user.id,
                vote=vote,
                improvement_note=improvement_note if improvement_note else None
            )
            db.session.add(feedback)
        
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Thank you for your feedback!'})
        
    except Exception as e:
        db.session.rollback()
        print(f"Error submitting feedback: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@analysis_bp.route('/export/<int:analysis_id>/<format>')
@login_required
def export_analysis(analysis_id, format):
    """Export analysis results to PDF, Excel, or Word"""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    # Parse stored JSON data
    import pandas as pd
    from io import StringIO
    coverage = pd.read_json(StringIO(analysis.coverage_data), orient='records')
    insights = json.loads(analysis.insights_data)
    criteria = json.loads(analysis.criteria_list)
    
    try:
        if format == 'pdf':
            # Export as Executive Summary PDF
            from flask import send_file
            from export_utils import to_executive_summary_pdf
            from io import StringIO
            import pandas as pd
            
            # Parse data
            coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
            
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
                return redirect(url_for('analysis.results', analysis_id=analysis_id))
            
            # Return as download
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d')
            safe_title = re.sub(r'[^\w\s-]', '', analysis.job_title).strip().replace(' ', '_')[:50]
            return send_file(
                io.BytesIO(pdf_bytes),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'results_{safe_title}_{timestamp}.pdf'
            )
            
        elif format == 'excel':
            # Export as Excel coverage matrix
            from flask import send_file
            from export_utils import to_excel_coverage_matrix
            from io import StringIO
            import pandas as pd
            
            # Parse data
            coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
            
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
                return redirect(url_for('analysis.results', analysis_id=analysis_id))
            
            # Return as download
            safe_title = re.sub(r'[^\w\s-]', '', analysis.job_title).strip().replace(' ', '_')[:50]
            return send_file(
                io.BytesIO(excel_bytes),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'results_{safe_title}.xlsx'
            )
            
        elif format == 'word':
            # TODO: Implement Word export
            flash('Word export coming soon!', 'info')
            return redirect(url_for('analysis.results', analysis_id=analysis_id))
        else:
            flash('Invalid export format', 'error')
            return redirect(url_for('analysis.results', analysis_id=analysis_id))
    except Exception as e:
        flash(f'Export failed: {str(e)}', 'error')
        return redirect(url_for('analysis.results', analysis_id=analysis_id))

@analysis_bp.route('/review-criteria', methods=['GET', 'POST'])
@login_required
def review_criteria():
    """Review and edit criteria BEFORE running analysis"""
    if request.method == 'GET':
        # Check if we have criteria in database
        draft = Draft.query.filter_by(user_id=current_user.id).first()
        if not draft or not draft.criteria_data:
            flash('Please upload a JD first to extract criteria', 'info')
            return redirect(url_for('analysis.analyze'))
        
        criteria_data = json.loads(draft.criteria_data)
        
        # Check if CURRENT draft has been analyzed (draft created before latest analysis)
        latest_analysis = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).first()
        analysis_completed = False
        draft_modified_after_analysis = False
        current_draft_analysis_id = None
        
        if latest_analysis and draft.created_at < latest_analysis.created_at:
            analysis_completed = True
            current_draft_analysis_id = latest_analysis.id
            # Check if draft was updated after the analysis
            if draft.updated_at > latest_analysis.created_at:
                draft_modified_after_analysis = True
        
        # Load GPT settings to show user-configured text limit
        from analysis import load_gpt_settings
        gpt_settings = load_gpt_settings()
        jd_display_limit = gpt_settings.get('jd_text_chars', 15000)
        
        return render_template('review_criteria.html',
                             criteria_data=criteria_data,
                             jd_filename=draft.jd_filename,
                             job_title=draft.job_title or "Position Not Specified",
                             jd_text=draft.jd_text[:jd_display_limit],  # Respect admin setting
                             has_pdf=bool(draft.jd_bytes),  # Check if file bytes exist, not just filename
                             in_workflow=True, has_unsaved_work=True,
                             analysis_completed=analysis_completed,
                             draft_modified_after_analysis=draft_modified_after_analysis,
                             latest_analysis_id=current_draft_analysis_id)
    
    # POST - update criteria
    try:
        data = request.get_json()
        criteria_list = data.get('criteria', [])
        job_title = data.get('job_title', '')
        
        # Update database
        draft = Draft.query.filter_by(user_id=current_user.id).first()
        if draft:
            draft.criteria_data = json.dumps(criteria_list)
            if job_title:
                draft.job_title = job_title
            db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Saved {len(criteria_list)} criteria. Return to Upload & Analyse to add resumes and run analysis.',
            'redirect': url_for('analysis.analyze')
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@analysis_bp.route('/export-criteria')
@login_required
def export_criteria():
    """Export current criteria as CSV"""
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    if not draft or not draft.criteria_data:
        flash('No criteria to export', 'error')
        return redirect(url_for('analysis.review_criteria'))
    
    criteria_list = json.loads(draft.criteria_data)
    
    # Build CSV
    import csv
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Criterion', 'Category'])
    for item in criteria_list:
        if item.get('use', True):  # Only export checked criteria
            writer.writerow([item['criterion'], item.get('category', '')])
    
    csv_content = output.getvalue()
    
    from flask import Response
    return Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=criteria.csv'}
    )

@analysis_bp.route('/import-criteria', methods=['POST'])
@login_required
def import_criteria():
    """Import criteria from CSV file"""
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    if not draft:
        return json.dumps({'success': False, 'message': 'Please upload a JD first'})
    
    csv_file = request.files.get('criteria_file')
    if not csv_file:
        return json.dumps({'success': False, 'message': 'No file uploaded'})
    
    try:
        import csv
        from io import StringIO
        
        # Read CSV
        csv_content = csv_file.read().decode('utf-8')
        reader = csv.DictReader(StringIO(csv_content))
        
        criteria_list = []
        for row in reader:
            if row.get('Criterion'):
                criteria_list.append({
                    'criterion': row['Criterion'].strip(),
                    'category': row.get('Category', 'Other Requirements').strip() or 'Other Requirements',
                    'use': True
                })
        
        if not criteria_list:
            return json.dumps({'success': False, 'message': 'No valid criteria found in CSV'})
        
        # Update draft
        draft.criteria_data = json.dumps(criteria_list)
        db.session.commit()
        
        return json.dumps({
            'success': True,
            'message': f'Imported {len(criteria_list)} criteria successfully',
            'count': len(criteria_list)
        })
    except Exception as e:
        return json.dumps({'success': False, 'message': f'Import failed: {str(e)}'})

@analysis_bp.route('/view-jd-pdf')
@login_required
def view_jd_pdf():
    """View JD PDF file"""
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    if not draft or not draft.jd_bytes:
        # Return HTML error message instead of flash + redirect
        # This prevents the iframe from loading a redirect and showing the whole page
        return '''
            <html>
            <body style="font-family: sans-serif; padding: 40px; text-align: center; color: #6b7280;">
                <h3 style="color: #ef4444;">📄 No PDF File Available</h3>
                <p>The original JD file is not available (may have been pasted as text or loaded from history).</p>
                <p style="font-size: 14px; margin-top: 20px;">The full JD text is shown below the criteria list.</p>
            </body>
            </html>
        ''', 404
    
    from flask import Response
    return Response(
        draft.jd_bytes,
        mimetype='application/pdf',
        headers={'Content-Disposition': f'inline; filename={draft.jd_filename}'}
    )

@analysis_bp.route('/view-candidate-file/<int:analysis_id>/<candidate_name>')
@login_required
def view_candidate_file(analysis_id, candidate_name):
    """View candidate resume file"""
    from database import CandidateFile
    candidate_file = CandidateFile.query.filter_by(
        analysis_id=analysis_id,
        candidate_name=candidate_name
    ).first()
    
    if not candidate_file:
        flash('Candidate file not found', 'error')
        return redirect(url_for('analysis.insights', analysis_id=analysis_id))
    
    from flask import Response
    return Response(
        candidate_file.file_bytes,
        mimetype='application/pdf',
        headers={'Content-Disposition': f'inline; filename={candidate_file.file_name}'}
    )

@analysis_bp.route('/insights/<int:analysis_id>')
@login_required
def insights(analysis_id):
    """View detailed insights for candidates from an analysis"""
    import json
    
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
    
    # Check if analysis has insights data
    if not analysis.insights_data:
        flash('No insights available for this analysis', 'info')
        return redirect(url_for('analysis.results_enhanced', analysis_id=analysis_id))
    
    coverage_data = json.loads(analysis.coverage_data)
    insights_data = json.loads(analysis.insights_data)
    
    # Get candidate name from query parameter or default to first candidate
    selected_candidate = request.args.get('candidate')
    
    # Sort candidates by overall score
    candidates_list = []
    for idx, row in enumerate(coverage_data):
        candidate_name = row['Candidate']
        overall_score = row['Overall']
        candidates_list.append({
            'name': candidate_name,
            'score': overall_score,
            'data': row
        })
    
    # Sort by score descending and add ranks
    candidates_list.sort(key=lambda x: x['score'], reverse=True)
    for rank, cand in enumerate(candidates_list, 1):
        cand['rank'] = rank
    
    # Default to top candidate if none selected
    if not selected_candidate or selected_candidate not in [c['name'] for c in candidates_list]:
        selected_candidate = candidates_list[0]['name']
    
    # Find current candidate index
    current_idx = next(i for i, c in enumerate(candidates_list) if c['name'] == selected_candidate)
    current_candidate_data = candidates_list[current_idx]
    
    # Get thresholds
    hi_threshold = 0.70
    lo_threshold = 0.45
    
    # Calculate criteria breakdown
    criteria_cols = [k for k in current_candidate_data['data'].keys() if k not in ['Candidate', 'Overall']]
    strong_count = sum(1 for c in criteria_cols if current_candidate_data['data'][c] >= hi_threshold)
    moderate_count = sum(1 for c in criteria_cols if lo_threshold <= current_candidate_data['data'][c] < hi_threshold)
    weak_count = sum(1 for c in criteria_cols if current_candidate_data['data'][c] < lo_threshold)
    
    # Get insights for current candidate
    candidate_insights = insights_data.get(selected_candidate, {})
    
    # Check if insights exist (be flexible with the structure)
    has_gpt_insights = False
    if candidate_insights:
        # Check for any of the expected insight fields
        has_gpt_insights = bool(
            candidate_insights.get('top') or 
            candidate_insights.get('gaps') or 
            candidate_insights.get('notes') or
            candidate_insights.get('strengths') or  # Alternative field name
            candidate_insights.get('weaknesses') or  # Alternative field name
            candidate_insights.get('recommendation') or  # Alternative field name
            len(candidate_insights) > 0  # Any insights data at all
        )
    
    # Get evidence snippets first (handle both tuple and pipe formats)
    evidence_data_raw = json.loads(analysis.evidence_data) if analysis.evidence_data else {}
    evidence_data = {}
    for key_str, value in evidence_data_raw.items():
        if '|||' in key_str:
            evidence_data[key_str] = value
        else:
            try:
                key_tuple = eval(key_str)
                if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                    evidence_data[f"{key_tuple[0]}|||{key_tuple[1]}"] = value
            except:
                pass
    
    # Group scores by category
    cat_map = json.loads(analysis.category_map) if analysis.category_map else {}
    scores_by_category = {}
    for criterion in criteria_cols:
        category = cat_map.get(criterion, 'Other Requirements')
        score = current_candidate_data['data'][criterion]
        
        # Get evidence from 4-tuple structure
        key = f"{selected_candidate}|||{criterion}"
        evidence_tuple = evidence_data.get(key, ("", "", 0, 0))
        raw_evidence = evidence_tuple[0] if isinstance(evidence_tuple, (list, tuple)) and len(evidence_tuple) > 0 else ""
        justification = evidence_tuple[1] if isinstance(evidence_tuple, (list, tuple)) and len(evidence_tuple) > 1 else ""
        
        # Determine color and rating
        if score >= hi_threshold:
            color = '#28a745'
            rating = 'Strong'
        elif score >= lo_threshold:
            color = '#ffc107'
            rating = 'Moderate'
        else:
            color = '#dc3545'
            rating = 'Weak'
        
        if category not in scores_by_category:
            scores_by_category[category] = []
        
        scores_by_category[category].append({
            'criterion': criterion,
            'score': score,
            'color': color,
            'rating': rating,
            'raw_evidence': raw_evidence,
            'justification': justification
        })
    
    # Build evidence list for backward compatibility
    evidence_data_raw = json.loads(analysis.evidence_data) if analysis.evidence_data else {}
    evidence_data = {}
    for key_str, value in evidence_data_raw.items():
        if '|||' in key_str:
            evidence_data[key_str] = value
        else:
            try:
                key_tuple = eval(key_str)
                if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                    evidence_data[f"{key_tuple[0]}|||{key_tuple[1]}"] = value
            except:
                pass
    
    evidence_list = []
    for criterion in criteria_cols:
        key = f"{selected_candidate}|||{criterion}"
        if key in evidence_data:
            evidence_tuple = evidence_data[key]
            # evidence_tuple structure: [0]=raw_evidence, [1]=justification, [2]=score, [3]=density
            raw_evidence = evidence_tuple[0] if isinstance(evidence_tuple, (list, tuple)) and len(evidence_tuple) > 0 else ""
            justification = evidence_tuple[1] if isinstance(evidence_tuple, (list, tuple)) and len(evidence_tuple) > 1 else ""
            
            score = current_candidate_data['data'][criterion]
            
            if score >= hi_threshold:
                color = '#28a745'
            elif score >= lo_threshold:
                color = '#ffc107'
            else:
                color = '#dc3545'
            
            evidence_list.append({
                'criterion': criterion,
                'raw_evidence': raw_evidence,
                'justification': justification,
                'score': score,
                'color': color
            })
    
    # Sort evidence by score descending
    evidence_list.sort(key=lambda x: x['score'], reverse=True)
    
    # Get candidate file data
    from database import CandidateFile
    candidate_file = CandidateFile.query.filter_by(
        analysis_id=analysis_id,
        candidate_name=selected_candidate
    ).first()
    
    current_candidate = {
        'name': selected_candidate,
        'score': current_candidate_data['score'],
        'rank': current_candidate_data['rank'],
        'strong_count': strong_count,
        'moderate_count': moderate_count,
        'weak_count': weak_count,
        'has_gpt_insights': has_gpt_insights,
        'insights': candidate_insights,
        'scores_by_category': scores_by_category,
        'evidence': evidence_list,
        'file': candidate_file
    }
    
    # Check if this analysis is for the current draft or a historical one
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    is_current_draft_analysis = False
    if draft and draft.created_at < analysis.created_at and (not draft.updated_at or draft.updated_at <= analysis.created_at):
        is_current_draft_analysis = True
    
    # Check if user has Full Radar (all candidates have insights)
    # If gpt_candidates is not set or contains all candidates, assume Full Radar was used
    gpt_candidates_list = []
    if analysis.gpt_candidates:
        try:
            gpt_candidates_list = json.loads(analysis.gpt_candidates)
        except:
            gpt_candidates_list = []
    
    # Full Radar if all candidates have insights or no restriction was set
    has_full_radar = (len(gpt_candidates_list) == 0 or len(gpt_candidates_list) >= analysis.num_candidates)
    
    # Check if admin is viewing another user's analysis
    viewing_other_user = is_admin and analysis.user_id != current_user.id
    analysis_owner = User.query.get(analysis.user_id) if viewing_other_user else None
    analysis_owner_email = analysis_owner.email if analysis_owner else None
    
    from config import Config
    pricing = Config.get_pricing()
    return render_template('insights.html',
                         analysis=analysis,
                         analysis_id=analysis_id,
                         candidates=candidates_list,
                         current_candidate=current_candidate,
                         current_idx=current_idx,
                         has_insights=True,
                         in_workflow=is_current_draft_analysis,
                         has_unsaved_work=False,
                         analysis_completed=True,
                         is_current_draft_analysis=is_current_draft_analysis,
                         has_full_radar=has_full_radar,
                         pricing=pricing,
                         viewing_other_user=viewing_other_user,
                         analysis_owner_email=analysis_owner_email)

@analysis_bp.route('/unlock-candidate/<int:analysis_id>/<candidate_name>', methods=['POST'])
@login_required
def unlock_candidate(analysis_id, candidate_name):
    """Unlock insights for a single candidate by generating AI insights and deducting the configured price"""
    from decimal import Decimal
    from config import Config
    
    try:
        # Get dynamic pricing
        pricing = Config.get_pricing()
        unlock_price = Decimal(str(pricing['EXTRA_INSIGHT_PRICE']))
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            return jsonify({'success': False, 'error': 'Analysis not found'}), 404
        
        # Check balance (compare Decimal with Decimal)
        if current_user.balance_usd < unlock_price:
            return jsonify({
                'success': False, 
                'error': 'Insufficient balance',
                'balance': float(current_user.balance_usd),
                'redirect': url_for('payments.buy_credits')
            }), 402
        
        # Parse existing insights
        insights_data = json.loads(analysis.insights_data) if analysis.insights_data else {}
        
        # Check if already unlocked
        if candidate_name in insights_data and insights_data[candidate_name]:
            return jsonify({
                'success': False,
                'error': 'Candidate insights already exist',
                'insights': insights_data[candidate_name]
            })
        
        # Get candidate data
        from database import CandidateFile
        candidate_file = CandidateFile.query.filter_by(
            analysis_id=analysis_id,
            candidate_name=candidate_name
        ).first()
        
        if not candidate_file:
            return jsonify({'success': False, 'error': 'Candidate file not found'}), 404
        
        # Get coverage data for scores
        coverage_data = json.loads(analysis.coverage_data)
        candidate_row = next((row for row in coverage_data if row['Candidate'] == candidate_name), None)
        
        if not candidate_row:
            return jsonify({'success': False, 'error': 'Candidate scores not found'}), 404
        
        # Prepare data for insights generation
        criteria = json.loads(analysis.criteria_list)
        evidence_map_raw = json.loads(analysis.evidence_data)
        
        # Convert evidence_map keys from strings to tuples
        evidence_map = {}
        for key_str, value in evidence_map_raw.items():
            try:
                # Try eval first for tuple format: "('Candidate Name', 'Criterion')"
                key_tuple = eval(key_str)
                evidence_map[key_tuple] = value
            except:
                # Try splitting by ||| for alternate format: "Candidate Name|||Criterion"
                if '|||' in key_str:
                    parts = key_str.split('|||')
                    if len(parts) == 2:
                        key_tuple = (parts[0], parts[1])
                        # Only add if this is the current candidate
                        if parts[0] == candidate_name:
                            evidence_map[key_tuple] = value
        
        # Get scores for this candidate
        candidate_scores = {col: candidate_row[col] for col in candidate_row.keys() 
                          if col not in ['Candidate', 'Overall']}
        
        # Generate insights using GPT
        from analysis import gpt_candidate_insights
        insights = gpt_candidate_insights(
            candidate_name=candidate_name,
            candidate_text=candidate_file.extracted_text,
            jd_text=analysis.jd_full_text,
            coverage_scores=candidate_scores,
            criteria=criteria,
            evidence_map=evidence_map,
            model="gpt-4o"
        )
        
        # Store insights
        insights_data[candidate_name] = insights
        analysis.insights_data = json.dumps(insights_data)
        
        # Update gpt_candidates list
        gpt_candidates_list = []
        if analysis.gpt_candidates:
            try:
                gpt_candidates_list = json.loads(analysis.gpt_candidates)
            except:
                pass
        if candidate_name not in gpt_candidates_list:
            gpt_candidates_list.append(candidate_name)
        analysis.gpt_candidates = json.dumps(gpt_candidates_list)
        
        # Deduct from balance using proper transaction recording
        current_user.deduct_funds(
            amount_usd=unlock_price,
            description=f'Unlock Insights - {candidate_name} (Job: {analysis.job_title or "Analysis #" + str(analysis_id)})'
        )
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'insights': insights,
            'new_balance': float(current_user.balance_usd),
            'message': f'Insights unlocked for {candidate_name}'
        })
        
    except Exception as e:
        db.session.rollback()
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@analysis_bp.route('/download-candidate-pdf/<int:analysis_id>/<candidate_name>')
@login_required
def download_candidate_pdf(analysis_id, candidate_name):
    """Download individual candidate report as PDF"""
    from flask import send_file
    from export_candidate import to_individual_candidate_pdf
    
    # Check if include_justifications parameter is passed
    include_justifications = request.args.get('include_justifications', 'false').lower() == 'true'
    
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
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
    
    # Get candidate data
    coverage_row = coverage_df[coverage_df['Candidate'] == candidate_name].iloc[0]
    cand_insights = insights.get(candidate_name, {})
    
    # Generate PDF
    pdf_bytes = to_individual_candidate_pdf(
        candidate_name=candidate_name,
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
    
    if pdf_bytes is None:
        flash('PDF generation not available (ReportLab not installed)', 'error')
        return redirect(url_for('analysis.insights', analysis_id=analysis_id, candidate=candidate_name))
    
    # Return as download with sanitized filename
    safe_name = re.sub(r'[^\w\s-]', '', candidate_name).strip().replace(' ', '_')
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'{safe_name}_report.pdf'
    )

@analysis_bp.route('/criteria/<int:analysis_id>')
@login_required
def view_criteria(analysis_id):
    """View and edit criteria for an analysis"""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard.dashboard'))
    
    criteria = json.loads(analysis.criteria_list)
    
    # Build category map (same logic as results page)
    criteria_with_categories = []
    for crit in criteria:
        # Handle multiple formats: string, dict with 'criterion', or dict with 'text'
        if isinstance(crit, dict):
            crit_text = crit.get('criterion') or crit.get('text') or str(crit)
        else:
            crit_text = crit
        
        crit_lower = crit_text.lower()
        if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
            category = 'Technical Skills'
        elif any(word in crit_lower for word in ['experience', 'years', 'background']):
            category = 'Experience'
        elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
            category = 'Qualifications'
        elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
            category = 'Soft Skills'
        elif any(word in crit_lower for word in ['responsibility', 'duty', 'manage', 'lead']):
            category = 'Responsibilities'
        else:
            category = 'Other Requirements'
        
        criteria_with_categories.append({
            'criterion': crit_text,
            'category': category
        })
    
    return render_template('criteria.html',
                         analysis_id=analysis_id,
                         job_title=analysis.job_title,
                         criteria_with_categories=criteria_with_categories)

@analysis_bp.route('/criteria/<int:analysis_id>/update', methods=['POST'])
@login_required
def update_criteria(analysis_id):
    """Update criteria and trigger re-analysis"""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    if not analysis:
        return json.dumps({'success': False, 'message': 'Analysis not found'})
    
    try:
        data = request.get_json()
        criteria_list = [item['criterion'] for item in data['criteria']]
        
        # Update criteria in database
        analysis.criteria_list = json.dumps(criteria_list)
        analysis.num_criteria = len(criteria_list)
        db.session.commit()
        
        # Redirect to analyze page with instruction to re-run
        # For now, just redirect back to results with a message
        return json.dumps({
            'success': True,
            'message': 'Criteria updated. Please re-run analysis from the Analyze page.',
            'redirect': url_for('analysis.results', analysis_id=analysis_id)
        })
    except Exception as e:
        return json.dumps({'success': False, 'message': str(e)})

