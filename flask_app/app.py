from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify, current_app
from flask_login import LoginManager, login_required, current_user
from config import config
from database import db, init_db, User, Transaction, Analysis, Draft, DraftResume, CandidateFile, UserSettings, Feedback, AdminLoginAttempt, AdminAuditLog
import os
import re
from datetime import datetime, timezone, timedelta
import json
import hashlib
from werkzeug.utils import secure_filename
from decimal import Decimal
import io
from dotenv import load_dotenv
import pyotp
import qrcode
from io import BytesIO
import base64

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
def create_app(config_name=None):
    """Application factory"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    init_db(app)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        user = User.query.get(int(user_id))
        # Block suspended users from logging in
        if user and user.is_suspended:
            return None
        # Update last_seen timestamp for online status tracking (naive UTC)
        if user:
            user.last_seen = datetime.utcnow()
            db.session.commit()
        return user
    
    # Add template filters
    @app.template_filter('to_local_time')
    def to_local_time(dt):
        """Convert UTC datetime to ISO format for JavaScript conversion"""
        if dt:
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        return ''
    
    @app.template_filter('to_aest')
    def to_aest(dt):
        """Convert UTC datetime to AEST (UTC+10)"""
        if dt:
            from datetime import timedelta
            return dt + timedelta(hours=10)
        return dt
    
    # Make datetime and timedelta available in templates
    @app.context_processor
    def inject_datetime():
        return {
            'now': datetime.utcnow,
            'timedelta': timedelta
        }
    
    @app.template_filter('number_format')
    def number_format(value):
        """Format number with thousands separator"""
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return value
    
    # Register blueprints
    from auth import auth_bp
    from payments import payments
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(payments)
    
    # Main routes
    @app.route('/')
    def landing():
        """Landing page - shows marketing page for non-logged-in users, redirects to dashboard for logged-in"""
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        # Load pricing for display
        pricing_path = os.path.join(os.path.dirname(__file__), 'config', 'pricing_settings.json')
        with open(pricing_path, 'r', encoding='utf-8') as f:
            pricing_config = json.load(f)
        
        return render_template('landing.html', pricing=pricing_config)
    
    @app.route('/pricing')
    def pricing():
        """Pricing page"""
        pricing_path = os.path.join(os.path.dirname(__file__), 'config', 'pricing_settings.json')
        with open(pricing_path, 'r', encoding='utf-8') as f:
            pricing_config = json.load(f)
        test_drive_price = pricing_config['standard_tier_price']['value'] - pricing_config['new_user_welcome_credit']['value']
        return render_template('pricing.html', 
                             pricing=pricing_config,
                             test_drive_price=test_drive_price)
    
    @app.route('/security')
    def security():
        """Security and data privacy page"""
        return render_template('security.html')
    
    @app.route('/favicon.ico')
    def favicon():
        """Return 204 No Content for favicon requests to avoid 404s"""
        return '', 204
    
    @app.route('/features')
    def features():
        """Features deep dive page"""
        return render_template('features.html')
    
    @app.route('/privacy')
    def privacy():
        """Privacy Policy page"""
        return render_template('privacy.html')
    
    @app.route('/terms')
    def terms():
        """Terms of Service page"""
        return render_template('terms.html')
    
    @app.route('/support', methods=['GET', 'POST'])
    def support():
        """Contact Support page with form submission"""
        if request.method == 'POST':
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            subject = request.form.get('subject', '').strip()
            message = request.form.get('message', '').strip()
            
            if not all([name, email, subject, message]):
                flash('Please fill in all fields.', 'error')
                return render_template('support.html')
            
            # Send support email
            try:
                from email_utils import send_support_email
                send_support_email(
                    user_name=name,
                    user_email=email,
                    subject=subject,
                    message=message
                )
                flash('Your message has been sent! We\'ll get back to you within 24 hours.', 'success')
                return redirect(url_for('support'))
            except Exception as e:
                print(f"ERROR: Failed to send support email: {e}")
                flash('There was an error sending your message. Please try emailing us directly at contact@candidateevaluator.com', 'error')
                return render_template('support.html')
        
        return render_template('support.html')
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        """Dashboard - logged-in users only"""
        # Get recent analyses for the user
        recent_analyses = Analysis.query.filter_by(
            user_id=current_user.id
        ).order_by(
            Analysis.created_at.desc()
        ).limit(5).all()
        
        # Get current draft if exists
        draft = Draft.query.filter_by(user_id=current_user.id).first()
        
        # Load pricing settings for welcome credit display
        pricing_path = os.path.join(os.path.dirname(__file__), 'config', 'pricing_settings.json')
        with open(pricing_path, 'r') as f:
            pricing_config = json.load(f)
        
        return render_template('dashboard.html', user=current_user, recent_analyses=recent_analyses, draft=draft, pricing=pricing_config)
    
    @app.route('/api/get-balance')
    @login_required
    def get_balance():
        """API endpoint to get current user balance (for updating after Stripe payment)"""
        return jsonify({
            'balance': float(current_user.balance_usd),
            'user_id': current_user.id
        })
    
    def load_system_settings():
        """Load system settings from JSON file"""
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'system_settings.json')
        try:
            with open(settings_path, 'r') as f:
                settings_data = json.load(f)
            # Return simple dict of setting_name -> value
            result = {key: val['value'] for key, val in settings_data.items() if key != '_metadata'}
            return result
        except Exception as e:
            print(f"ERROR loading system settings: {e}")
            import traceback
            traceback.print_exc()
            # Return defaults
            return {
                'enable_document_length_warnings': True,
                'registration_enabled': False,
                'maintenance_mode': False,
                'max_file_size_mb': 10,
                'new_user_welcome_credit': 0.0
            }
    
    @app.route('/jd-length-warning')
    @login_required
    def jd_length_warning_route():
        """Show JD length warning page"""
        jd_length = request.args.get('jd_length', type=int)
        jd_limit = request.args.get('jd_limit', type=int)
        
        if not session.get('show_jd_length_warning'):
            flash('Invalid access to warning page', 'error')
            return redirect(url_for('analyze'))
        
        return render_template('jd_length_warning.html',
                             jd_length=jd_length,
                             jd_limit=jd_limit)
    
    @app.route('/resume-length-warning')
    @login_required
    def resume_length_warning_route():
        """Show resume length warning page"""
        truncated_count = request.args.get('truncated_count', type=int)
        resume_limit = request.args.get('resume_limit', type=int)
        
        if not session.get('show_resume_length_warning'):
            flash('Invalid access to warning page', 'error')
            return redirect(url_for('analyze', step='resumes'))
        
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

    @app.route('/document-warnings')
    @login_required
    def document_warnings_route():
        """Show combined document warnings page (too short AND/OR too long)"""
        if not session.get('show_document_warnings'):
            flash('Invalid access to warning page', 'error')
            return redirect(url_for('analyze'))
        
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

    @app.route('/analyze', methods=['GET', 'POST'])

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
                    return redirect(url_for('analyze'))
                
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
                                return redirect(url_for('document_warnings_route'))
                            elif jd_too_long and not jd_too_short:
                                # Only too long - use existing single-issue page
                                session['show_jd_length_warning'] = True
                                session.modified = True
                                return redirect(url_for('jd_length_warning_route', jd_length=jd_length, jd_limit=jd_limit))
                            else:
                                # Both issues - use combined page
                                session['show_document_warnings'] = True
                                session.modified = True
                                return redirect(url_for('document_warnings_route'))
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
                    return redirect(url_for('analyze'))
                
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
                return redirect(url_for('review_criteria'))
                
            except Exception as e:
                flash(f'JD processing failed: {str(e)}', 'error')
                import traceback
                traceback.print_exc()
                return redirect(url_for('analyze'))
        
        elif action == 'confirm_jd_length':
            # User confirmed to proceed with long JD
            try:
                from analysis import read_file_bytes, hash_bytes, extract_jd_sections_with_gpt, build_criteria_from_sections
                
                # Retrieve JD data from draft (stored during warning display)
                draft = Draft.query.filter_by(user_id=current_user.id).first()
                if not draft or not draft.jd_text:
                    flash('Session expired. Please upload your JD again.', 'error')
                    return redirect(url_for('analyze'))
                
                # Clear session flag
                session.pop('show_jd_length_warning', None)
                
                jd_text_content = draft.jd_text
                jd_hash = draft.jd_hash
                
                # Extract criteria
                sections = extract_jd_sections_with_gpt(jd_text_content)
                criteria, cat_map = build_criteria_from_sections(sections, per_section=999, cap_total=10000)
                
                if not criteria:
                    flash('Could not extract criteria from job description', 'error')
                    return redirect(url_for('analyze'))
                
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
                return redirect(url_for('review_criteria'))
                
            except Exception as e:
                flash(f'JD processing failed: {str(e)}', 'error')
                import traceback
                traceback.print_exc()
                return redirect(url_for('analyze'))
        
        elif action == 'upload_resumes':
            # Step 3: Upload resumes and store in draft (don't run analysis yet)
            try:
                from analysis import read_file_bytes, hash_bytes, infer_candidate_name
                
                # Check we have a draft with criteria
                draft = Draft.query.filter_by(user_id=current_user.id).first()
                if not draft or not draft.criteria_data:
                    flash('Please upload a JD and review criteria first', 'error')
                    return redirect(url_for('analyze'))
                
                # Get resume files
                resume_files = request.files.getlist('resumes')
                if not resume_files or not resume_files[0].filename:
                    flash('Please select at least one resume file', 'error')
                    return redirect(url_for('analyze', step='resumes'))
                
                # Process and store resumes temporarily
                resumes_added = 0
                processed_resumes = []  # Store for length checking
                
                for resume_file in resume_files:
                    if not resume_file.filename:
                        continue
                    
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
                            return redirect(url_for('document_warnings_route'))
                        elif short_resumes and not long_resumes:
                            # Only too short - use combined page
                            session['show_document_warnings'] = True
                            session.modified = True
                            return redirect(url_for('document_warnings_route'))
                        elif long_resumes and not short_resumes:
                            # Only too long - use existing single-issue page
                            session['show_resume_length_warning'] = True
                            session.modified = True
                            return redirect(url_for('resume_length_warning_route',
                                                  truncated_count=len(long_resumes),
                                                  resume_limit=resume_limit))
                        else:
                            # Both short and long - use combined page
                            session['show_document_warnings'] = True
                            session.modified = True
                            return redirect(url_for('document_warnings_route'))
                
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
                
                if resumes_added == 0:
                    flash('No new resumes added (duplicates skipped)', 'info')
                else:
                    flash(f'✅ {resumes_added} resume(s) uploaded successfully!', 'success')
                
                # Redirect to Run Analysis page
                return redirect(url_for('run_analysis_route'))
                
            except Exception as e:
                db.session.rollback()
                flash(f'Resume upload failed: {str(e)}', 'error')
                import traceback
                traceback.print_exc()
                return redirect(url_for('analyze', step='resumes'))
        
        elif action == 'confirm_resume_length':
            # User confirmed to proceed with long resumes (already saved in DB)
            try:
                # Check we have the session flag
                if not session.get('show_resume_length_warning'):
                    flash('Session expired. Please upload your resumes again.', 'error')
                    return redirect(url_for('analyze', step='resumes'))
                
                # Get draft
                draft = Draft.query.filter_by(user_id=current_user.id).first()
                if not draft:
                    flash('Draft not found. Please start over.', 'error')
                    return redirect(url_for('analyze'))
                
                # Resumes already in database, clear session flags
                resumes_added = session.get('resumes_added', 0)
                session.pop('show_resume_length_warning', None)
                session.pop('resumes_added', None)
                
                # Mark that user has already confirmed truncation for this draft
                # So run_analysis doesn't show the warning again
                session['truncation_confirmed'] = True
                session.modified = True
                
                flash(f'✅ {resumes_added} resume(s) added to analysis!', 'success')
                return redirect(url_for('run_analysis_route'))
                
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
                import traceback
                traceback.print_exc()
                return redirect(url_for('analyze', step='resumes'))
        
        elif action == 'confirm_document_warnings':
            # User confirmed to proceed despite document warnings (too short and/or too long)
            try:
                # Check we have the session flag
                if not session.get('show_document_warnings'):
                    flash('Session expired. Please start over.', 'error')
                    return redirect(url_for('analyze'))
                
                # Get draft
                draft = Draft.query.filter_by(user_id=current_user.id).first()
                if not draft:
                    flash('Draft not found. Please start over.', 'error')
                    return redirect(url_for('analyze'))
                
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
                            return redirect(url_for('analyze'))
                        
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
                        return redirect(url_for('review_criteria'))
                    else:
                        # Criteria already extracted, go to next step
                        if has_resumes:
                            # Have everything, go to run analysis
                            resumes_added = session.pop('resumes_added', 0)
                            if resumes_added:
                                flash(f'✅ {resumes_added} resume(s) added to analysis!', 'success')
                            return redirect(url_for('run_analysis_route'))
                        else:
                            # Need resumes
                            return redirect(url_for('analyze', step='resumes'))
                else:
                    # Only had resume warnings
                    resumes_added = session.pop('resumes_added', 0)
                    if resumes_added:
                        flash(f'✅ {resumes_added} resume(s) added to analysis!', 'success')
                    return redirect(url_for('run_analysis_route'))
                    
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
                import traceback
                traceback.print_exc()
                return redirect(url_for('analyze'))
                
                flash(f'✅ {resumes_added} resume(s) uploaded successfully!', 'success')
                return redirect(url_for('run_analysis_route'))
                
            except Exception as e:
                db.session.rollback()
                flash(f'Resume confirmation failed: {str(e)}', 'error')
                import traceback
                traceback.print_exc()
                return redirect(url_for('analyze', step='resumes'))
                return redirect(url_for('analyze', step='resumes'))
        
        elif action == 'run_analysis':
            # Legacy handler - redirect to proper flow
            flash('Please upload resumes first', 'info')
            return redirect(url_for('analyze', step='resumes'))
        
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
                    return redirect(url_for('analyze'))
                
                # Get resume files
                resume_files = request.files.getlist('resumes')
                if not resume_files:
                    flash('Please upload at least one resume', 'error')
                    return redirect(url_for('analyze'))
                
                # Get options
                insights_mode = request.form.get('insights_mode', 'top3')
                
                # Read JD from draft
                jd_text = draft.jd_text
                
                # Get active criteria from draft
                criteria_list = json.loads(draft.criteria_data)
                criteria = [c['criterion'] for c in criteria_list if c.get('use', True)]
                
                if not criteria:
                    flash('No criteria selected. Please review criteria first.', 'error')
                    return redirect(url_for('review_criteria'))
                
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
                    return redirect(url_for('analyze'))
                
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
                return redirect(url_for('results', analysis_id=analysis.id))
                
            except Exception as e:
                db.session.rollback()
                flash(f'Analysis failed: {str(e)}', 'error')
                import traceback
                traceback.print_exc()
                return redirect(url_for('analyze'))
        
        else:
            flash('Invalid action', 'error')
            return redirect(url_for('analyze'))
    
    @app.route('/clear-session', methods=['POST'])
    @login_required
    def clear_session():
        """Clear JD and criteria from database"""
        draft = Draft.query.filter_by(user_id=current_user.id).first()
        if draft:
            db.session.delete(draft)
            db.session.commit()
        flash('Draft cleared. You can upload a new JD.', 'info')
        return redirect(url_for('analyze'))
    
    @app.route('/delete-resume/<int:resume_id>', methods=['POST'])
    @login_required
    def delete_resume(resume_id):
        """Delete a specific resume from draft"""
        resume = DraftResume.query.get_or_404(resume_id)
        
        # Verify ownership through draft
        draft = Draft.query.filter_by(id=resume.draft_id, user_id=current_user.id).first()
        if not draft:
            flash('Resume not found', 'error')
            return redirect(url_for('analyze', step='resumes'))
        
        db.session.delete(resume)
        db.session.commit()
        flash(f'Removed resume: {resume.file_name}', 'success')
        return redirect(url_for('analyze', step='resumes'))
    
    @app.route('/load-analysis-to-draft/<int:analysis_id>', methods=['POST'])
    @login_required
    def load_analysis_to_draft(analysis_id):
        """Load a historical analysis into draft for editing and re-running"""
        # Get the analysis
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('job_history'))
        
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
        return redirect(url_for('review_criteria'))
    
    @app.route('/run-analysis', methods=['GET', 'POST'])
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
                return redirect(url_for('analyze'))
            
            # Check if returning from payment with auto_submit flag
            auto_submit = request.args.get('auto_submit') == '1'
            saved_insights_mode = session.pop('pending_insights_mode', None)
            
            # Get uploaded resumes count
            resume_count = DraftResume.query.filter_by(draft_id=draft.id).count()
            if resume_count == 0:
                flash('Please upload at least one resume', 'error')
                return redirect(url_for('analyze', step='resumes'))
            
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
                    return redirect(url_for('results', analysis_id=recent_analysis.id))
                else:
                    return redirect(url_for('dashboard'))
            
            # NOTE: Token will be consumed AFTER document length check
            # (so it can be restored if we need to show truncation warning)
            
            # Get configuration
            insights_mode = request.form.get('insights_mode', 'top3')
            
            # Get draft and validate
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if not draft or not draft.criteria_data:
                flash('Please upload a JD and review criteria first', 'error')
                return redirect(url_for('analyze'))
            
            # Get resumes from DraftResume table
            draft_resumes = DraftResume.query.filter_by(draft_id=draft.id).all()
            if not draft_resumes:
                flash('No resumes found. Please upload resumes first.', 'error')
                return redirect(url_for('analyze', step='resumes'))
            
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
                return redirect(url_for('review_criteria'))
            
            # CRITICAL: Calculate cost and check balance BEFORE running analysis
            num_candidates = len(candidates)
            
            # Initialize cost variable
            estimated_cost = Decimal('10.00')  # Default to standard tier
            
            # NEW PRICING MODEL: Calculate cost based on tier
            if insights_mode == 'standard':
                # Standard: $10 base (includes Top 5 insights)
                estimated_cost = Decimal('10.00')
                num_insights = min(5, num_candidates)
            elif insights_mode == 'deep_dive':
                # Deep Dive: $10 base + $10 extra (includes Top 15 insights)
                estimated_cost = Decimal('20.00')
                num_insights = min(15, num_candidates)
            elif insights_mode == 'full_radar':
                # Full Radar: $10 base + $1 per candidate beyond 5
                extra_candidates = max(0, num_candidates - 5)
                estimated_cost = Decimal('10.00') + (Decimal('1.00') * extra_candidates)
                num_insights = num_candidates
            # Legacy support for old values (map to new tiers)
            elif insights_mode in ['top3', 'top5']:
                estimated_cost = Decimal('10.00')
                num_insights = min(5, num_candidates)
            elif insights_mode == 'top10':
                estimated_cost = Decimal('20.00')
                num_insights = min(15, num_candidates)
            elif insights_mode == 'all':
                extra_candidates = max(0, num_candidates - 5)
                estimated_cost = Decimal('10.00') + (Decimal('1.00') * extra_candidates)
                num_insights = num_candidates
            else:
                estimated_cost = Decimal('10.00')
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
            return jsonify({'redirect': url_for('results', analysis_id=analysis.id)})
            
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
            
            return redirect(url_for('run_analysis_route'))
    
    @app.route('/results/<int:analysis_id>')
    @login_required
    def results(analysis_id):
        """Display analysis results with enhanced coverage matrix"""
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
        # Get user settings for threshold preferences
        user_settings = UserSettings.get_or_create(current_user.id)
        
        # Parse stored JSON data
        import pandas as pd
        from io import StringIO
        coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
        insights = json.loads(analysis.insights_data)
        criteria = json.loads(analysis.criteria_list)
        
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
            # Simple categorization based on keywords
            crit_lower = crit.lower()
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
        candidate_files_info = {}
        candidate_files = CandidateFile.query.filter_by(analysis_id=analysis.id).all()
        resume_limit = 12000  # From gpt_settings
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
                             candidate_files_info=candidate_files_info)
    
    @app.route('/submit_feedback', methods=['POST'])
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
    
    @app.route('/export/<int:analysis_id>/<format>')
    @login_required
    def export_analysis(analysis_id, format):
        """Export analysis results to PDF, Excel, or Word"""
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
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
                    crit_lower = crit.lower()
                    if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                        category_map[crit] = 'Technical Skills'
                    elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                        category_map[crit] = 'Experience'
                    elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                        category_map[crit] = 'Qualifications'
                    elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                        category_map[crit] = 'Soft Skills'
                    else:
                        category_map[crit] = 'Other Requirements'
                
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
                    return redirect(url_for('results', analysis_id=analysis_id))
                
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
                    crit_lower = crit.lower()
                    if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                        category_map[crit] = 'Technical Skills'
                    elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                        category_map[crit] = 'Experience'
                    elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                        category_map[crit] = 'Qualifications'
                    elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                        category_map[crit] = 'Soft Skills'
                    else:
                        category_map[crit] = 'Other Requirements'
                
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
                    return redirect(url_for('results', analysis_id=analysis_id))
                
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
                return redirect(url_for('results', analysis_id=analysis_id))
            else:
                flash('Invalid export format', 'error')
                return redirect(url_for('results', analysis_id=analysis_id))
        except Exception as e:
            flash(f'Export failed: {str(e)}', 'error')
            return redirect(url_for('results', analysis_id=analysis_id))
    
    @app.route('/review-criteria', methods=['GET', 'POST'])
    @login_required
    def review_criteria():
        """Review and edit criteria BEFORE running analysis"""
        if request.method == 'GET':
            # Check if we have criteria in database
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if not draft or not draft.criteria_data:
                flash('Please upload a JD first to extract criteria', 'info')
                return redirect(url_for('analyze'))
            
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
            
            return render_template('review_criteria.html',
                                 criteria_data=criteria_data,
                                 jd_filename=draft.jd_filename,
                                 job_title=draft.job_title or "Position Not Specified",
                                 jd_text=draft.jd_text[:2000],
                                 has_pdf=bool(draft.jd_bytes),  # Check if file bytes exist, not just filename
                                 in_workflow=True, has_unsaved_work=True,
                                 analysis_completed=analysis_completed,
                                 draft_modified_after_analysis=draft_modified_after_analysis,
                                 latest_analysis_id=current_draft_analysis_id)  # Preview first 2000 chars
        
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
            
            return json.dumps({
                'success': True,
                'message': f'Saved {len(criteria_list)} criteria. Return to Upload & Analyse to add resumes and run analysis.',
                'redirect': url_for('analyze')
            })
        except Exception as e:
            return json.dumps({'success': False, 'message': str(e)})
    
    @app.route('/export-criteria')
    @login_required
    def export_criteria():
        """Export current criteria as CSV"""
        draft = Draft.query.filter_by(user_id=current_user.id).first()
        if not draft or not draft.criteria_data:
            flash('No criteria to export', 'error')
            return redirect(url_for('review_criteria'))
        
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
    
    @app.route('/import-criteria', methods=['POST'])
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
    
    @app.route('/view-jd-pdf')
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
    
    @app.route('/view-candidate-file/<int:analysis_id>/<candidate_name>')
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
            return redirect(url_for('insights', analysis_id=analysis_id))
        
        from flask import Response
        return Response(
            candidate_file.file_bytes,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'inline; filename={candidate_file.file_name}'}
        )
    
    @app.route('/insights/<int:analysis_id>')
    @login_required
    def insights(analysis_id):
        """View detailed insights for candidates from an analysis"""
        import json
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
        # Check if analysis has insights data
        if not analysis.insights_data:
            flash('No insights available for this analysis', 'info')
            return redirect(url_for('results_enhanced', analysis_id=analysis_id))
        
        coverage_data = json.loads(analysis.coverage_data)
        insights_data = json.loads(analysis.insights_data)
        print(f"DEBUG insights.html: insights_data keys = {list(insights_data.keys())}")
        print(f"DEBUG insights.html: insights_data content = {insights_data}")
        
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
        print(f"DEBUG insights.html: Looking up insights for candidate: '{selected_candidate}'")
        print(f"DEBUG insights.html: All insights_data keys: {list(insights_data.keys())}")
        print(f"DEBUG insights.html: Found insights: {candidate_insights}")
        
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
        
        print(f"DEBUG insights.html: has_gpt_insights = {has_gpt_insights}")
        
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
                             pricing=pricing)
    
    @app.route('/unlock-candidate/<int:analysis_id>/<candidate_name>', methods=['POST'])
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
    
    @app.route('/download-candidate-pdf/<int:analysis_id>/<candidate_name>')
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
            return redirect(url_for('dashboard'))
        
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
            crit_lower = crit.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit] = 'Soft Skills'
            else:
                category_map[crit] = 'Other Requirements'
        
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
            return redirect(url_for('insights', analysis_id=analysis_id, candidate=candidate_name))
        
        # Return as download with sanitized filename
        safe_name = re.sub(r'[^\w\s-]', '', candidate_name).strip().replace(' ', '_')
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'{safe_name}_report.pdf'
        )
    
    @app.route('/criteria/<int:analysis_id>')
    @login_required
    def view_criteria(analysis_id):
        """View and edit criteria for an analysis"""
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
        criteria = json.loads(analysis.criteria_list)
        
        # Build category map (same logic as results page)
        criteria_with_categories = []
        for crit in criteria:
            crit_lower = crit.lower()
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
                'criterion': crit,
                'category': category
            })
        
        return render_template('criteria.html',
                             analysis_id=analysis_id,
                             job_title=analysis.job_title,
                             criteria_with_categories=criteria_with_categories)
    
    @app.route('/criteria/<int:analysis_id>/update', methods=['POST'])
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
                'redirect': url_for('results', analysis_id=analysis_id)
            })
        except Exception as e:
            return json.dumps({'success': False, 'message': str(e)})
    
    # Note: /buy-credits route now handled by payments blueprint
    
    @app.route('/account')
    @login_required
    def account():
        """Account balance and transaction history"""
        # Get transactions ordered newest first for display
        transactions = current_user.transactions.order_by(
            db.desc('created_at')
        ).limit(50).all()
        
        # Check which analysis IDs still exist (hard delete means missing = deleted)
        analysis_ids = [t.analysis_id for t in transactions if t.analysis_id]
        existing_analyses = set()
        transactions_analyses = {}
        if analysis_ids:
            all_analyses = Analysis.query.filter(Analysis.id.in_(analysis_ids)).all()
            existing_analyses = {a.id for a in all_analyses}
            transactions_analyses = {a.id: a for a in all_analyses}
        
        # Calculate running balance for display (forward in time: oldest to newest)
        # Reverse the list to process oldest first, then reverse back
        transactions_reversed = list(reversed(transactions))
        
        # Start with zero and calculate forward
        running_balance = Decimal('0')
        balances_forward = []
        
        for txn in transactions_reversed:
            # Add this transaction amount
            running_balance += Decimal(str(txn.amount_usd))
            # Store the balance AFTER this transaction
            balances_forward.append(float(running_balance))
        
        # Reverse balances back to match the newest-first display order
        balances = list(reversed(balances_forward))
        
        # Verify final balance matches user's actual balance
        if balances and abs(float(current_user.balance_usd) - balances[0]) > 0.01:
            discrepancy = float(current_user.balance_usd) - balances[0]
            print(f"⚠️ WARNING: Balance mismatch for user {current_user.id} ({current_user.email}). "
                  f"Calculated: ${balances[0]:.2f}, Actual: ${float(current_user.balance_usd):.2f}, "
                  f"Discrepancy: ${discrepancy:.2f}")
            
            # Send alert to admin (only once per user per session to avoid spam)
            if not hasattr(current_user, '_balance_alert_sent'):
                send_balance_mismatch_alert(current_user, float(current_user.balance_usd), balances[0], discrepancy)
                current_user._balance_alert_sent = True
        
        return render_template('account.html', 
                             user=current_user,
                             transactions=transactions,
                             balances=balances,
                             existing_analyses=existing_analyses,
                             transactions_analyses=transactions_analyses)
    
    @app.route('/delete-account', methods=['POST'])
    @login_required
    def delete_account():
        """Permanently delete user account and all associated data"""
        user_id = current_user.id
        user_email = current_user.email
        
        try:
            # Delete all user's analyses (cascades to candidate_files)
            Analysis.query.filter_by(user_id=user_id).delete()
            
            # Delete all transactions
            Transaction.query.filter_by(user_id=user_id).delete()
            
            # Delete feedback
            Feedback.query.filter_by(user_id=user_id).delete()
            
            # Delete user settings if they exist
            from database import UserSettings
            UserSettings.query.filter_by(user_id=user_id).delete()
            
            # Logout the user before deleting
            logout_user()
            
            # Finally delete the user account
            User.query.filter_by(id=user_id).delete()
            
            db.session.commit()
            
            flash(f'Account {user_email} has been permanently deleted.', 'success')
            return redirect(url_for('landing'))
            
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while deleting your account. Please contact support.', 'danger')
            return redirect(url_for('account'))
    
    @app.route('/job-history')
    @login_required
    def job_history():
        """Job analysis history - only show completed analyses"""
        # Filter to only show analyses with coverage_data (completed jobs)
        # This excludes incomplete analyses where the job failed
        analyses = Analysis.query.filter_by(user_id=current_user.id).filter(
            Analysis.coverage_data != '',
            Analysis.coverage_data.isnot(None)
        ).order_by(db.desc(Analysis.created_at)).limit(50).all()
        
        return render_template('job_history.html', 
                             user=current_user,
                             analyses=analyses)
    
    @app.route('/delete-analysis/<int:analysis_id>', methods=['POST'])
    @login_required
    def delete_analysis(analysis_id):
        """Hard delete an analysis and its related data"""
        analysis = Analysis.query.get_or_404(analysis_id)
        
        # Verify ownership
        if analysis.user_id != current_user.id:
            flash('Unauthorized', 'error')
            return redirect(url_for('job_history'))
        
        try:
            # Mark deletion timestamp on related transactions BEFORE deleting analysis
            deletion_time = datetime.now(timezone.utc)
            Transaction.query.filter_by(analysis_id=analysis_id).update({
                'analysis_deleted_at': deletion_time
            })
            
            # Delete related candidate files (cascade will handle this automatically due to ondelete='CASCADE')
            # Transaction records will have analysis_id set to NULL (due to ondelete='SET NULL')
            # But analysis_deleted_at will preserve when it was deleted
            
            db.session.delete(analysis)
            db.session.commit()
            
            flash('✅ Analysis deleted successfully', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting analysis: {str(e)}', 'error')
        
        return redirect(url_for('job_history'))
    
    @app.route('/help')
    def help():
        """Help and FAQ page"""
        return render_template('help.html', user=current_user)
    
    @app.route('/settings', methods=['GET', 'POST'])
    @login_required
    def settings():
        """User settings and preferences"""
        user_settings = UserSettings.get_or_create(current_user.id)
        
        if request.method == 'POST':
            try:
                # Update threshold preferences
                hi_threshold = request.form.get('hi_threshold', type=int)
                lo_threshold = request.form.get('lo_threshold', type=int)
                
                if hi_threshold and lo_threshold:
                    if lo_threshold >= hi_threshold:
                        flash('Low threshold must be less than high threshold', 'error')
                        return redirect(url_for('settings'))
                    if lo_threshold < 0 or hi_threshold > 100:
                        flash('Thresholds must be between 0 and 100', 'error')
                        return redirect(url_for('settings'))
                    
                    user_settings.default_hi_threshold = hi_threshold
                    user_settings.default_lo_threshold = lo_threshold
                
                # Update display preferences
                results_per_page = request.form.get('results_per_page', type=int)
                if results_per_page and 5 <= results_per_page <= 100:
                    user_settings.results_per_page = results_per_page
                
                show_percentages = request.form.get('show_percentages') == 'on'
                user_settings.show_percentages = show_percentages
                
                # Update export preferences
                include_evidence = request.form.get('include_evidence_by_default') == 'on'
                user_settings.include_evidence_by_default = include_evidence
                
                db.session.commit()
                flash('Settings updated successfully', 'success')
                return redirect(url_for('settings'))
                
            except Exception as e:
                db.session.rollback()
                flash(f'Error updating settings: {str(e)}', 'error')
                return redirect(url_for('settings'))
        
        return render_template('settings.html', 
                             user=current_user,
                             settings=user_settings)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        flash('Page not found', 'error')
        return redirect(url_for('dashboard') if current_user.is_authenticated else url_for('landing'))
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        flash('An internal error occurred', 'error')
        return redirect(url_for('dashboard') if current_user.is_authenticated else url_for('landing'))
    
    # ============================================================================
    # EXPORT ROUTES - Report generation and downloads
    # ============================================================================
    
    @app.route('/exports/<int:analysis_id>')
    @login_required
    def exports(analysis_id):
        """Main export page with all export options"""
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
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
        
        return render_template('export.html', 
                             analysis=analysis, 
                             coverage=coverage_df,
                             user_settings=user_settings,
                             is_current_draft_analysis=is_current_draft_analysis,
                             gpt_candidates=gpt_candidates_list)
    
    @app.route('/export/<int:analysis_id>/preview-pdf')
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
            crit_lower = crit.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit] = 'Soft Skills'
            else:
                category_map[crit] = 'Other Requirements'
        
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
    
    @app.route('/export/<int:analysis_id>/preview-pdf-inline')
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
                crit_lower = crit.lower()
                if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                    category_map[crit] = 'Technical Skills'
                elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                    category_map[crit] = 'Experience'
                elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                    category_map[crit] = 'Qualifications'
                elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                    category_map[crit] = 'Soft Skills'
                else:
                    category_map[crit] = 'Other Requirements'
            
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
    
    @app.route('/export/<int:analysis_id>/executive-pdf')
    @login_required
    def export_executive_pdf(analysis_id):
        """Download executive summary PDF"""
        from flask import send_file
        from export_utils import to_executive_summary_pdf
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
        # Parse data
        import pandas as pd
        from io import StringIO
        coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
        insights = json.loads(analysis.insights_data)
        criteria = json.loads(analysis.criteria_list)
        
        # Build category map
        category_map = {}
        for crit in criteria:
            crit_lower = crit.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit] = 'Soft Skills'
            else:
                category_map[crit] = 'Other Requirements'
        
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
            return redirect(url_for('exports', analysis_id=analysis_id))
        
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
    
    @app.route('/export/<int:analysis_id>/executive-docx')
    @login_required
    def export_executive_docx(analysis_id):
        """Download executive summary Word document"""
        from flask import send_file
        from export_utils import to_executive_summary_word
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
        # Parse data
        import pandas as pd
        from io import StringIO
        coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
        insights = json.loads(analysis.insights_data)
        criteria = json.loads(analysis.criteria_list)
        
        # Build category map
        category_map = {}
        for crit in criteria:
            crit_lower = crit.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit] = 'Soft Skills'
            else:
                category_map[crit] = 'Other Requirements'
        
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
            return redirect(url_for('exports', analysis_id=analysis_id))
        
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
    
    @app.route('/export/<int:analysis_id>/coverage-excel')
    @login_required
    def export_coverage_excel(analysis_id):
        """Download coverage matrix Excel file"""
        from flask import send_file
        from export_utils import to_excel_coverage_matrix
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
        # Parse data
        import pandas as pd
        from io import StringIO
        coverage_df = pd.read_json(StringIO(analysis.coverage_data), orient='records')
        criteria = json.loads(analysis.criteria_list)
        
        # Build category map
        category_map = {}
        for crit in criteria:
            crit_lower = crit.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit] = 'Soft Skills'
            else:
                category_map[crit] = 'Other Requirements'
        
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
            return redirect(url_for('exports', analysis_id=analysis_id))
        
        # Return as download with sanitized filename
        safe_title = re.sub(r'[^\w\s-]', '', analysis.job_title).strip().replace(' ', '_')[:50]
        return send_file(
            io.BytesIO(excel_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'coverage_matrix_{safe_title}.xlsx'
        )
    
    @app.route('/export/<int:analysis_id>/coverage-csv')
    @login_required
    def export_coverage_csv(analysis_id):
        """Download coverage matrix CSV (fallback)"""
        from flask import send_file
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
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
    
    @app.route('/export/<int:analysis_id>/individual-pdf', methods=['POST'])
    @login_required
    def export_individual_pdf(analysis_id):
        """Generate and merge individual candidate PDFs"""
        from flask import send_file
        from export_candidate import to_individual_candidate_pdf
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            return {'error': 'Analysis not found'}, 404
        
        # Get request data
        data = request.get_json()
        candidates = data.get('candidates', [])
        include_justifications = data.get('include_justifications', False)
        
        if not candidates:
            return {'error': 'No candidates selected'}, 400
        
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
            crit_lower = crit.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit] = 'Soft Skills'
            else:
                category_map[crit] = 'Other Requirements'
        
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
            return {'error': 'PDF generation failed'}, 500
        
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
            return {'error': f'PDF generation failed: {str(e)}'}, 500
    
    @app.route('/export/<int:analysis_id>/individual-docx', methods=['POST'])
    @login_required
    def export_individual_docx(analysis_id):
        """Generate individual candidate Word documents (ZIP if multiple)"""
        from flask import send_file
        from export_candidate import to_individual_candidate_docx
        import zipfile
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            return {'error': 'Analysis not found'}, 404
        
        # Get request data
        data = request.get_json()
        candidates = data.get('candidates', [])
        include_justifications = data.get('include_justifications', False)
        
        if not candidates:
            return {'error': 'No candidates selected'}, 400
        
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
            crit_lower = crit.lower()
            if any(word in crit_lower for word in ['python', 'java', 'sql', 'javascript', 'programming', 'coding', 'technical', 'software']):
                category_map[crit] = 'Technical Skills'
            elif any(word in crit_lower for word in ['experience', 'years', 'background']):
                category_map[crit] = 'Experience'
            elif any(word in crit_lower for word in ['education', 'degree', 'certification', 'qualification']):
                category_map[crit] = 'Qualifications'
            elif any(word in crit_lower for word in ['communication', 'leadership', 'team', 'collaboration']):
                category_map[crit] = 'Soft Skills'
            else:
                category_map[crit] = 'Other Requirements'
        
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
            return {'error': 'Word generation failed'}, 500
        
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
    
    @app.route('/export/<int:analysis_id>/candidates-csv')
    @login_required
    def export_candidates_csv(analysis_id):
        """Download simple candidate list CSV"""
        from flask import send_file
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
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
    
    @app.route('/export/<int:analysis_id>/criteria-csv')
    @login_required
    def export_criteria_csv(analysis_id):
        """Download criteria list CSV"""
        from flask import send_file
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
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
    # ADMIN ROUTES - Protected configuration panel
    # ============================================================================
    
    # Admin Security Helper Functions
    def get_client_ip():
        """Get client IP address (handles proxies/load balancers)"""
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        return request.remote_addr
    
    def check_brute_force(ip_address):
        """Check if IP is locked out due to failed attempts"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=15)
        recent_attempts = AdminLoginAttempt.query.filter(
            AdminLoginAttempt.ip_address == ip_address,
            AdminLoginAttempt.attempted_at > cutoff_time,
            AdminLoginAttempt.success == False
        ).count()
        
        return recent_attempts >= 5
    
    def record_login_attempt(ip_address, success):
        """Record admin login attempt"""
        attempt = AdminLoginAttempt(
            ip_address=ip_address,
            attempted_at=datetime.utcnow(),
            success=success
        )
        db.session.add(attempt)
        db.session.commit()
    
    def clear_login_attempts(ip_address):
        """Clear failed login attempts after successful login"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=15)
        AdminLoginAttempt.query.filter(
            AdminLoginAttempt.ip_address == ip_address,
            AdminLoginAttempt.attempted_at > cutoff_time
        ).delete()
        db.session.commit()
    
    def log_admin_action(action, details=None):
        """Log admin action for audit trail"""
        log = AdminAuditLog(
            action=action,
            details=json.dumps(details) if details else None,
            ip_address=get_client_ip(),
            user_agent=request.headers.get('User-Agent', '')[:255]
        )
        db.session.add(log)
        db.session.commit()
    
    @app.route('/admin/login', methods=['GET', 'POST'])
    def admin_login():
        """Admin login with brute-force protection and optional 2FA"""
        client_ip = get_client_ip()
        
        # Check for brute-force lockout
        if check_brute_force(client_ip):
            flash('⛔ Too many failed attempts. Please try again in 15 minutes.', 'danger')
            log_admin_action('login_blocked_brute_force', {'ip': client_ip})
            return render_template('admin_login.html', locked_out=True)
        
        if request.method == 'POST':
            password = request.form.get('password')
            totp_code = request.form.get('totp_code', '').strip()
            
            # Use environment variable or default password
            admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')
            
            # Step 1: Verify password
            if password != admin_password:
                record_login_attempt(client_ip, False)
                flash('❌ Invalid password', 'danger')
                log_admin_action('login_failed_password', {'ip': client_ip})
                return render_template('admin_login.html')
            
            # Step 2: Verify TOTP if enabled
            totp_secret = os.environ.get('ADMIN_TOTP_SECRET')
            if totp_secret:
                if not totp_code:
                    flash('⚠️ 2FA code required', 'warning')
                    return render_template('admin_login.html', password_verified=True)
                
                totp = pyotp.TOTP(totp_secret)
                if not totp.verify(totp_code, valid_window=1):
                    record_login_attempt(client_ip, False)
                    flash('❌ Invalid 2FA code', 'danger')
                    log_admin_action('login_failed_2fa', {'ip': client_ip})
                    return render_template('admin_login.html', password_verified=True)
            
            # Success - clear failed attempts and log in
            clear_login_attempts(client_ip)
            record_login_attempt(client_ip, True)
            session['admin_logged_in'] = True
            session['admin_last_activity'] = datetime.now(timezone.utc).isoformat()
            session.permanent = True
            log_admin_action('login_success', {'ip': client_ip})
            # No flash message - prevents it from showing on user pages if they navigate there
            return redirect(url_for('admin_settings'))
        
        # Check if 2FA is enabled
        totp_enabled = bool(os.environ.get('ADMIN_TOTP_SECRET'))
        return render_template('admin_login.html', totp_enabled=totp_enabled)
    
    
    @app.route('/admin/logout')
    def admin_logout():
        """Logout from admin panel"""
        log_admin_action('logout')
        session.pop('admin_logged_in', None)
        session.pop('admin_last_activity', None)
        flash('Logged out from admin panel', 'info')
        return redirect(url_for('admin_login'))
    
    
    def admin_required(f):
        """Decorator to protect admin routes with 30-minute inactivity timeout"""
        from functools import wraps
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('admin_logged_in'):
                flash('⚠️ Admin access required', 'warning')
                return redirect(url_for('admin_login'))
            
            # Check for inactivity timeout (30 minutes)
            last_activity_str = session.get('admin_last_activity')
            if last_activity_str:
                last_activity = datetime.fromisoformat(last_activity_str)
                inactivity = datetime.now(timezone.utc) - last_activity
                
                if inactivity > timedelta(minutes=30):
                    session.pop('admin_logged_in', None)
                    session.pop('admin_last_activity', None)
                    flash('⏱️ Session expired due to inactivity. Please login again.', 'warning')
                    return redirect(url_for('admin_login'))
            
            # Update last activity timestamp
            session['admin_last_activity'] = datetime.now(timezone.utc).isoformat()
            return f(*args, **kwargs)
        return decorated_function
    
    
    @app.route('/admin/setup-2fa')
    def admin_setup_2fa():
        """Generate QR code for 2FA setup (only accessible without 2FA enabled)"""
        # Only allow this if 2FA is not yet configured
        if os.environ.get('ADMIN_TOTP_SECRET'):
            flash('⚠️ 2FA is already configured. To reset, remove ADMIN_TOTP_SECRET from environment.', 'warning')
            return redirect(url_for('admin_login'))
        
        # Generate new secret
        secret = pyotp.random_base32()
        
        # Create provisioning URI for QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name='Admin',
            issuer_name='Candidate Evaluator'
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 for display
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return render_template('admin_2fa_setup.html', 
                             secret=secret, 
                             qr_code=img_str,
                             provisioning_uri=provisioning_uri)
    
    
    @app.route('/admin/audit-logs')
    @admin_required
    def admin_audit_logs():
        """View admin audit logs"""
        page = request.args.get('page', 1, type=int)
        per_page = 50
        
        logs = AdminAuditLog.query.order_by(
            AdminAuditLog.created_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        return render_template('admin_audit_logs.html', logs=logs, active_tab='audit')
    
    
    @app.route('/admin/business-health')
    @admin_required
    def admin_business_health():
        """Business Health Monitor and Calculator Settings"""
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt_settings.json')
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Load pricing from single source of truth
        pricing_path = os.path.join(os.path.dirname(__file__), 'config', 'pricing_settings.json')
        with open(pricing_path, 'r', encoding='utf-8') as f:
            pricing = json.load(f)
        
        message = request.args.get('message')
        return render_template('admin_business_health.html', 
                             settings=settings, 
                             pricing=pricing,
                             active_tab='business_health',
                             message=message)
    
    
    @app.route('/admin/business-health/save', methods=['POST'])
    @admin_required
    def admin_save_business_health():
        """Save calculator settings"""
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt_settings.json')
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Save calculator settings
        settings['calculator_settings']['avg_candidates_per_job']['value'] = int(request.form.get('avg_candidates', 50))
        settings['calculator_settings']['avg_criteria_per_job']['value'] = int(request.form.get('avg_criteria', 20))
        settings['calculator_settings']['insights_generated']['value'] = int(request.form.get('insights_generated', 5))
        
        # Update metadata
        settings['_metadata']['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Save back to file
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        
        return redirect(url_for('admin_business_health', message='Calculator settings saved successfully!'))
    
    
    @app.route('/admin/pricing')
    @admin_required
    def admin_pricing():
        """Pricing & Revenue Configuration - Single Source of Truth"""
        pricing_path = os.path.join(os.path.dirname(__file__), 'config', 'pricing_settings.json')
        with open(pricing_path, 'r', encoding='utf-8') as f:
            pricing = json.load(f)
        
        message = request.args.get('message')
        return render_template('admin_pricing.html', 
                             pricing=pricing, 
                             active_tab='pricing',
                             message=message)
    
    
    @app.route('/admin/pricing/save', methods=['POST'])
    @admin_required
    def admin_save_pricing():
        """Save pricing configuration"""
        pricing_path = os.path.join(os.path.dirname(__file__), 'config', 'pricing_settings.json')
        
        with open(pricing_path, 'r', encoding='utf-8') as f:
            pricing = json.load(f)
        
        # Update pricing values
        pricing['standard_tier_price']['value'] = float(request.form.get('standard_tier_price', 10.0))
        pricing['deep_dive_price']['value'] = float(request.form.get('deep_dive_price', 10.0))
        pricing['individual_insight_price']['value'] = float(request.form.get('individual_insight_price', 1.0))
        pricing['hiring_sprint_charge']['value'] = float(request.form.get('hiring_sprint_charge', 45.0))
        pricing['hiring_sprint_credit']['value'] = float(request.form.get('hiring_sprint_credit', 50.0))
        pricing['volume_bonus_threshold']['value'] = float(request.form.get('volume_bonus_threshold', 50.0))
        pricing['volume_bonus_percentage']['value'] = float(request.form.get('volume_bonus_percentage', 15.0))
        pricing['minimum_topup_amount']['value'] = float(request.form.get('minimum_topup_amount', 5.0))
        pricing['new_user_welcome_credit']['value'] = float(request.form.get('new_user_welcome_credit', 0.0))
        
        # Save back to file
        with open(pricing_path, 'w', encoding='utf-8') as f:
            json.dump(pricing, f, indent=2, ensure_ascii=False)
        
        return redirect(url_for('admin_pricing', message='Pricing configuration saved successfully! Changes are now live across the platform.'))
    
    
    @app.route('/admin')
    @admin_required
    def admin_settings():
        """Display admin settings panel - redirect to Users page"""
        return redirect(url_for('admin_users'))
    
    
    @app.route('/admin/run-migrations/<secret>')
    def admin_run_migrations(secret):
        """Emergency migration endpoint - run analytics migrations via HTTP"""
        # Security: require secret key
        if secret != "migrate-analytics-2026":
            return "Unauthorized", 403
        
        from sqlalchemy import text
        results = []
        
        try:
            # Migration 1: Basic Analytics
            results.append("📊 Running Basic Analytics Migration...")
            basic_fields = [
                ("completed_at", "TIMESTAMP"),
                ("processing_duration_seconds", "INTEGER"),
                ("exceeded_resume_limit", "BOOLEAN DEFAULT FALSE"),
                ("user_chose_override", "BOOLEAN DEFAULT FALSE"),
            ]
            
            for field_name, field_type in basic_fields:
                try:
                    sql = f"ALTER TABLE analyses ADD COLUMN {field_name} {field_type}"
                    db.session.execute(text(sql))
                    db.session.commit()
                    results.append(f"✅ Added: {field_name}")
                except Exception as e:
                    if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                        results.append(f"⏭️ Skipped: {field_name} (already exists)")
                    else:
                        results.append(f"❌ Error: {field_name}: {str(e)[:100]}")
            
            # Migration 2: Document Metrics
            results.append("<br><br>📄 Running Document Metrics Migration...")
            doc_fields = [
                ("jd_character_count", "INTEGER"),
                ("avg_resume_character_count", "INTEGER"),
                ("min_resume_character_count", "INTEGER"),
                ("max_resume_character_count", "INTEGER"),
            ]
            
            for field_name, field_type in doc_fields:
                try:
                    sql = f"ALTER TABLE analyses ADD COLUMN {field_name} {field_type}"
                    db.session.execute(text(sql))
                    db.session.commit()
                    results.append(f"✅ Added: {field_name}")
                except Exception as e:
                    if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                        results.append(f"⏭️ Skipped: {field_name} (already exists)")
                    else:
                        results.append(f"❌ Error: {field_name}: {str(e)[:100]}")
            
            results.append("<br><br>✅ Migrations Complete!")
            return "<br>".join(results), 200
            
        except Exception as e:
            results.append(f"<br><br>❌ CRITICAL ERROR: {str(e)}")
            return "<br>".join(results), 500
    
    
    @app.route('/admin/analytics')
    @admin_required
    def admin_analytics():
        """Display analytics dashboard with system metrics"""
        from sqlalchemy import func, desc
        from datetime import datetime, timedelta
        
        # Summary Stats
        total_users = User.query.count()
        total_jobs = Analysis.query.count()
        total_candidates = db.session.query(func.sum(Analysis.num_candidates)).scalar() or 0
        total_revenue = db.session.query(func.sum(Analysis.cost_usd)).scalar() or 0
        
        # Average candidates per job
        avg_candidates = db.session.query(func.avg(Analysis.num_candidates)).scalar() or 0
        
        # Calculate median (SQLite-compatible)
        all_candidate_counts = [a.num_candidates for a in Analysis.query.with_entities(Analysis.num_candidates).all()]
        median_candidates = sorted(all_candidate_counts)[len(all_candidate_counts)//2] if all_candidate_counts else 0
        
        max_candidates = db.session.query(func.max(Analysis.num_candidates)).scalar() or 0
        
        # Processing times (only for jobs with data)
        avg_time = db.session.query(func.avg(Analysis.processing_duration_seconds)).filter(
            Analysis.processing_duration_seconds.isnot(None)
        ).scalar() or 0
        min_time = db.session.query(func.min(Analysis.processing_duration_seconds)).filter(
            Analysis.processing_duration_seconds.isnot(None)
        ).scalar() or 0
        max_time = db.session.query(func.max(Analysis.processing_duration_seconds)).filter(
            Analysis.processing_duration_seconds.isnot(None)
        ).scalar() or 0
        
        # Warning patterns
        warnings_shown = Analysis.query.filter_by(exceeded_resume_limit=True).count()
        overrides_chosen = Analysis.query.filter_by(exceeded_resume_limit=True, user_chose_override=True).count()
        override_rate = (overrides_chosen / warnings_shown * 100) if warnings_shown > 0 else 0
        
        # Jobs by day (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        jobs_by_day = db.session.query(
            func.date(Analysis.created_at).label('date'),
            func.count(Analysis.id).label('count')
        ).filter(
            Analysis.created_at >= thirty_days_ago
        ).group_by(func.date(Analysis.created_at)).order_by('date').all()
        
        # Handle both string dates (SQLite) and date objects (PostgreSQL)
        jobs_labels = []
        for row in jobs_by_day:
            if isinstance(row.date, str):
                # SQLite returns string like '2026-01-01'
                jobs_labels.append(row.date[5:].replace('-', '/'))
            else:
                jobs_labels.append(row.date.strftime('%m/%d'))
        jobs_data = [row.count for row in jobs_by_day]
        
        # Revenue by day (last 30 days)
        revenue_by_day = db.session.query(
            func.date(Analysis.created_at).label('date'),
            func.sum(Analysis.cost_usd).label('revenue')
        ).filter(
            Analysis.created_at >= thirty_days_ago
        ).group_by(func.date(Analysis.created_at)).order_by('date').all()
        
        # Handle both string dates (SQLite) and date objects (PostgreSQL)
        revenue_labels = []
        for row in revenue_by_day:
            if isinstance(row.date, str):
                # SQLite returns string like '2026-01-01'
                revenue_labels.append(row.date[5:].replace('-', '/'))
            else:
                revenue_labels.append(row.date.strftime('%m/%d'))
        revenue_data = [float(row.revenue) for row in revenue_by_day]
        
        # Candidates distribution
        candidates_dist = [
            Analysis.query.filter(Analysis.num_candidates.between(1, 10)).count(),
            Analysis.query.filter(Analysis.num_candidates.between(11, 50)).count(),
            Analysis.query.filter(Analysis.num_candidates.between(51, 100)).count(),
            Analysis.query.filter(Analysis.num_candidates.between(101, 200)).count(),
            Analysis.query.filter(Analysis.num_candidates > 200).count(),
        ]
        
        # Processing time vs candidates (scatter data)
        time_data = db.session.query(
            Analysis.num_candidates,
            Analysis.processing_duration_seconds
        ).filter(
            Analysis.processing_duration_seconds.isnot(None)
        ).all()
        
        time_vs_candidates = [{'x': row[0], 'y': row[1]} for row in time_data]
        
        # Document size analytics
        avg_jd_chars = db.session.query(func.avg(Analysis.jd_character_count)).filter(
            Analysis.jd_character_count.isnot(None)
        ).scalar() or 0
        
        # Get all resume character counts for bell curve
        all_resume_chars = []
        for analysis in Analysis.query.filter(Analysis.avg_resume_character_count.isnot(None)).all():
            # Approximate: use avg, min, max to estimate distribution
            if analysis.avg_resume_character_count:
                all_resume_chars.append(analysis.avg_resume_character_count)
        
        # Create histogram buckets for bell curve (0-20k in 2k increments)
        resume_size_buckets = list(range(0, 22000, 2000))
        resume_size_labels = [f"{i//1000}-{(i+2000)//1000}k" for i in resume_size_buckets[:-1]]
        resume_size_distribution = [0] * (len(resume_size_buckets) - 1)
        
        for char_count in all_resume_chars:
            for i in range(len(resume_size_buckets) - 1):
                if resume_size_buckets[i] <= char_count < resume_size_buckets[i+1]:
                    resume_size_distribution[i] += 1
                    break
        
        avg_resume_chars = int(sum(all_resume_chars) / len(all_resume_chars)) if all_resume_chars else 0
        min_resume_chars = db.session.query(func.min(Analysis.min_resume_character_count)).filter(
            Analysis.min_resume_character_count.isnot(None)
        ).scalar() or 0
        max_resume_chars = db.session.query(func.max(Analysis.max_resume_character_count)).filter(
            Analysis.max_resume_character_count.isnot(None)
        ).scalar() or 0
        
        # Top users
        top_users = db.session.query(
            User,
            func.sum(Analysis.num_candidates).label('total_candidates'),
            func.avg(Analysis.num_candidates).label('avg_batch')
        ).join(
            Analysis, User.id == Analysis.user_id
        ).group_by(User.id).order_by(
            desc(User.total_analyses_count)
        ).limit(10).all()
        
        top_users_data = [{
            'email': user.email,
            'total_analyses_count': user.total_analyses_count,
            'total_candidates': int(total_cand or 0),
            'total_revenue_usd': user.total_revenue_usd,
            'avg_batch': float(avg_b or 0),
            'last_seen': user.last_seen
        } for user, total_cand, avg_b in top_users]
        
        # Recent jobs
        recent_jobs = db.session.query(
            Analysis,
            User.email.label('user_email')
        ).join(
            User, Analysis.user_id == User.id
        ).order_by(
            desc(Analysis.created_at)
        ).limit(20).all()
        
        recent_jobs_data = [{
            'id': analysis.id,
            'user_email': user_email,
            'job_title': analysis.job_title or 'Untitled',
            'num_candidates': analysis.num_candidates,
            'cost_usd': analysis.cost_usd,
            'processing_duration_seconds': analysis.processing_duration_seconds,
            'created_at': analysis.created_at,
            'completed_at': analysis.completed_at,
            'exceeded_resume_limit': analysis.exceeded_resume_limit or False,
            'user_chose_override': analysis.user_chose_override or False
        } for analysis, user_email in recent_jobs]
        
        stats = {
            'total_users': total_users,
            'total_jobs': total_jobs,
            'total_candidates': total_candidates,
            'total_revenue': total_revenue,
            'avg_candidates_per_job': avg_candidates,
            'median_candidates': int(median_candidates),
            'max_candidates': max_candidates,
            'avg_processing_time': int(avg_time),
            'min_processing_time': int(min_time),
            'max_processing_time': int(max_time),
            'warnings_shown': warnings_shown,
            'overrides_chosen': overrides_chosen,
            'override_rate': override_rate,
            'jobs_by_day': {'labels': jobs_labels, 'data': jobs_data},
            'revenue_by_day': {'labels': revenue_labels, 'data': revenue_data},
            'candidates_distribution': candidates_dist,
            'time_vs_candidates': time_vs_candidates,
            'avg_jd_chars': int(avg_jd_chars),
            'avg_resume_chars': avg_resume_chars,
            'min_resume_chars': min_resume_chars,
            'max_resume_chars': max_resume_chars,
            'resume_size_labels': resume_size_labels,
            'resume_size_distribution': resume_size_distribution,
            'top_users': top_users_data,
            'recent_jobs': recent_jobs_data
        }
        
        return render_template('admin_analytics.html', stats=stats, active_tab='analytics')
    
    
    @app.route('/admin/gpt')
    @admin_required
    def admin_gpt_settings():
        """Display GPT settings tab with two-agent configuration"""
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt_settings.json')
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        message = request.args.get('message')
        return render_template('admin_gpt.html', settings=settings, message=message, active_tab='gpt')
    
    
    @app.route('/admin/migrate-db')
    @admin_required
    def admin_migrate_db():
        """Run database migration - add analysis_deleted_at to transactions, remove deleted_at from analyses"""
        try:
            from sqlalchemy import text
            
            # Add column to transactions
            db.session.execute(text("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS analysis_deleted_at TIMESTAMP;"))
            
            # Remove column from analyses
            db.session.execute(text("ALTER TABLE analyses DROP COLUMN IF EXISTS deleted_at;"))
            
            db.session.commit()
            
            return """
            <html>
            <head><title>Migration Complete</title></head>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1 style="color: #10b981;">✅ Migration Successful!</h1>
                <p>Database schema updated:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>Added <code>analysis_deleted_at</code> to transactions table</li>
                    <li>Removed <code>deleted_at</code> from analyses table</li>
                </ul>
                <p style="margin-top: 30px;">
                    <a href="/admin/gpt" style="color: #2563eb;">← Back to Admin Panel</a>
                </p>
            </body>
            </html>
            """
        except Exception as e:
            db.session.rollback()
            return f"""
            <html>
            <head><title>Migration Error</title></head>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1 style="color: #dc2626;">❌ Migration Failed</h1>
                <p>Error: {str(e)}</p>
                <p style="margin-top: 30px;">
                    <a href="/admin/gpt" style="color: #2563eb;">← Back to Admin Panel</a>
                </p>
            </body>
            </html>
            """
    
    @app.route('/admin/save', methods=['POST'])
    @admin_required
    def admin_save_settings():
        """Save updated two-agent settings"""
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt_settings.json')
        
        # Load current settings
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Update two-agent core settings
        settings['ranker_model']['value'] = request.form.get('ranker_model')
        settings['insight_model']['value'] = request.form.get('insight_model')
        settings['ranker_temperature']['value'] = float(request.form.get('ranker_temperature'))
        settings['insight_temperature']['value'] = float(request.form.get('insight_temperature'))
        
        # Update advanced API settings
        settings['advanced_api_settings']['presence_penalty']['value'] = float(request.form.get('presence_penalty'))
        settings['advanced_api_settings']['frequency_penalty']['value'] = float(request.form.get('frequency_penalty'))
        settings['advanced_api_settings']['ranker_max_tokens']['value'] = int(request.form.get('ranker_max_tokens'))
        settings['advanced_api_settings']['insight_max_tokens']['value'] = int(request.form.get('insight_max_tokens'))
        
        # Update text processing settings
        settings['text_processing']['candidate_text_chars']['value'] = int(request.form.get('candidate_text_chars'))
        settings['text_processing']['jd_text_chars']['value'] = int(request.form.get('jd_text_chars'))
        settings['text_processing']['evidence_snippet_chars']['value'] = int(request.form.get('evidence_snippet_chars'))
        
        # Update insight formatting
        settings['insight_formatting']['notes_length']['value'] = request.form.get('notes_length')
        settings['insight_formatting']['insight_tone']['value'] = request.form.get('insight_tone')
        
        # Update report balance
        settings['report_balance']['min_strengths']['value'] = int(request.form.get('min_strengths'))
        settings['report_balance']['max_strengths']['value'] = int(request.form.get('max_strengths'))
        settings['report_balance']['min_gaps']['value'] = int(request.form.get('min_gaps'))
        settings['report_balance']['max_gaps']['value'] = int(request.form.get('max_gaps'))
        
        # Update evidence depth
        settings['evidence_depth']['top_evidence_items']['value'] = int(request.form.get('top_evidence_items'))
        
        # Update evidence thresholds
        settings['evidence_thresholds']['strong_match_threshold']['value'] = float(request.form.get('strong_match_threshold'))
        settings['evidence_thresholds']['good_match_threshold']['value'] = float(request.form.get('good_match_threshold'))
        settings['evidence_thresholds']['moderate_match_threshold']['value'] = float(request.form.get('moderate_match_threshold'))
        settings['evidence_thresholds']['weak_match_threshold']['value'] = float(request.form.get('weak_match_threshold'))
        
        # Update score thresholds
        settings['score_thresholds']['high_threshold']['value'] = float(request.form.get('high_threshold'))
        settings['score_thresholds']['low_threshold']['value'] = float(request.form.get('low_threshold'))
        
        # Update metadata
        settings['_metadata']['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Save back to file
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        
        flash('✅ Settings saved successfully! Two-agent configuration active.', 'admin')
        return redirect(url_for('admin_gpt_settings'))
    
    
    @app.route('/admin/prompts')
    @admin_required
    def admin_prompts():
        """AI Prompts management panel"""
        prompts_path = os.path.join(os.path.dirname(__file__), 'config', 'prompts.json')
        
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        message = request.args.get('message')
        return render_template('admin_prompts.html', 
                             prompts=prompts, 
                             message=message,
                             active_tab='prompts')
    
    
    @app.route('/admin/prompts/save', methods=['POST'])
    @admin_required
    def admin_save_prompts():
        """Save updated prompts for two-agent system"""
        prompts_path = os.path.join(os.path.dirname(__file__), 'config', 'prompts.json')
        
        # Load current prompts
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        # Update RANKER prompts (nested structure with .value)
        if 'ranker_scoring' in prompts:
            if 'system_prompt' in prompts['ranker_scoring']:
                prompts['ranker_scoring']['system_prompt']['value'] = request.form.get('ranker_system_prompt', prompts['ranker_scoring']['system_prompt'].get('value', ''))
            if 'user_prompt_template' in prompts['ranker_scoring']:
                prompts['ranker_scoring']['user_prompt_template']['value'] = request.form.get('ranker_user_prompt', prompts['ranker_scoring']['user_prompt_template'].get('value', ''))
        
        # Update INSIGHT prompts (nested structure with .value)
        if 'insight_generation' in prompts:
            if 'system_prompt' in prompts['insight_generation']:
                prompts['insight_generation']['system_prompt']['value'] = request.form.get('insight_system_prompt', prompts['insight_generation']['system_prompt'].get('value', ''))
            if 'user_prompt_template' in prompts['insight_generation']:
                prompts['insight_generation']['user_prompt_template']['value'] = request.form.get('insight_user_prompt', prompts['insight_generation']['user_prompt_template'].get('value', ''))
        
        # Update JD Extraction prompts (nested structure with .value)
        if 'jd_extraction' in prompts:
            if 'system_prompt' in prompts['jd_extraction']:
                prompts['jd_extraction']['system_prompt']['value'] = request.form.get('jd_system_prompt', prompts['jd_extraction']['system_prompt'].get('value', ''))
            if 'user_prompt_template' in prompts['jd_extraction']:
                prompts['jd_extraction']['user_prompt_template']['value'] = request.form.get('jd_user_prompt', prompts['jd_extraction']['user_prompt_template'].get('value', ''))
        
        # Update metadata if it exists
        if '_metadata' in prompts:
            prompts['_metadata']['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            prompts['_metadata']['updated_by'] = 'admin'
        
        # Save back to file
        with open(prompts_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        
        return redirect(url_for('admin_prompts') + '?message=Prompts saved successfully! Two-agent configuration active.')
    
    
    @app.route('/admin/users')
    @admin_required
    def admin_users():
        """User management panel - list all users"""
        search = request.args.get('search', '').strip()
        sort_by = request.args.get('sort', 'created_at')
        online_filter = request.args.get('online', '')  # Filter for online users
        
        query = User.query
        
        # Online filter (users active in last 5 minutes)
        if online_filter == 'yes':
            five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
            query = query.filter(User.last_seen >= five_minutes_ago)
        elif online_filter == 'no':
            five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
            query = query.filter(
                db.or_(
                    User.last_seen < five_minutes_ago,
                    User.last_seen.is_(None)
                )
            )
        
        # Search filter
        if search:
            query = query.filter(
                db.or_(
                    User.email.ilike(f'%{search}%'),
                    User.name.ilike(f'%{search}%')
                )
            )
        
        # Sorting
        if sort_by == 'email':
            query = query.order_by(User.email)
        elif sort_by == 'balance':
            query = query.order_by(User.balance_usd.desc())
        elif sort_by == 'analyses':
            query = query.order_by(User.total_analyses_count.desc())
        elif sort_by == 'revenue':
            query = query.order_by(User.total_revenue_usd.desc())
        else:  # created_at
            query = query.order_by(User.created_at.desc())
        
        users = query.all()
        
        # Calculate stats
        total_users = User.query.count()
        total_revenue = db.session.query(db.func.sum(User.total_revenue_usd)).scalar() or 0
        total_analyses = db.session.query(db.func.sum(User.total_analyses_count)).scalar() or 0
        
        users = query.all()
        
        # Calculate stats
        total_users = User.query.count()
        total_revenue = db.session.query(db.func.sum(User.total_revenue_usd)).scalar() or 0
        total_analyses = db.session.query(db.func.sum(User.total_analyses_count)).scalar() or 0
        
        # Count online users (active in last 5 minutes)
        five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
        online_users = User.query.filter(User.last_seen >= five_minutes_ago).count()
        
        return render_template('admin_users.html', 
                             users=users, 
                             search=search,
                             sort_by=sort_by,
                             online_filter=online_filter,
                             active_tab='users',
                             total_users=total_users,
                             online_users=online_users,
                             total_revenue=total_revenue,
                             total_analyses=total_analyses)
    
    
    @app.route('/admin/users/<int:user_id>')
    @admin_required
    def admin_user_detail(user_id):
        """View detailed information about a specific user"""
        user = User.query.get_or_404(user_id)
        
        # Get user's analyses
        analyses = Analysis.query.filter_by(user_id=user_id).order_by(Analysis.created_at.desc()).all()
        
        # Get user's transactions
        transactions = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.created_at.desc()).limit(50).all()
        
        # Calculate bonus/promotional funds (not refundable)
        bonus_total = Decimal('0')
        for txn in Transaction.query.filter_by(user_id=user_id).all():
            # Check if transaction is a bonus/promotional type
            if any(keyword in txn.description.lower() for keyword in ['sign-up bonus', 'volume bonus', 'promotional']):
                if txn.amount_usd > 0:  # Only count positive bonuses
                    bonus_total += Decimal(str(txn.amount_usd))
        
        # Calculate maximum refundable balance
        max_refundable = max(Decimal('0'), user.balance_usd - bonus_total)
        
        return render_template('admin_user_detail.html',
                             user=user,
                             analyses=analyses,
                             transactions=transactions,
                             bonus_total=bonus_total,
                             max_refundable=max_refundable,
                             active_tab='users')
    
    
    @app.route('/admin/users/<int:user_id>/suspend', methods=['POST'])
    @admin_required
    def admin_suspend_user(user_id):
        """Suspend a user account"""
        user = User.query.get_or_404(user_id)
        reason = request.form.get('reason', '').strip()
        
        user.is_suspended = True
        user.suspension_reason = reason if reason else None
        db.session.commit()
        
        log_admin_action('user_suspended', {
            'user_id': user_id,
            'email': user.email,
            'reason': reason
        })
        
        flash(f'✅ User {user.email} has been suspended', 'admin')
        return redirect(url_for('admin_user_detail', user_id=user_id))
    
    
    @app.route('/admin/users/<int:user_id>/unsuspend', methods=['POST'])
    @admin_required
    def admin_unsuspend_user(user_id):
        """Unsuspend a user account"""
        user = User.query.get_or_404(user_id)
        
        user.is_suspended = False
        user.suspension_reason = None
        db.session.commit()
        
        log_admin_action('user_unsuspended', {
            'user_id': user_id,
            'email': user.email
        })
        
        flash(f'✅ User {user.email} has been unsuspended', 'admin')
        return redirect(url_for('admin_user_detail', user_id=user_id))
    
    
    # ============================================================================
    # BALANCE MONITORING & RECONCILIATION
    # ============================================================================
    
    def check_user_balance_integrity(user_id):
        """
        Check if a user's balance matches their transaction history.
        Returns (is_valid, actual_balance, calculated_balance, discrepancy)
        """
        from decimal import Decimal
        
        user = User.query.get(user_id)
        if not user:
            return (False, 0, 0, 0)
        
        # Calculate balance from transaction history
        transactions = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.created_at).all()
        calculated_balance = Decimal('0')
        for txn in transactions:
            calculated_balance += Decimal(str(txn.amount_usd))
        
        actual_balance = Decimal(str(user.balance_usd))
        discrepancy = actual_balance - calculated_balance
        
        # Allow 1 cent tolerance for floating point rounding
        is_valid = abs(discrepancy) < Decimal('0.01')
        
        return (is_valid, float(actual_balance), float(calculated_balance), float(discrepancy))
    
    
    def send_balance_mismatch_alert(user, actual_balance, calculated_balance, discrepancy):
        """Send email alert to admin about balance mismatch"""
        try:
            from email_utils import send_email
            
            admin_email = os.environ.get('ADMIN_EMAIL', 'admin@candidateevaluator.com')
            
            html_body = f"""
            <h2>🚨 Balance Mismatch Detected</h2>
            <p>A balance discrepancy has been detected for a user account.</p>
            
            <h3>User Details:</h3>
            <ul>
                <li><strong>User ID:</strong> {user.id}</li>
                <li><strong>Email:</strong> {user.email}</li>
                <li><strong>Registration:</strong> {user.created_at.strftime('%Y-%m-%d %H:%M UTC')}</li>
            </ul>
            
            <h3>Balance Information:</h3>
            <ul>
                <li><strong>Actual Balance (Database):</strong> ${actual_balance:.2f}</li>
                <li><strong>Calculated Balance (Transactions):</strong> ${calculated_balance:.2f}</li>
                <li><strong>Discrepancy:</strong> ${discrepancy:.2f}</li>
            </ul>
            
            <p><a href="{url_for('admin_balance_audit', _external=True)}">View Balance Audit Dashboard</a></p>
            
            <p style="color: #666; font-size: 12px;">This alert was generated automatically by the balance monitoring system.</p>
            """
            
            send_email(
                subject=f'🚨 Balance Mismatch Alert - User {user.email}',
                recipients=[admin_email],
                html_body=html_body
            )
            print(f"✅ Balance mismatch alert sent to {admin_email}")
        except Exception as e:
            print(f"❌ Failed to send balance mismatch alert: {e}")
    
    
    @app.route('/admin/balance-audit')
    @admin_required
    def admin_balance_audit():
        """Admin dashboard showing users with balance mismatches"""
        users_with_issues = []
        
        # Check all users with transactions
        all_users = User.query.all()
        for user in all_users:
            is_valid, actual, calculated, discrepancy = check_user_balance_integrity(user.id)
            if not is_valid:
                users_with_issues.append({
                    'user': user,
                    'actual_balance': actual,
                    'calculated_balance': calculated,
                    'discrepancy': discrepancy,
                    'transaction_count': Transaction.query.filter_by(user_id=user.id).count()
                })
        
        return render_template('admin_balance_audit.html', 
                             users_with_issues=users_with_issues,
                             active_tab='balance_audit')
    
    
    @app.route('/admin/balance-adjustment/<int:user_id>', methods=['GET', 'POST'])
    @admin_required
    def admin_balance_adjustment(user_id):
        """Create manual balance adjustment transaction"""
        user = User.query.get_or_404(user_id)
        
        if request.method == 'POST':
            from decimal import Decimal
            
            adjustment_amount = Decimal(str(request.form.get('adjustment_amount', 0)))
            reason = request.form.get('reason', 'Manual balance adjustment by admin')
            
            if abs(adjustment_amount) < Decimal('0.01'):
                flash('Adjustment amount must be at least $0.01', 'danger')
                return redirect(url_for('admin_balance_adjustment', user_id=user_id))
            
            # Create adjustment transaction
            if adjustment_amount > 0:
                user.add_funds(
                    amount_usd=adjustment_amount,
                    description=f'Admin Balance Adjustment: {reason}'
                )
            else:
                user.deduct_funds(
                    amount_usd=abs(adjustment_amount),
                    description=f'Admin Balance Adjustment: {reason}'
                )
            
            db.session.commit()
            
            log_admin_action('balance_adjustment', {
                'user_id': user_id,
                'user_email': user.email,
                'adjustment_amount': float(adjustment_amount),
                'reason': reason,
                'new_balance': float(user.balance_usd)
            })
            
            flash(f'✅ Balance adjusted by ${adjustment_amount:.2f} for {user.email}', 'admin')
            return redirect(url_for('admin_balance_audit'))
        
        # GET request - show adjustment form
        is_valid, actual, calculated, discrepancy = check_user_balance_integrity(user_id)
        
        return render_template('admin_balance_adjustment.html',
                             user=user,
                             actual_balance=actual,
                             calculated_balance=calculated,
                             discrepancy=discrepancy,
                             suggested_adjustment=-discrepancy,  # Negative of discrepancy to fix it
                             active_tab='balance_audit')
    
    
    @app.route('/admin/system')
    @admin_required
    def admin_system():
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'system_settings.json')
        
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        message = request.args.get('message')
        return render_template('admin_system.html', 
                             settings=settings, 
                             message=message,
                             active_tab='system')
    
    @app.route('/admin/failed-jobs')
    @admin_required
    def admin_failed_jobs():
        """Admin page showing failed job analyses with error details"""
        from datetime import timedelta
        
        # Get all failed jobs (have error_message but no coverage_data)
        failed_jobs = Analysis.query.filter(
            Analysis.error_message.isnot(None),
            Analysis.error_message != ''
        ).order_by(db.desc(Analysis.failed_at)).limit(50).all()
        
        # Calculate stats
        now = datetime.utcnow()
        failed_last_24h = sum(1 for job in failed_jobs if job.failed_at and (now - job.failed_at).total_seconds() < 86400)
        affected_users = list(set(job.user_id for job in failed_jobs))
        
        # Extract common error patterns
        error_patterns = {}
        for job in failed_jobs:
            if job.error_message:
                # Get first line of error (usually the exception type)
                error_line = job.error_message.split('\n')[0]
                error_type = error_line.split(':')[0] if ':' in error_line else error_line[:50]
                
                if error_type not in error_patterns:
                    error_patterns[error_type] = {'count': 0, 'example': error_line}
                error_patterns[error_type]['count'] += 1
        
        common_errors = [(k, v['count'], v['example']) for k, v in sorted(error_patterns.items(), key=lambda x: x[1]['count'], reverse=True)]
        
        return render_template('admin_failed_jobs.html',
                             user=current_user,
                             failed_jobs=failed_jobs,
                             failed_last_24h=failed_last_24h,
                             affected_users=affected_users,
                             common_errors=common_errors,
                             active_tab='failed_jobs')
    
    
    @app.route('/admin/system/save', methods=['POST'])
    @admin_required
    def admin_system_save():
        """Save system settings"""
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'system_settings.json')
        
        # Load current settings
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        # Update values from form
        settings['registration_enabled']['value'] = request.form.get('registration_enabled') == 'on'
        settings['maintenance_mode']['value'] = request.form.get('maintenance_mode') == 'on'
        settings['max_file_size_mb']['value'] = int(request.form.get('max_file_size_mb', 10))
        settings['enable_document_length_warnings']['value'] = request.form.get('enable_document_length_warnings') == 'on'
        settings['max_resumes_per_upload']['value'] = int(request.form.get('max_resumes_per_upload', 200))
        
        # Update metadata
        settings['_metadata']['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        settings['_metadata']['updated_by'] = 'admin'
        
        # Save back to file
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        flash('✅ System settings saved successfully!', 'admin')
        return redirect(url_for('admin_system'))
    
    
    @app.route('/admin/stats')
    @admin_required
    def admin_stats():
        """Usage statistics and analytics"""
        from datetime import timedelta as td
        
        # User stats
        total_users = User.query.count()
        users_with_analyses = User.query.filter(User.total_analyses_count > 0).count()
        
        # Revenue stats
        total_revenue = db.session.query(db.func.sum(User.total_revenue_usd)).scalar() or 0
        total_balance = db.session.query(db.func.sum(User.balance_usd)).scalar() or 0
        
        # Analysis stats
        total_analyses = Analysis.query.count()
        analyses_this_month = Analysis.query.filter(
            Analysis.created_at >= datetime.now().replace(day=1, hour=0, minute=0, second=0)
        ).count()
        
        # Recent signups (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_signups = User.query.filter(User.created_at >= thirty_days_ago).count()
        
        # Top users by revenue
        top_revenue_users = User.query.order_by(User.total_revenue_usd.desc()).limit(10).all()
        
        # Top users by analyses
        top_analysis_users = User.query.order_by(User.total_analyses_count.desc()).limit(10).all()
        
        # Recent analyses
        recent_analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(20).all()
        
        # Financial reporting with date range
        from_date = request.args.get('from_date')
        to_date = request.args.get('to_date')
        
        # Default to last 30 days if no dates specified
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        # Parse dates - assume user input is in AEST, convert to UTC for database query
        start_datetime = datetime.strptime(from_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(to_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
        
        # Convert AEST to UTC (subtract 10 hours)
        start_datetime_utc = start_datetime - timedelta(hours=10)
        end_datetime_utc = end_datetime - timedelta(hours=10)
        
        # Query transactions in date range (using UTC times for database)
        transactions_query = Transaction.query.filter(
            Transaction.created_at >= start_datetime_utc,
            Transaction.created_at <= end_datetime_utc
        )
        
        # Calculate financial metrics
        signup_bonuses = db.session.query(db.func.sum(Transaction.amount_usd)).filter(
            Transaction.created_at >= start_datetime_utc,
            Transaction.created_at <= end_datetime_utc,
            Transaction.description.ilike('%sign-up bonus%')
        ).scalar() or Decimal('0')
        
        volume_bonuses = db.session.query(db.func.sum(Transaction.amount_usd)).filter(
            Transaction.created_at >= start_datetime_utc,
            Transaction.created_at <= end_datetime_utc,
            Transaction.description.ilike('%volume bonus%')
        ).scalar() or Decimal('0')
        
        refunds = abs(db.session.query(db.func.sum(Transaction.amount_usd)).filter(
            Transaction.created_at >= start_datetime_utc,
            Transaction.created_at <= end_datetime_utc,
            Transaction.description.ilike('%refund%')
        ).scalar() or Decimal('0'))
        
        stripe_revenue = db.session.query(db.func.sum(Transaction.amount_usd)).filter(
            Transaction.created_at >= start_datetime_utc,
            Transaction.created_at <= end_datetime_utc,
            Transaction.description.ilike('stripe purchase%')
        ).scalar() or Decimal('0')
        
        analysis_spending = abs(db.session.query(db.func.sum(Transaction.amount_usd)).filter(
            Transaction.created_at >= start_datetime_utc,
            Transaction.created_at <= end_datetime_utc,
            Transaction.description.ilike('analysis:%')
        ).scalar() or Decimal('0'))
        
        manual_credits = db.session.query(db.func.sum(Transaction.amount_usd)).filter(
            Transaction.created_at >= start_datetime_utc,
            Transaction.created_at <= end_datetime_utc,
            Transaction.description.ilike('%manual credit%')
        ).scalar() or Decimal('0')
        
        manual_debits = abs(db.session.query(db.func.sum(Transaction.amount_usd)).filter(
            Transaction.created_at >= start_datetime_utc,
            Transaction.created_at <= end_datetime_utc,
            Transaction.description.ilike('%manual debit%')
        ).scalar() or Decimal('0'))
        
        # Transaction counts
        signup_bonus_count = transactions_query.filter(Transaction.description.ilike('%sign-up bonus%')).count()
        volume_bonus_count = transactions_query.filter(Transaction.description.ilike('%volume bonus%')).count()
        refund_count = transactions_query.filter(Transaction.description.ilike('%refund%')).count()
        stripe_purchase_count = transactions_query.filter(Transaction.description.ilike('stripe purchase%')).count()
        
        # Calculate totals
        total_promotional = signup_bonuses + volume_bonuses
        total_revenue_period = stripe_revenue
        net_balance_change = stripe_revenue + manual_credits - refunds - manual_debits - analysis_spending
        
        return render_template('admin_stats.html',
                             active_tab='stats',
                             total_users=total_users,
                             users_with_analyses=users_with_analyses,
                             total_revenue=total_revenue,
                             total_balance=total_balance,
                             total_analyses=total_analyses,
                             analyses_this_month=analyses_this_month,
                             recent_signups=recent_signups,
                             top_revenue_users=top_revenue_users,
                             top_analysis_users=top_analysis_users,
                             recent_analyses=recent_analyses,
                             timedelta=td,  # Pass timedelta to template for timezone conversion
                             # Financial reporting
                             from_date=from_date,
                             to_date=to_date,
                             signup_bonuses=signup_bonuses,
                             volume_bonuses=volume_bonuses,
                             total_promotional=total_promotional,
                             refunds=refunds,
                             stripe_revenue=stripe_revenue,
                             analysis_spending=analysis_spending,
                             manual_credits=manual_credits,
                             manual_debits=manual_debits,
                             net_balance_change=net_balance_change,
                             signup_bonus_count=signup_bonus_count,
                             volume_bonus_count=volume_bonus_count,
                             refund_count=refund_count,
                             stripe_purchase_count=stripe_purchase_count)
    
    @app.route('/admin/feedback')
    @admin_required
    def admin_feedback():
        """View user feedback on AI analysis accuracy"""
        # Get filter parameters
        vote_filter = request.args.get('vote', 'all')  # 'all', 'up', 'down'
        sort_by = request.args.get('sort', 'recent')  # 'recent', 'oldest'
        
        # Build query
        query = Feedback.query
        
        if vote_filter != 'all':
            query = query.filter_by(vote=vote_filter)
        
        # Sort
        if sort_by == 'recent':
            query = query.order_by(Feedback.created_at.desc())
        else:
            query = query.order_by(Feedback.created_at.asc())
        
        feedback_list = query.all()
        
        # Calculate statistics
        total_feedback = Feedback.query.count()
        thumbs_up = Feedback.query.filter_by(vote='up').count()
        thumbs_down = Feedback.query.filter_by(vote='down').count()
        satisfaction_rate = (thumbs_up / total_feedback * 100) if total_feedback > 0 else 0
        
        # Get feedback with improvement notes
        feedback_with_notes = Feedback.query.filter(
            Feedback.vote == 'down',
            Feedback.improvement_note.isnot(None),
            Feedback.improvement_note != ''
        ).count()
        
        return render_template('admin_feedback.html',
                             active_tab='feedback',
                             feedback_list=feedback_list,
                             total_feedback=total_feedback,
                             thumbs_up=thumbs_up,
                             thumbs_down=thumbs_down,
                             satisfaction_rate=satisfaction_rate,
                             feedback_with_notes=feedback_with_notes,
                             vote_filter=vote_filter,
                             sort_by=sort_by)
    
    
    @app.route('/admin/users/<int:user_id>/adjust-balance', methods=['POST'])
    @admin_required
    def admin_adjust_balance(user_id):
        """Adjust user balance with support for credits, debits, refunds, and corrections"""
        user = User.query.get_or_404(user_id)
        
        transaction_type = request.form.get('transaction_type')
        amount = Decimal(str(request.form.get('amount', 0)))
        reason = request.form.get('reason', 'Manual adjustment by admin')
        
        if amount <= 0:
            flash('❌ Amount must be positive', 'danger')
            return redirect(url_for('admin_user_detail', user_id=user_id))
        
        # Calculate bonus total for refund validation
        bonus_total = Decimal('0')
        for txn in Transaction.query.filter_by(user_id=user_id).all():
            if any(keyword in txn.description.lower() for keyword in ['sign-up bonus', 'volume bonus', 'promotional']):
                if txn.amount_usd > 0:
                    bonus_total += Decimal(str(txn.amount_usd))
        max_refundable = max(Decimal('0'), user.balance_usd - bonus_total)
        
        # Handle different transaction types
        if transaction_type == 'manual_credit':
            # Add funds (positive transaction)
            user.balance_usd += amount
            transaction = Transaction(
                user_id=user_id,
                amount_usd=amount,
                transaction_type='credit',
                description=f'Manual Credit - {reason}'
            )
            db.session.add(transaction)
            flash(f'✅ Added ${amount:.2f} to {user.email}', 'success')
            
        elif transaction_type == 'manual_debit':
            # Remove funds (negative transaction)
            if user.balance_usd < amount:
                flash(f'❌ Cannot debit ${amount:.2f} - user only has ${user.balance_usd:.2f}', 'danger')
                return redirect(url_for('admin_user_detail', user_id=user_id))
            
            user.balance_usd -= amount
            transaction = Transaction(
                user_id=user_id,
                amount_usd=-amount,  # Negative for debit
                transaction_type='debit',
                description=f'Manual Debit - {reason}'
            )
            db.session.add(transaction)
            flash(f'✅ Deducted ${amount:.2f} from {user.email}', 'success')
            
        elif transaction_type == 'refund':
            # Refund (cannot exceed refundable balance)
            if amount > max_refundable:
                flash(f'❌ Cannot refund ${amount:.2f} - maximum refundable balance is ${max_refundable:.2f} (excluding ${bonus_total:.2f} in bonuses)', 'danger')
                return redirect(url_for('admin_user_detail', user_id=user_id))
            
            if user.balance_usd < amount:
                flash(f'❌ Cannot refund ${amount:.2f} - user only has ${user.balance_usd:.2f}', 'danger')
                return redirect(url_for('admin_user_detail', user_id=user_id))
            
            user.balance_usd -= amount
            transaction = Transaction(
                user_id=user_id,
                amount_usd=-amount,  # Negative for refund
                transaction_type='debit',
                description=f'Refund - {reason}'
            )
            db.session.add(transaction)
            flash(f'✅ Refunded ${amount:.2f} to {user.email}', 'success')
            
        elif transaction_type == 'bonus_correction':
            # Bonus correction (can be positive or negative based on context)
            # For now, treat as positive adjustment to promotional balance
            user.balance_usd += amount
            transaction = Transaction(
                user_id=user_id,
                amount_usd=amount,
                transaction_type='credit',
                description=f'Bonus Correction - {reason}'
            )
            db.session.add(transaction)
            flash(f'✅ Applied bonus correction of ${amount:.2f} to {user.email}', 'success')
        
        else:
            flash('❌ Invalid transaction type', 'danger')
            return redirect(url_for('admin_user_detail', user_id=user_id))
        
        db.session.commit()
        return redirect(url_for('admin_user_detail', user_id=user_id))
    
    
    @app.route('/admin/users/<int:user_id>/reset-password', methods=['POST'])
    @admin_required
    def admin_reset_password(user_id):
        """Reset user password (generate temporary password)"""
        user = User.query.get_or_404(user_id)
        
        # Generate temporary password
        import secrets
        temp_password = secrets.token_urlsafe(12)
        user.set_password(temp_password)
        db.session.commit()
        
        flash(f'✅ Password reset for {user.email}. Temporary password: {temp_password}', 'admin')
        return redirect(url_for('admin_user_detail', user_id=user_id))
    
    @app.route('/api/analysis-progress/<int:analysis_id>')
    @login_required
    def get_analysis_progress(analysis_id):
        """API endpoint to fetch real-time analysis progress"""
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Check if analysis is complete (has coverage_data populated)
        is_complete = bool(analysis.coverage_data)
        
        return jsonify({
            'resumes_processed': analysis.resumes_processed,
            'total_candidates': analysis.num_candidates,
            'is_complete': is_complete
        })
    
    @app.route('/api/analysis-progress/latest')
    @login_required
    def get_latest_analysis_progress():
        """API endpoint to fetch progress of the most recent in-progress analysis"""
        # Get the most recent analysis that's not yet complete (ordered by created_at DESC)
        analysis = Analysis.query.filter_by(
            user_id=current_user.id
        ).order_by(Analysis.created_at.desc()).first()
        
        if not analysis:
            return jsonify({'error': 'No analysis found'}), 404
        
        # Check if analysis is complete (has coverage_data populated)
        is_complete = bool(analysis.coverage_data)
        
        # Determine current phase (using analysis_size field temporarily as phase indicator)
        phase = analysis.analysis_size if analysis.analysis_size in ['phase1', 'phase2', 'complete'] else 'phase1'
        
        return jsonify({
            'analysis_id': analysis.id,
            'resumes_processed': analysis.resumes_processed,
            'total_candidates': analysis.num_candidates,
            'is_complete': is_complete,
            'phase': phase
        })
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
