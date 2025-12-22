from flask import Flask, render_template, redirect, url_for, flash, request, session
from flask_login import LoginManager, login_required, current_user
from config import config
from database import db, init_db, User, Transaction, Analysis, Draft, DraftResume, CandidateFile, UserSettings, Feedback
import os
import re
from datetime import datetime, timezone, timedelta
import json
import hashlib
from werkzeug.utils import secure_filename
from decimal import Decimal
import io
from dotenv import load_dotenv

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
        return User.query.get(int(user_id))
    
    # Add template filters
    @app.template_filter('to_local_time')
    def to_local_time(dt):
        """Convert UTC datetime to ISO format for JavaScript conversion"""
        if dt:
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        return ''
    
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
        return render_template('landing.html')
    
    @app.route('/pricing')
    def pricing():
        """Pricing page"""
        return render_template('pricing.html')
    
    @app.route('/security')
    def security():
        """Security and data privacy page"""
        return render_template('security.html')
    
    @app.route('/features')
    def features():
        """Features deep dive page"""
        return render_template('features.html')
    
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
        
        return render_template('dashboard.html', user=current_user, recent_analyses=recent_analyses, draft=draft)
    
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
                return render_template('analyze.html', user=current_user, jd_data=None, criteria_count=0, current_step='jd',
                                     in_workflow=False, has_unsaved_work=False, analysis_completed=False)
            
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
                    
                return render_template('analyze.html', user=current_user, jd_data=jd_data, 
                                     criteria_count=len(criteria_data), current_step=step,
                                     latest_analysis_id=current_draft_analysis_id,
                                     uploaded_resumes=uploaded_resumes,
                                     in_workflow=True, has_unsaved_work=has_unsaved_work,
                                     analysis_completed=analysis_completed,
                                     draft_modified_after_analysis=draft_modified_after_analysis)
            else:
                return render_template('analyze.html', user=current_user, jd_data=None, 
                                     criteria_count=0, current_step='jd', latest_analysis_id=None,
                                     uploaded_resumes=[], in_workflow=False, has_unsaved_work=False,
                                     analysis_completed=False)
        
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
                
                # Extract criteria (no limit - let user uncheck unwanted ones)
                print(f"DEBUG: Extracting JD sections from text (length: {len(jd_text_content)} chars)")
                sections = extract_jd_sections_with_gpt(jd_text_content)
                print(f"DEBUG: Sections extracted: {sections}")
                criteria, cat_map = build_criteria_from_sections(sections, per_section=999, cap_total=10000)
                print(f"DEBUG: Criteria extracted: {len(criteria)} items")
                
                if not criteria:
                    flash('Could not extract criteria from job description', 'error')
                    return redirect(url_for('analyze'))
                
                # Extract job title from JD
                jd_lines = [line.strip() for line in jd_text_content.split('\n') if line.strip()]
                job_title = "Position Not Specified"
                for line in jd_lines[:5]:
                    if len(line) < 10 or 'http' in line.lower() or any(char.isdigit() for char in line[:4]):
                        continue
                    if any(indicator in line.lower() for indicator in ['position:', 'role:', 'job title:', 'title:']):
                        job_title = line.split(':', 1)[1].strip() if ':' in line else line
                        break
                    elif len(line) > 15 and len(line) < 100:
                        job_title = line
                        break
                
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
                
                # Process and store resumes
                resumes_added = 0
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
                    candidate_name = infer_candidate_name(resume_file.filename, resume_text)
                    
                    draft_resume = DraftResume(
                        draft_id=draft.id,
                        file_name=resume_file.filename,
                        file_bytes=resume_bytes,
                        extracted_text=resume_text,
                        candidate_name=candidate_name,
                        file_hash=resume_hash
                    )
                    db.session.add(draft_resume)
                    resumes_added += 1
                
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
        
        elif action == 'run_analysis':
            # Legacy handler - redirect to proper flow
            flash('Please upload resumes first', 'info')
            return redirect(url_for('analyze', step='resumes'))
        
        elif action == 'run_analysis':
            # Step 2: Run full analysis with resumes
            try:
                from analysis import (
                    read_file_bytes, hash_bytes, Candidate,
                    analyse_candidates, infer_candidate_name
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
                
                # Run analysis
                coverage, insights, evidence_map = analyse_candidates(
                    candidates=candidates,
                    criteria=criteria,
                    weights=None,
                    chunk_chars=1200,
                    overlap=150
                )
                
                # Calculate cost
                from config import Config
                total_cost = Config.BASE_ANALYSIS_PRICE
                
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
                
                num_extra_insights = max(0, num_insights - 3)
                total_cost += num_extra_insights * Config.EXTRA_INSIGHT_PRICE
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
                
                # Generate GPT insights for selected candidates
                from analysis import gpt_candidate_insights
                insights_data = {}
                gpt_candidates_list = []  # Track which candidates had GPT insights
                
                if num_insights > 0:
                    # Get top N candidates by overall score
                    top_candidates = coverage.nlargest(num_insights, 'Overall')
                    
                    for idx, row in top_candidates.iterrows():
                        candidate_name = row['Candidate']
                        gpt_candidates_list.append(candidate_name)  # Track for disclaimers
                        
                        # Find the candidate object
                        candidate_obj = next((c for c in candidates if c.name == candidate_name), None)
                        if candidate_obj:
                            # Get scores for this candidate
                            candidate_scores = {col: row[col] for col in coverage.columns if col not in ['Candidate', 'Overall']}
                            
                            # Generate insights with evidence snippets
                            insights = gpt_candidate_insights(
                                candidate_name=candidate_name,
                                candidate_text=candidate_obj.text,
                                jd_text=jd_text,
                                coverage_scores=candidate_scores,
                                criteria=criteria,
                                evidence_map=evidence_map,
                                model="gpt-4o"
                            )
                            insights_data[candidate_name] = insights
                
                # Build category map from criteria_list (not criteria which is just strings)
                category_map = {c['criterion']: c.get('category', 'Other Requirements') for c in criteria_list if c.get('use', True)}
                
                analysis = Analysis(
                    user_id=current_user.id,
                    job_title=job_title,
                    job_description_text=jd_text[:5000],
                    num_candidates=len(candidates),
                    num_criteria=len(criteria),
                    coverage_data=coverage.to_json(orient='records'),
                    insights_data=json.dumps(insights_data),
                    evidence_data=json.dumps({f"{k[0]}|||{k[1]}": v for k, v in evidence_map.items()}),
                    criteria_list=json.dumps(criteria),
                    category_map=json.dumps(category_map),
                    gpt_candidates=json.dumps(gpt_candidates_list),
                    cost_usd=estimated_cost,
                    analysis_size='small' if len(candidates) <= 5 else ('medium' if len(candidates) <= 15 else 'large')
                )
                db.session.add(analysis)
                
                # Flush to get analysis.id before linking to transaction
                db.session.flush()
                
                # Link transaction to analysis
                transaction.analysis_id = analysis.id
                
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
        
        # Rebuild criteria_data from analysis
        criteria_list = json.loads(analysis.criteria_list)
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
        if request.method == 'GET':
            # Check we have draft with JD, criteria, and resumes
            draft = Draft.query.filter_by(user_id=current_user.id).first()
            if not draft or not draft.criteria_data:
                flash('Please upload a JD and review criteria first', 'error')
                return redirect(url_for('analyze'))
            
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
            
            return render_template('run_analysis.html', 
                                 user=current_user,
                                 resume_count=resume_count,
                                 latest_analysis_id=current_draft_analysis_id,
                                 in_workflow=True, has_unsaved_work=True,
                                 analysis_completed=analysis_completed,
                                 draft_modified_after_analysis=draft_modified_after_analysis)
        
        # POST - Run the analysis
        try:
            from analysis import Candidate, analyse_candidates
            from config import Config
            
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
            
            num_extra_insights = max(0, num_insights - 3)
            estimated_cost = Config.BASE_ANALYSIS_PRICE + (num_extra_insights * Config.EXTRA_INSIGHT_PRICE)
            estimated_cost = Decimal(str(estimated_cost))
            
            # Check funds BEFORE running analysis
            if current_user.balance_usd < estimated_cost:
                # Don't redirect - show modal to add funds and continue
                resume_count = len(candidates)
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
                                     selected_insights_mode=insights_mode)
            
            # Run analysis
            coverage, insights, evidence_map = analyse_candidates(
                candidates=candidates,
                criteria=criteria,
                weights=None,
                chunk_chars=1200,
                overlap=150
            )
            
            # Deduct funds (already checked above)
            current_user.balance_usd -= estimated_cost
            
            # Create transaction
            transaction = Transaction(
                user_id=current_user.id,
                amount_usd=-estimated_cost,
                transaction_type='debit',
                description=f'Analysis: {len(candidates)} candidates, {len(criteria)} criteria'
            )
            db.session.add(transaction)
            
            # Generate GPT insights
            from analysis import gpt_candidate_insights
            insights_data = {}
            gpt_candidates_list = []
            
            if num_insights > 0:
                top_candidates = coverage.nlargest(num_insights, 'Overall')
                for idx, row in top_candidates.iterrows():
                    candidate_name = row['Candidate']
                    gpt_candidates_list.append(candidate_name)
                    candidate_obj = next((c for c in candidates if c.name == candidate_name), None)
                    if candidate_obj:
                        candidate_scores = {col: row[col] for col in coverage.columns if col not in ['Candidate', 'Overall']}
                        insights = gpt_candidate_insights(
                            candidate_name=candidate_name,
                            candidate_text=candidate_obj.text,
                            jd_text=jd_text,
                            coverage_scores=candidate_scores,
                            criteria=criteria,
                            evidence_map=evidence_map,
                            model="gpt-4o"
                        )
                        insights_data[candidate_name] = insights
            
            # Build category map
            category_map = {c['criterion']: c.get('category', 'Other Requirements') for c in criteria_list if c.get('use', True)}
            
            # Use job title from draft (already extracted/edited by user)
            job_title = draft.job_title or "Position Not Specified"
            
            # Create analysis record
            analysis = Analysis(
                user_id=current_user.id,
                job_title=job_title,
                job_description_text=jd_text[:5000],  # Truncated for display
                jd_full_text=jd_text,  # Full text for recreating drafts
                jd_filename=draft.jd_filename,  # Original filename
                num_candidates=len(candidates),
                num_criteria=len(criteria),
                coverage_data=coverage.to_json(orient='records'),
                insights_data=json.dumps(insights_data),
                evidence_data=json.dumps({f"{k[0]}|||{k[1]}": v for k, v in evidence_map.items()}),
                criteria_list=json.dumps(criteria),
                category_map=json.dumps(category_map),
                gpt_candidates=json.dumps(gpt_candidates_list),
                cost_usd=estimated_cost,
                analysis_size='small' if len(candidates) <= 5 else ('medium' if len(candidates) <= 15 else 'large')
            )
            db.session.add(analysis)
            db.session.flush()
            
            # Link transaction to analysis
            transaction.analysis_id = analysis.id
            
            # Update user analytics tracking
            current_user.total_analyses_count += 1
            current_user.total_revenue_usd += total_cost
            
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
            
            db.session.commit()
            
            flash(f'✅ Analysis complete! Cost: ${total_cost:.2f}. Remaining balance: ${current_user.balance_usd:.2f}', 'success')
            return redirect(url_for('results', analysis_id=analysis.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Analysis failed: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
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
        
        # Parse evidence_map (keep as string keys for JSON serialization in template)
        evidence_map = json.loads(analysis.evidence_data)
        
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
                             is_current_draft_analysis=is_current_draft_analysis)
    
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
        coverage = pd.read_json(analysis.coverage_data, orient='records')
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
                    jd_filename=analysis.job_title
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
                    lo=0.35
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
                                 has_pdf=draft.jd_filename.lower().endswith('.pdf') if draft.jd_filename else False,
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
            flash('No JD file available', 'error')
            return redirect(url_for('review_criteria'))
        
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
        has_gpt_insights = bool(candidate_insights.get('top') or candidate_insights.get('gaps') or candidate_insights.get('notes'))
        
        # Group scores by category
        cat_map = json.loads(analysis.category_map) if analysis.category_map else {}
        scores_by_category = {}
        for criterion in criteria_cols:
            category = cat_map.get(criterion, 'Other Requirements')
            score = current_candidate_data['data'][criterion]
            
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
                'rating': rating
            })
        
        # Get evidence snippets
        evidence_data = json.loads(analysis.evidence_data) if analysis.evidence_data else {}
        evidence_list = []
        for criterion in criteria_cols:
            key = f"{selected_candidate}|||{criterion}"
            if key in evidence_data:
                snippet = evidence_data[key]
                score = current_candidate_data['data'][criterion]
                
                if score >= hi_threshold:
                    color = '#28a745'
                elif score >= lo_threshold:
                    color = '#ffc107'
                else:
                    color = '#dc3545'
                
                evidence_list.append({
                    'criterion': criterion,
                    'snippet': snippet[:500],  # Limit snippet length
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
        
        return render_template('insights.html',
                             analysis_id=analysis_id,
                             candidates=candidates_list,
                             current_candidate=current_candidate,
                             current_idx=current_idx,
                             has_insights=True,
                             in_workflow=is_current_draft_analysis,
                             has_unsaved_work=False,
                             analysis_completed=True,
                             is_current_draft_analysis=is_current_draft_analysis)
    
    @app.route('/download-candidate-pdf/<int:analysis_id>/<candidate_name>')
    @login_required
    def download_candidate_pdf(analysis_id, candidate_name):
        """Download individual candidate report as PDF"""
        from flask import send_file
        from export_candidate import to_individual_candidate_pdf
        
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
        
        # Convert evidence_map keys from "candidate|||criterion" strings back to tuples
        evidence_map = {}
        for k, v in evidence_map_raw.items():
            parts = k.split('|||', 1)
            if len(parts) == 2:
                evidence_map[(parts[0], parts[1])] = v
        
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
            include_evidence=True,  # Include evidence by default for individual downloads
            job_title=analysis.job_title,
            gpt_candidates=gpt_candidates_raw
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
        transactions = current_user.transactions.order_by(
            db.desc('created_at')
        ).limit(50).all()
        
        # Check which analysis IDs still exist (hard delete means missing = deleted)
        analysis_ids = [t.analysis_id for t in transactions if t.analysis_id]
        existing_analyses = set()
        if analysis_ids:
            all_analyses = Analysis.query.filter(Analysis.id.in_(analysis_ids)).all()
            existing_analyses = {a.id for a in all_analyses}
        
        # Calculate running balance for each transaction (newest first)
        running_balance = float(current_user.balance_usd)
        balances = []
        for txn in transactions:
            balances.append(running_balance)
            # Move backwards in time: subtract this transaction to get previous balance
            running_balance -= float(txn.amount_usd)
        
        return render_template('account.html', 
                             user=current_user,
                             transactions=transactions,
                             balances=balances,
                             existing_analyses=existing_analyses)
    
    @app.route('/job-history')
    @login_required
    def job_history():
        """Job analysis history"""
        analyses = current_user.analyses.order_by(
            db.desc('created_at')
        ).limit(50).all()
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
        coverage_df = pd.read_json(analysis.coverage_data, orient='records')
        
        return render_template('export.html', 
                             analysis=analysis, 
                             coverage=coverage_df,
                             user_settings=user_settings,
                             is_current_draft_analysis=is_current_draft_analysis)
    
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
        coverage_df = pd.read_json(analysis.coverage_data, orient='records')
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
            jd_filename=analysis.job_title
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
        coverage_df = pd.read_json(analysis.coverage_data, orient='records')
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
            jd_filename=analysis.job_title
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
        from export_candidate import to_executive_summary_docx
        
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
        # Parse data
        import pandas as pd
        coverage_df = pd.read_json(analysis.coverage_data, orient='records')
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
        
        # Generate Word doc
        docx_bytes = to_executive_summary_docx(
            coverage=coverage_df,
            insights=insights,
            jd_text=analysis.job_description_text,
            cat_map=category_map,
            hi=0.75,
            lo=0.35,
            jd_filename=analysis.job_title
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
        coverage_df = pd.read_json(analysis.coverage_data, orient='records')
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
            lo=0.35
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
        coverage_df = pd.read_json(analysis.coverage_data, orient='records')
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        coverage_df.to_csv(csv_buffer, index=False)
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
        include_evidence = data.get('include_evidence', False)
        
        if not candidates:
            return {'error': 'No candidates selected'}, 400
        
        # Parse data
        import pandas as pd
        coverage_df = pd.read_json(analysis.coverage_data, orient='records')
        insights = json.loads(analysis.insights_data)
        evidence_map_raw = json.loads(analysis.evidence_data)
        criteria = json.loads(analysis.criteria_list)
        gpt_candidates_raw = json.loads(analysis.gpt_candidates) if analysis.gpt_candidates else []
        
        # Convert evidence_map keys from "candidate|||criterion" strings back to tuples
        evidence_map = {}
        for k, v in evidence_map_raw.items():
            parts = k.split('|||', 1)  # Split only on first |||
            if len(parts) == 2:
                evidence_map[(parts[0], parts[1])] = v
        
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
                include_evidence=include_evidence,
                job_title=analysis.job_title,
                gpt_candidates=gpt_candidates_raw
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
        
        if not candidates:
            return {'error': 'No candidates selected'}, 400
        
        # Parse data
        import pandas as pd
        coverage_df = pd.read_json(analysis.coverage_data, orient='records')
        insights = json.loads(analysis.insights_data)
        criteria = json.loads(analysis.criteria_list)
        gpt_candidates_raw = json.loads(analysis.gpt_candidates) if analysis.gpt_candidates else []
        
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
                cat_map=category_map,
                hi=0.75,
                lo=0.35,
                job_title=analysis.job_title,
                gpt_candidates=gpt_candidates_raw
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
        coverage_df = pd.read_json(analysis.coverage_data, orient='records')
        
        # Create simple CSV with just Candidate and Overall
        simple_df = coverage_df[['Candidate', 'Overall']].copy()
        simple_df.columns = ['Candidate Name', 'Overall Score']
        
        # Convert to CSV
        csv_buffer = io.StringIO()
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
        
        # Convert to CSV
        csv_buffer = io.StringIO()
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
    
    @app.route('/admin/login', methods=['GET', 'POST'])
    def admin_login():
        """Simple password protection for admin panel"""
        if request.method == 'POST':
            password = request.form.get('password')
            # Use environment variable or default password (CHANGE THIS!)
            admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')
            
            if password == admin_password:
                session['admin_logged_in'] = True
                flash('✅ Admin access granted', 'success')
                return redirect(url_for('admin_settings'))
            else:
                flash('❌ Invalid password', 'danger')
        
        return render_template('admin_login.html')
    
    
    @app.route('/admin/logout')
    def admin_logout():
        """Logout from admin panel"""
        session.pop('admin_logged_in', None)
        flash('Logged out from admin panel', 'info')
        return redirect(url_for('admin_login'))
    
    
    def admin_required(f):
        """Decorator to protect admin routes"""
        from functools import wraps
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('admin_logged_in'):
                flash('⚠️ Admin access required', 'warning')
                return redirect(url_for('admin_login'))
            return f(*args, **kwargs)
        return decorated_function
    
    
    @app.route('/admin')
    @admin_required
    def admin_settings():
        """Display admin settings panel - redirect to GPT settings tab"""
        return redirect(url_for('admin_gpt_settings'))
    
    
    @app.route('/admin/gpt')
    @admin_required
    def admin_gpt_settings():
        """Display GPT settings tab"""
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt_settings.json')
        
        with open(settings_path, 'r') as f:
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
        """Save updated settings"""
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt_settings.json')
        
        # Load current settings
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        # Update values from form
        settings['gpt_model']['value'] = request.form.get('gpt_model')
        settings['temperature']['value'] = float(request.form.get('temperature'))
        settings['max_tokens']['value'] = int(request.form.get('max_tokens'))
        settings['evidence_snippet_chars']['value'] = int(request.form.get('evidence_snippet_chars'))
        settings['candidate_text_chars']['value'] = int(request.form.get('candidate_text_chars'))
        settings['jd_text_chars']['value'] = int(request.form.get('jd_text_chars'))
        settings['score_thresholds']['high_threshold']['value'] = float(request.form.get('high_threshold'))
        settings['score_thresholds']['low_threshold']['value'] = float(request.form.get('low_threshold'))
        settings['advanced_settings']['top_evidence_items']['value'] = int(request.form.get('top_evidence_items'))
        
        # Update metadata
        settings['_metadata']['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Save back to file
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        flash('✅ Settings saved successfully! Changes take effect immediately.', 'success')
        return redirect(url_for('admin_gpt_settings'))
    
    
    @app.route('/admin/prompts')
    @admin_required
    def admin_prompts():
        """AI Prompts management panel"""
        prompts_path = os.path.join(os.path.dirname(__file__), 'config', 'prompts.json')
        
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        
        message = request.args.get('message')
        return render_template('admin_prompts.html', 
                             prompts=prompts, 
                             message=message,
                             active_tab='prompts')
    
    
    @app.route('/admin/prompts/save', methods=['POST'])
    @admin_required
    def admin_prompts_save():
        """Save updated prompts"""
        prompts_path = os.path.join(os.path.dirname(__file__), 'config', 'prompts.json')
        
        # Load current prompts
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        
        # Update JD Extraction prompts
        prompts['jd_extraction']['system_prompt']['value'] = request.form.get('jd_extraction_system')
        prompts['jd_extraction']['user_prompt_template']['value'] = request.form.get('jd_extraction_user')
        
        # Update Candidate Insights prompts
        prompts['candidate_insights']['system_prompt']['value'] = request.form.get('candidate_insights_system')
        prompts['candidate_insights']['user_prompt_template']['value'] = request.form.get('candidate_insights_user')
        
        # Update metadata
        prompts['_metadata']['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        prompts['_metadata']['updated_by'] = 'admin'
        
        # Save back to file
        with open(prompts_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        flash('✅ Prompts saved successfully! Changes take effect immediately.', 'success')
        return redirect(url_for('admin_prompts'))
    
    
    @app.route('/admin/users')
    @admin_required
    def admin_users():
        """User management panel - list all users"""
        search = request.args.get('search', '').strip()
        sort_by = request.args.get('sort', 'created_at')
        
        query = User.query
        
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
        
        return render_template('admin_users.html', 
                             users=users, 
                             search=search,
                             sort_by=sort_by,
                             active_tab='users',
                             total_users=total_users,
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
        
        return render_template('admin_user_detail.html',
                             user=user,
                             analyses=analyses,
                             transactions=transactions,
                             active_tab='users')
    
    
    @app.route('/admin/system')
    @admin_required
    def admin_system():
        """System controls and settings"""
        settings_path = os.path.join(os.path.dirname(__file__), 'config', 'system_settings.json')
        
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        message = request.args.get('message')
        return render_template('admin_system.html', 
                             settings=settings, 
                             message=message,
                             active_tab='system')
    
    
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
        settings['new_user_welcome_credit']['value'] = float(request.form.get('new_user_welcome_credit', 0))
        
        # Update metadata
        settings['_metadata']['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        settings['_metadata']['updated_by'] = 'admin'
        
        # Save back to file
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        flash('✅ System settings saved successfully!', 'success')
        return redirect(url_for('admin_system'))
    
    
    @app.route('/admin/stats')
    @admin_required
    def admin_stats():
        """Usage statistics and analytics"""
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
                             recent_analyses=recent_analyses)
    
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
    
    
    @app.route('/admin/users/<int:user_id>/add-funds', methods=['POST'])
    @admin_required
    def admin_add_funds(user_id):
        """Manually add funds to a user account"""
        user = User.query.get_or_404(user_id)
        amount = float(request.form.get('amount', 0))
        reason = request.form.get('reason', 'Manual credit adjustment by admin')
        
        if amount > 0:
            user.add_funds(amount, reason)
            db.session.commit()
            flash(f'✅ Added ${amount:.2f} to {user.email}', 'success')
        else:
            flash('❌ Amount must be positive', 'danger')
        
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
        
        flash(f'✅ Password reset for {user.email}. Temporary password: {temp_password}', 'success')
        return redirect(url_for('admin_user_detail', user_id=user_id))
    
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
