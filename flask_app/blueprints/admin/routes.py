from flask import render_template, redirect, url_for, flash, request, jsonify, session, send_file, send_from_directory
from flask_login import login_required, current_user, logout_user
from database import db, User, Transaction, Analysis, Feedback, AdminLoginAttempt, AdminAuditLog, Draft
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from collections import defaultdict
import json
import os
import pyotp
import qrcode
from io import BytesIO
import base64

from blueprints.admin import admin_bp

# ============================================================================
# Admin Security Helper Functions
# ============================================================================

def get_client_ip():
    """Get client IP address (handles proxies/load balancers)"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr

def check_brute_force(ip_address):
    """Check if IP is locked out due to failed attempts"""
    cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=15)
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
        attempted_at=datetime.now(timezone.utc),
        success=success
    )
    db.session.add(attempt)
    db.session.commit()

def clear_login_attempts(ip_address):
    """Clear failed login attempts after successful login"""
    cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=15)
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

# ============================================================================
# Admin Routes
# ============================================================================

@admin_bp.route('/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login with brute-force protection and optional 2FA"""
    client_ip = get_client_ip()
    
    # Check for brute-force lockout
    if check_brute_force(client_ip):
        flash('🚫 Too many failed attempts. Please try again in 15 minutes.', 'danger')
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
                flash('🔐 2FA code required', 'warning')
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
        return redirect(url_for('admin.admin_settings'))
    
    # Check if 2FA is enabled
    totp_enabled = bool(os.environ.get('ADMIN_TOTP_SECRET'))
    return render_template('admin_login.html', totp_enabled=totp_enabled)


@admin_bp.route('/logout')
def admin_logout():
    """Logout from admin panel"""
    log_admin_action('logout')
    session.pop('admin_logged_in', None)
    session.pop('admin_last_activity', None)
    flash('Logged out from admin panel', 'info')
    return redirect(url_for('admin.admin_login'))


def admin_required(f):
    """Decorator to protect admin routes with 30-minute inactivity timeout"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('��� Admin access required', 'warning')
            return redirect(url_for('admin.admin_login'))
        
        # Check for inactivity timeout (30 minutes)
        last_activity_str = session.get('admin_last_activity')
        if last_activity_str:
            last_activity = datetime.fromisoformat(last_activity_str)
            inactivity = datetime.now(timezone.utc) - last_activity
            
            if inactivity > timedelta(minutes=30):
                session.pop('admin_logged_in', None)
                session.pop('admin_last_activity', None)
                flash('Ŧ� Session expired due to inactivity. Please login again.', 'warning')
                return redirect(url_for('admin.admin_login'))
        
        # Update last activity timestamp
        session['admin_last_activity'] = datetime.now(timezone.utc).isoformat()
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/setup-2fa')
def admin_setup_2fa():
    """enerate QR code for 2FA setup (only accessible without 2FA enabled)"""
    # Only allow this if 2FA is not yet configured
    if os.environ.get('ADMIN_TOTP_SECRET'):
        flash('��� 2FA is already configured. To reset, remove ADMIN_TOTP_SECRET from environment.', 'warning')
        return redirect(url_for('admin.admin_login'))
    
    # enerate new secret
    secret = pyotp.random_base32()
    
    # Create provisioning URI for QR code
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name='Admin',
        issuer_name='Candidate Evaluator'
    )
    
    # enerate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64 for display
    buffered = BytesIO()
    img.save(buffered, format="PN")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return render_template('admin_2fa_setup.html', 
                         secret=secret, 
                         qr_code=img_str,
                         provisioning_uri=provisioning_uri)


@admin_bp.route('/audit-logs')
@admin_required
def admin_audit_logs():
    """View admin audit logs"""
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    logs = AdminAuditLog.query.order_by(
        AdminAuditLog.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('admin_audit_logs.html', logs=logs, active_tab='audit')


@admin_bp.route('/business-health')
@admin_required
def admin_business_health():
    """Business Health Monitor and Calculator Settings"""
    settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'gpt_settings.json')
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    
    # Load pricing from single source of truth
    pricing_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'pricing_settings.json')
    with open(pricing_path, 'r', encoding='utf-8') as f:
        pricing = json.load(f)
    
    message = request.args.get('message')
    return render_template('admin_business_health.html', 
                         settings=settings, 
                         pricing=pricing,
                         active_tab='business_health',
                         message=message)


@admin_bp.route('/business-health/save', methods=['POST'])
@admin_required
def admin_save_business_health():
    """Save calculator settings"""
    settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'gpt_settings.json')
    
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
    
    return redirect(url_for('admin.admin_business_health', message='Calculator settings saved successfully!'))


@admin_bp.route('/pricing')
@admin_required
def admin_pricing():
    """Pricing & Revenue Configuration - Single Source of Truth"""
    pricing_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'pricing_settings.json')
    with open(pricing_path, 'r', encoding='utf-8') as f:
        pricing = json.load(f)
    
    message = request.args.get('message')
    return render_template('admin_pricing.html', 
                         pricing=pricing, 
                         active_tab='pricing',
                         message=message)


@admin_bp.route('/pricing/save', methods=['POST'])
@admin_required
def admin_save_pricing():
    """Save pricing configuration"""
    pricing_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'pricing_settings.json')
    
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
    
    return redirect(url_for('admin.admin_pricing', message='Pricing configuration saved successfully! Changes are now live across the platform.'))


@admin_bp.route('/admin')
@admin_required
def admin_settings():
    """Display admin settings panel - redirect to Users page"""
    return redirect(url_for('admin.admin_users'))


@admin_bp.route('/run-migrations/<secret>')
def admin_run_migrations(secret):
    """Emergency migration endpoint - run cost breakdown migrations via HTTP"""
    # Security: require secret key
    if secret != "migrate-cost-breakdown-2026":
        return "Unauthorized", 403
    
    from sqlalchemy import text
    results = []
    
    try:
        # Check existing columns first
        inspector = db.inspect(db.engine)
        existing_columns = [col['name'] for col in inspector.get_columns('analyses')]
        results.append(f"=== Current columns: {len(existing_columns)}<br><br>")
        
        # Define all columns that need to exist
        all_fields = [
            ("openai_cost_usd", "NUMERIC(10, 4)", "Actual OpenAI API costs"),
            ("ranker_cost_usd", "NUMERIC(10, 4)", "Cost for ranking calls"),
            ("insight_cost_usd", "NUMERIC(10, 4)", "Cost for insight generation"),
            ("retry_count", "INTEGER DEFAULT 0", "Number of API retries"),
            ("json_fallback_count", "INTEGER DEFAULT 0", "JSON parsing fallbacks"),
            ("api_calls_made", "INTEGER", "Total API calls"),
            ("avg_api_response_ms", "INTEGER", "Average response time"),
        ]
        
        added_count = 0
        skipped_count = 0
        
        for field_name, field_type, description in all_fields:
            if field_name in existing_columns:
                results.append(f"✓ Skipped: {field_name} (already exists)<br>")
                skipped_count += 1
            else:
                try:
                    sql = f"ALTER TABLE analyses ADD COLUMN {field_name} {field_type}"
                    db.session.execute(text(sql))
                    db.session.commit()
                    results.append(f"✅ Added: {field_name} - {description}<br>")
                    added_count += 1
                except Exception as e:
                    if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                        results.append(f"⚠️ Skipped: {field_name} (already exists)<br>")
                        skipped_count += 1
                    else:
                        results.append(f"❌ Error: {field_name}: {str(e)[:200]}<br>")
        
        results.append(f"<br><br>🎉 Migration Complete!<br>")
        results.append(f"✅ Added: {added_count} columns<br>")
        results.append(f"⏭️ Skipped: {skipped_count} columns<br>")
        return "".join(results), 200
        
    except Exception as e:
        results.append(f"<br><br>❌ CRITICAL ERROR: {str(e)}")
        return "".join(results), 500


@admin_bp.route('/health-monitor')
@admin_required
def admin_health_monitor():
    """Display system health monitoring dashboard with performance metrics"""
    from sqlalchemy import func, desc
    from datetime import datetime, timedelta
    
    # ============================================
    # SYSTEM HEALTH CALCULATION
    # ============================================
    
    # Define alert thresholds
    THRESHOLDS = {
        'retry_count': 5,           # Alert if >5 retries in one job
        'json_fallback_count': 3,   # Alert if >3 JSON failures  
        'avg_api_response_ms': 5000,# Alert if avg API response >5s
        'processing_time_per_resume': 5  # Alert if >5 seconds per resume
    }
    
    # Get recent jobs (last 50) for health analysis
    recent_analyses = Analysis.query.filter(
        Analysis.completed_at.isnot(None)
    ).order_by(desc(Analysis.completed_at)).limit(50).all()
    
    # Calculate health metrics
    total_recent = len(recent_analyses)
    healthy_count = 0
    warning_count = 0
    critical_count = 0
    alerts = []
    
    for analysis in recent_analyses:
        issues = []
        severity = 'healthy'
        
        # Check retry count
        if analysis.retry_count and analysis.retry_count > THRESHOLDS['retry_count']:
            issues.append(f"{analysis.retry_count} retries")
            severity = 'warning'
        
        # Check JSON fallbacks
        if analysis.json_fallback_count and analysis.json_fallback_count > THRESHOLDS['json_fallback_count']:
            issues.append(f"{analysis.json_fallback_count} JSON fallbacks")
            severity = 'critical' if severity != 'critical' else severity
        
        # Check API response time
        if analysis.avg_api_response_ms and analysis.avg_api_response_ms > THRESHOLDS['avg_api_response_ms']:
            issues.append(f"{analysis.avg_api_response_ms}ms avg API response")
            severity = 'warning' if severity == 'healthy' else severity
        
        # Check processing time per resume
        if analysis.processing_duration_seconds and analysis.num_candidates:
            time_per_resume = analysis.processing_duration_seconds / analysis.num_candidates
            if time_per_resume > THRESHOLDS['processing_time_per_resume']:
                issues.append(f"{time_per_resume:.1f}s per resume")
                severity = 'warning' if severity == 'healthy' else severity
        
        # Count by severity
        if severity == 'healthy':
            healthy_count += 1
        elif severity == 'warning':
            warning_count += 1
        else:
            critical_count += 1
        
        # Add to alerts if has issues
        if issues:
            alerts.append({
                'analysis_id': analysis.id,
                'job_title': analysis.job_title or 'Untitled',
                'num_candidates': analysis.num_candidates,
                'severity': severity,
                'issues': issues,
                'completed_at': analysis.completed_at,
                'openai_cost_usd': float(analysis.openai_cost_usd) if analysis.openai_cost_usd else 0.0,
                'retry_count': analysis.retry_count or 0,
                'json_fallback_count': analysis.json_fallback_count or 0,
                'avg_api_response_ms': analysis.avg_api_response_ms or 0,
                'processing_duration_seconds': analysis.processing_duration_seconds
            })
    
    # Calculate overall system health score (0-100)
    if total_recent > 0:
        health_score = int((healthy_count / total_recent) * 100)
        health_status = 'healthy' if health_score >= 80 else ('warning' if health_score >= 60 else 'critical')
    else:
        health_score = 100
        health_status = 'healthy'
    
    # ============================================
    # PERFORMANCE TRENDS
    # ============================================
    
    # Cost breakdown trends (last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    
    # Get all jobs with cost breakdown data
    recent_jobs_for_trend = Analysis.query.filter(
        Analysis.completed_at >= thirty_days_ago,
        Analysis.completed_at.isnot(None),
        Analysis.num_candidates > 0,
        Analysis.ranker_cost_usd.isnot(None),
        Analysis.insight_cost_usd.isnot(None)
    ).all()
    
    # Group by date for ranker cost per resume
    daily_ranker_costs = defaultdict(lambda: {'total_cost_per_resume': 0, 'count': 0})
    # Group by date for insight cost per job
    daily_insight_costs = defaultdict(lambda: {'total_cost': 0, 'count': 0})
    
    for job in recent_jobs_for_trend:
        date_key = job.completed_at.date().strftime('%Y-%m-%d')
        
        # Ranker cost per resume (this is the per-resume metric)
        ranker_cost_per_resume = float(job.ranker_cost_usd) / float(job.num_candidates)
        daily_ranker_costs[date_key]['total_cost_per_resume'] += ranker_cost_per_resume
        daily_ranker_costs[date_key]['count'] += 1
        
        # Insight cost per job (fixed cost regardless of batch size)
        daily_insight_costs[date_key]['total_cost'] += float(job.insight_cost_usd)
        daily_insight_costs[date_key]['count'] += 1
    
    # Ranker cost trend (per resume)
    cost_trend_labels = []
    cost_trend_data = []
    for date_str in sorted(daily_ranker_costs.keys()):
        cost_trend_labels.append(date_str[5:].replace('-', '/'))
        avg = daily_ranker_costs[date_str]['total_cost_per_resume'] / daily_ranker_costs[date_str]['count']
        cost_trend_data.append(round(avg, 4))
    
    # Insight cost trend (per job)
    insight_trend_labels = []
    insight_trend_data = []
    for date_str in sorted(daily_insight_costs.keys()):
        insight_trend_labels.append(date_str[5:].replace('-', '/'))
        avg = daily_insight_costs[date_str]['total_cost'] / daily_insight_costs[date_str]['count']
        insight_trend_data.append(round(avg, 4))
    
    # ============================================
    # KEY PERFORMANCE INDICATORS
    # ============================================
    
    # Average metrics across all completed jobs
    avg_cost_multiplier = db.session.query(func.avg(Analysis.cost_multiplier)).filter(
        Analysis.cost_multiplier.isnot(None)
    ).scalar() or 1.0
    
    avg_retry_count = db.session.query(func.avg(Analysis.retry_count)).filter(
        Analysis.retry_count.isnot(None)
    ).scalar() or 0
    
    avg_json_fallbacks = db.session.query(func.avg(Analysis.json_fallback_count)).filter(
        Analysis.json_fallback_count.isnot(None)
    ).scalar() or 0
    
    avg_api_response = db.session.query(func.avg(Analysis.avg_api_response_ms)).filter(
        Analysis.avg_api_response_ms.isnot(None)
    ).scalar() or 0
    
    # Job success rate
    total_jobs = Analysis.query.count()
    successful_jobs = Analysis.query.filter(Analysis.completed_at.isnot(None)).count()
    failed_jobs = Analysis.query.filter(Analysis.failed_at.isnot(None)).count()
    success_rate = (successful_jobs / total_jobs * 100) if total_jobs > 0 else 100
    
    # Recent jobs with full metrics for table
    recent_jobs_full = db.session.query(
        Analysis,
        User.email.label('user_email')
    ).join(
        User, Analysis.user_id == User.id
    ).filter(
        Analysis.completed_at.isnot(None)
    ).order_by(
        desc(Analysis.completed_at)
    ).limit(20).all()
    
    recent_jobs_data = []
    for analysis, user_email in recent_jobs_full:
        # Determine status color
        status = 'success'
        if (analysis.retry_count and analysis.retry_count > 5) or (analysis.json_fallback_count and analysis.json_fallback_count > 3):
            status = 'warning'
        
        recent_jobs_data.append({
            'id': analysis.id,
            'user_email': user_email,
            'job_title': analysis.job_title or 'Untitled',
            'num_candidates': analysis.num_candidates,
            'openai_cost_usd': float(analysis.openai_cost_usd) if analysis.openai_cost_usd else 0.0,
            'ranker_cost_usd': float(analysis.ranker_cost_usd) if analysis.ranker_cost_usd else 0.0,
            'insight_cost_usd': float(analysis.insight_cost_usd) if analysis.insight_cost_usd else 0.0,
            'retry_count': analysis.retry_count or 0,
            'json_fallback_count': analysis.json_fallback_count or 0,
            'api_calls_made': analysis.api_calls_made or 0,
            'avg_api_response_ms': analysis.avg_api_response_ms or 0,
            'processing_duration_seconds': analysis.processing_duration_seconds,
            'completed_at': analysis.completed_at,
            'status': status
        })
    
    # Package all stats
    stats = {
        # System Health
        'health_score': health_score,
        'health_status': health_status,
        'healthy_count': healthy_count,
        'warning_count': warning_count,
        'critical_count': critical_count,
        'total_recent': total_recent,
        'alerts': alerts[:10],  # Top 10 most recent alerts
        'thresholds': THRESHOLDS,
        
        # Performance Trends
        'cost_trend': {'labels': cost_trend_labels, 'data': cost_trend_data},
        'insight_trend': {'labels': insight_trend_labels, 'data': insight_trend_data},
        
        # KPIs
        'avg_retry_count': float(avg_retry_count),
        'avg_json_fallbacks': float(avg_json_fallbacks),
        'avg_api_response_ms': int(avg_api_response),
        'success_rate': float(success_rate),
        'total_jobs': total_jobs,
        'successful_jobs': successful_jobs,
        'failed_jobs': failed_jobs,
        
        # Recent Jobs
        'recent_jobs': recent_jobs_data
    }
    
    return render_template('admin_health_monitor.html', stats=stats, active_tab='health')


@admin_bp.route('/analytics')
@admin_required
def admin_analytics():
    """Display analytics dashboard with business metrics - revenue, users, job history"""
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
        'cost_usd': float(analysis.cost_usd) if analysis.cost_usd else 0.0,
        'processing_duration_seconds': analysis.processing_duration_seconds,
        'completed_at': analysis.completed_at,
        'created_at': analysis.created_at,
        'exceeded_resume_limit': analysis.exceeded_resume_limit,
        'user_chose_override': analysis.user_chose_override
    } for analysis, user_email in recent_jobs]
    
    stats = {
        'total_users': total_users,
        'total_jobs': total_jobs,
        'total_candidates': total_candidates,
        'total_revenue': total_revenue,
        'avg_candidates_per_job': avg_candidates,
        'median_candidates': median_candidates,
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
        'min_resume_chars': int(min_resume_chars),
        'max_resume_chars': int(max_resume_chars),
        'resume_size_labels': resume_size_labels,
        'resume_size_distribution': resume_size_distribution,
        'top_users': top_users_data,
        'recent_jobs': recent_jobs_data
    }
    
    return render_template('admin_analytics.html', stats=stats, active_tab='analytics')


@admin_bp.route('/gpt')
@admin_required
def admin_gpt_settings():
    """Display PT settings tab with two-agent configuration"""
    settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'gpt_settings.json')
    
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    
    message = request.args.get('message')
    return render_template('admin_gpt.html', settings=settings, message=message, active_tab='gpt')


@admin_bp.route('/migrate-db')
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
            <h1 style="color: #10b981;">�� Migration Successful!</h1>
            <p>Database schema updated:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>Added <code>analysis_deleted_at</code> to transactions table</li>
                <li>Removed <code>deleted_at</code> from analyses table</li>
            </ul>
            <p style="margin-top: 30px;">
                <a href="/admin/gpt" style="color: #2563eb;">�� Back to Admin Panel</a>
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
            <h1 style="color: #dc2626;">�� Migration Failed</h1>
            <p>Error: {str(e)}</p>
            <p style="margin-top: 30px;">
                <a href="/admin/gpt" style="color: #2563eb;">�� Back to Admin Panel</a>
            </p>
        </body>
        </html>
        """

@admin_bp.route('/migrate-jd-bytes')
@admin_required
def admin_migrate_jd_bytes():
    """Run database migration - add jd_bytes column to analyses table for storing PDF files"""
    try:
        from sqlalchemy import text
        
        # Add jd_bytes column (BYTEA for PostgreSQL, BLOB for SQLite)
        db.session.execute(text("ALTER TABLE analyses ADD COLUMN IF NOT EXISTS jd_bytes BYTEA;"))
        db.session.commit()
        
        return """
        <html>
        <head><title>Migration Complete</title></head>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1 style="color: #10b981;">�� Migration Successful!</h1>
            <p>Database schema updated:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>Added <code>jd_bytes</code> column to analyses table</li>
                <li>Future analyses will preserve PDF files in job history</li>
                <li>When loading from history, original PDFs will be available</li>
            </ul>
            <p style="margin-top: 20px; color: #6b7280; font-size: 14px;">
                Note: Existing job history records will not have PDF bytes (they will show text only).
            </p>
            <p style="margin-top: 30px;">
                <a href="/admin/gpt" style="color: #2563eb;">�� Back to Admin Panel</a>
            </p>
        </body>
        </html>
        """
    except Exception as e:
        # Check if error is just "column already exists"
        if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
            return """
            <html>
            <head><title>Migration Already Applied</title></head>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1 style="color: #f59e0b;">��� Migration Already Applied</h1>
                <p>The <code>jd_bytes</code> column already exists in the analyses table.</p>
                <p style="color: #10b981; margin-top: 20px;">�� No action needed - your database is up to date.</p>
                <p style="margin-top: 30px;">
                    <a href="/admin/gpt" style="color: #2563eb;">�� Back to Admin Panel</a>
                </p>
            </body>
            </html>
            """
        
        db.session.rollback()
        return f"""
        <html>
        <head><title>Migration Error</title></head>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1 style="color: #dc2626;">�� Migration Failed</h1>
            <p>Error: {str(e)}</p>
            <p style="margin-top: 30px;">
                <a href="/admin/gpt" style="color: #2563eb;">�� Back to Admin Panel</a>
            </p>
        </body>
        </html>
        """

@admin_bp.route('/save', methods=['POST'])
@admin_required
def admin_save_settings():
    """Save updated two-agent settings"""
    settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'gpt_settings.json')
    
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
    
    flash('�� Settings saved successfully! Two-agent configuration active.', 'admin')
    return redirect(url_for('admin.admin_gpt_settings'))


@admin_bp.route('/prompts')
@admin_required
def admin_prompts():
    """AI Prompts management panel"""
    prompts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'prompts.json')
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    message = request.args.get('message')
    return render_template('admin_prompts.html', 
                         prompts=prompts, 
                         message=message,
                         active_tab='prompts')


@admin_bp.route('/prompts/save', methods=['POST'])
@admin_required
def admin_save_prompts():
    """Save updated prompts for two-agent system"""
    prompts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'prompts.json')
    
    # Load current prompts
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    # Update RANKER prompts (nested structure with .value)
    if 'ranker_scoring' in prompts:
        if 'system_prompt' in prompts['ranker_scoring']:
            prompts['ranker_scoring']['system_prompt']['value'] = request.form.get('ranker_system_prompt', prompts['ranker_scoring']['system_prompt'].get('value', ''))
        if 'user_prompt_template' in prompts['ranker_scoring']:
            prompts['ranker_scoring']['user_prompt_template']['value'] = request.form.get('ranker_user_prompt', prompts['ranker_scoring']['user_prompt_template'].get('value', ''))
    
    # Update INSIHT prompts (nested structure with .value)
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
    
    return redirect(url_for('admin.admin_prompts') + '?message=Prompts saved successfully! Two-agent configuration active.')


@admin_bp.route('/users')
@admin_required
def admin_users():
    """User management panel - list all users"""
    search = request.args.get('search', '').strip()
    sort_by = request.args.get('sort', 'created_at')
    online_filter = request.args.get('online', '')  # Filter for online users
    
    query = User.query
    
    # Online filter (users active in last 5 minutes)
    if online_filter == 'yes':
        five_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
        query = query.filter(User.last_seen >= five_minutes_ago)
    elif online_filter == 'no':
        five_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
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
    five_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
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


@admin_bp.route('/users/<int:user_id>')
@admin_required
def admin_user_detail(user_id):
    """View detailed information about a specific user"""
    user = User.query.get_or_404(user_id)
    
    # et user's analyses
    analyses = Analysis.query.filter_by(user_id=user_id).order_by(Analysis.created_at.desc()).all()
    
    # et ALL user's transactions (no limit) ordered oldest first for balance calculation
    all_transactions = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.created_at.asc()).all()
    
    # Calculate running balance for each transaction
    from decimal import Decimal
    running_balance = Decimal('0')
    balances = []
    for txn in all_transactions:
        running_balance += Decimal(str(txn.amount_usd))
        balances.append(float(running_balance))
    
    # Reverse to show newest first in the UI
    transactions = list(reversed(all_transactions))
    balances = list(reversed(balances))
    
    # Calculate bonus/promotional funds (not refundable)
    bonus_total = Decimal('0')
    for txn in all_transactions:
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
                         balances=balances,
                         bonus_total=bonus_total,
                         max_refundable=max_refundable,
                         active_tab='users')


@admin_bp.route('/users/<int:user_id>/suspend', methods=['POST'])
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
    
    flash(f'�� User {user.email} has been suspended', 'admin')
    return redirect(url_for('admin.admin_user_detail', user_id=user_id))


@admin_bp.route('/users/<int:user_id>/unsuspend', methods=['POST'])
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
    
    flash(f'�� User {user.email} has been unsuspended', 'admin')
    return redirect(url_for('admin.admin_user_detail', user_id=user_id))


# ============================================================================
# BALANCE MONITORIN & RECONCILIATION
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
    # All visible transactions should be included
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
        
        admin_email = os.environ.get('ADMIN_EMAIL', 'contact@candidateevaluator.com')
        
        html_body = f"""
        <h2>=�ܿ Balance Mismatch Detected</h2>
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
        
        <p><a href="{url_for('admin.admin_balance_audit', _external=True)}">View Balance Audit Dashboard</a></p>
        
        <p style="color: #666; font-size: 12px;">This alert was generated automatically by the balance monitoring system.</p>
        """
        
        send_email(
            subject=f'=�ܿ Balance Mismatch Alert - User {user.email}',
            recipients=[admin_email],
            html_body=html_body
        )
        print(f"�� Balance mismatch alert sent to {admin_email}")
    except Exception as e:
        print(f"�� Failed to send balance mismatch alert: {e}")


@admin_bp.route('/balance-audit')
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


@admin_bp.route('/balance-adjustment/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_balance_adjustment(user_id):
    """Create manual balance adjustment transaction"""
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        from decimal import Decimal
        
        # et the current discrepancy
        is_valid, actual_balance, calculated_balance, discrepancy = check_user_balance_integrity(user_id)
        
        # Determine adjustment type
        adjustment_type = request.form.get('adjustment_type', 'silent')
        adjustment_amount = Decimal(str(request.form.get('adjustment_amount', 0)))
        reason = request.form.get('reason', 'Manual balance adjustment by admin')
        
        if abs(adjustment_amount) < Decimal('0.01'):
            flash('Adjustment amount must be at least $0.01', 'danger')
            return redirect(url_for('admin.admin_balance_adjustment', user_id=user_id))
        
        old_balance = user.balance_usd
        
        if adjustment_type == 'silent':
            # Type 1: Silent Adjustment - Only adjust Available Balance (no transaction created)
            # Transaction History is the source of truth, Available Balance is adjusted to match
            user.balance_usd += adjustment_amount
            db.session.commit()
            
            # Verify the adjustment fixed the issue
            is_valid_after, actual_after, calculated_after, discrepancy_after = check_user_balance_integrity(user_id)
            
            if is_valid_after:
                flash(f'�� Silent adjustment applied. Available Balance changed from ${float(old_balance):.2f} to ${float(user.balance_usd):.2f}', 'admin')
            else:
                flash(f'��� Silent adjustment applied from ${float(old_balance):.2f} to ${float(user.balance_usd):.2f}, but discrepancy remains: ${abs(float(discrepancy_after)):.2f}', 'warning')
            
            log_admin_action('balance_adjustment_silent', {
                'user_id': user_id,
                'user_email': user.email,
                'adjustment_amount': float(adjustment_amount),
                'reason': reason,
                'old_balance': float(old_balance),
                'new_balance': float(user.balance_usd)
            })
            
        else:  # adjustment_type == 'transaction'
            # Type 2: Transaction Adjustment - Create visible transaction (affects both balances)
            # This is like a normal credit/debit transaction
            adjustment_txn = Transaction(
                user_id=user_id,
                amount_usd=adjustment_amount,
                transaction_type='credit' if adjustment_amount > 0 else 'debit',
                description=reason
            )
            db.session.add(adjustment_txn)
            
            # Also update the user's balance
            user.balance_usd += adjustment_amount
            db.session.commit()
            
            # Verify balances are in sync
            is_valid_after, actual_after, calculated_after, discrepancy_after = check_user_balance_integrity(user_id)
            
            if is_valid_after:
                flash(f'�� Transaction adjustment created. ${float(adjustment_amount):.2f} {"added to" if adjustment_amount > 0 else "removed from"} account. New balance: ${float(user.balance_usd):.2f}', 'admin')
            else:
                flash(f'��� Transaction adjustment created but discrepancy detected: ${abs(float(discrepancy_after)):.2f}', 'warning')
            
            log_admin_action('balance_adjustment_transaction', {
                'user_id': user_id,
                'user_email': user.email,
                'adjustment_amount': float(adjustment_amount),
                'reason': reason,
                'old_balance': float(old_balance),
                'new_balance': float(user.balance_usd),
                'transaction_id': adjustment_txn.id
            })
        
        return redirect(url_for('admin.admin_user_detail', user_id=user_id))
    
    # ET request - show adjustment form
    is_valid, actual, calculated, discrepancy = check_user_balance_integrity(user_id)
    
    # Calculate bonus/promotional funds (not refundable)
    from decimal import Decimal
    bonus_total = Decimal('0')
    for txn in Transaction.query.filter_by(user_id=user_id).all():
        # Check if transaction is a bonus/promotional type
        if any(keyword in txn.description.lower() for keyword in ['sign-up bonus', 'volume bonus', 'promotional']):
            if txn.amount_usd > 0:  # Only count positive bonuses
                bonus_total += Decimal(str(txn.amount_usd))
    
    # Calculate maximum refundable balance
    max_refundable = max(Decimal('0'), user.balance_usd - bonus_total)
    
    return render_template('admin_balance_adjustment.html',
                         user=user,
                         actual_balance=actual,
                         calculated_balance=calculated,
                         discrepancy=discrepancy,
                         suggested_adjustment=-discrepancy,  # Negative of discrepancy to fix it
                         bonus_total=bonus_total,
                         max_refundable=max_refundable,
                         active_tab='balance_audit')


@admin_bp.route('/refund/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_refund(user_id):
    """Process user refund (dedicated refund page)"""
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        from decimal import Decimal
        
        refund_amount = Decimal(str(request.form.get('refund_amount', 0)))
        refund_description = request.form.get('refund_description', 'Refund processed by admin')
        
        # Validate refund amount is negative
        if refund_amount >= 0:
            flash('�� Refund amount must be negative (e.g., -25.00)', 'danger')
            return redirect(url_for('admin.admin_refund', user_id=user_id))
        
        # Calculate max refundable (excluding bonuses)
        bonus_total = Decimal('0')
        for txn in Transaction.query.filter_by(user_id=user_id).all():
            if txn.description and any(keyword in txn.description.lower() for keyword in ['sign-up bonus', 'volume bonus', 'promotional']):
                if txn.amount_usd > 0:
                    bonus_total += Decimal(str(txn.amount_usd))
        
        max_refundable = max(Decimal('0'), user.balance_usd - bonus_total)
        
        # Validate refund doesn't exceed max refundable
        if abs(refund_amount) > max_refundable:
            flash(f'�� Refund amount (${abs(float(refund_amount)):.2f}) exceeds maximum refundable balance (${float(max_refundable):.2f}). Promotional funds cannot be refunded.', 'danger')
            return redirect(url_for('admin.admin_refund', user_id=user_id))
        
        # Create refund transaction (visible to user)
        old_balance = user.balance_usd
        refund_txn = Transaction(
            user_id=user_id,
            amount_usd=refund_amount,
            transaction_type='debit',
            description=refund_description
        )
        db.session.add(refund_txn)
        
        # Update user balance
        user.balance_usd += refund_amount
        db.session.commit()
        
        # Verify balance integrity
        is_valid, actual_after, calculated_after, discrepancy_after = check_user_balance_integrity(user_id)
        
        flash(f'�� Refund processed: ${abs(float(refund_amount)):.2f} refunded to {user.email}. New balance: ${float(user.balance_usd):.2f}', 'admin')
        
        if not is_valid:
            flash(f'��� Note: Balance discrepancy detected after refund (${abs(float(discrepancy_after)):.2f}). Consider using Silent Adjustment to fix.', 'warning')
        
        log_admin_action('user_refund', {
            'user_id': user_id,
            'user_email': user.email,
            'refund_amount': float(refund_amount),
            'description': refund_description,
            'old_balance': float(old_balance),
            'new_balance': float(user.balance_usd),
            'transaction_id': refund_txn.id
        })
        
        return redirect(url_for('admin.admin_user_detail', user_id=user_id))
    
    # ET request - show refund form
    is_valid, actual, calculated, discrepancy = check_user_balance_integrity(user_id)
    
    # Calculate bonus/promotional funds (not refundable)
    from decimal import Decimal
    bonus_total = Decimal('0')
    for txn in Transaction.query.filter_by(user_id=user_id).all():
        if txn.description and any(keyword in txn.description.lower() for keyword in ['sign-up bonus', 'volume bonus', 'promotional']):
            if txn.amount_usd > 0:
                bonus_total += Decimal(str(txn.amount_usd))
    
    max_refundable = max(Decimal('0'), user.balance_usd - bonus_total)
    
    return render_template('admin_refund.html',
                         user=user,
                         actual_balance=actual,
                         calculated_balance=calculated,
                         discrepancy=discrepancy,
                         bonus_total=bonus_total,
                         max_refundable=max_refundable,
                         active_tab='balance_audit')


@admin_bp.route('/system')
@admin_required
def admin_system():
    settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'system_settings.json')
    
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    
    message = request.args.get('message')
    return render_template('admin_system.html', 
                         settings=settings, 
                         message=message,
                         active_tab='system')

@admin_bp.route('/failed-jobs')
@admin_required
def admin_failed_jobs():
    """Admin page showing failed job analyses with error details"""
    from datetime import timedelta
    
    # et all failed jobs (have error_message but no coverage_data)
    failed_jobs = Analysis.query.filter(
        Analysis.error_message.isnot(None),
        Analysis.error_message != ''
    ).order_by(db.desc(Analysis.failed_at)).limit(50).all()
    
    # Calculate stats
    now = datetime.now(timezone.utc)
    failed_last_24h = sum(1 for job in failed_jobs if job.failed_at and (now - job.failed_at).total_seconds() < 86400)
    affected_users = list(set(job.user_id for job in failed_jobs))
    
    # Extract common error patterns
    error_patterns = {}
    for job in failed_jobs:
        if job.error_message:
            # et first line of error (usually the exception type)
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


@admin_bp.route('/system/save', methods=['POST'])
@admin_required
def admin_system_save():
    """Save system settings"""
    settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'system_settings.json')
    
    # Load current settings
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    
    # Update values from form
    settings['registration_enabled']['value'] = request.form.get('registration_enabled') == 'on'
    settings['maintenance_mode']['value'] = request.form.get('maintenance_mode') == 'on'
    settings['max_file_size_mb']['value'] = int(request.form.get('max_file_size_mb', 10))
    settings['enable_document_length_warnings']['value'] = request.form.get('enable_document_length_warnings') == 'on'
    settings['max_resumes_per_upload']['value'] = int(request.form.get('max_resumes_per_upload', 200))
    settings['background_queue_threshold']['value'] = int(request.form.get('background_queue_threshold', 75))
    
    # Update metadata
    settings['_metadata']['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    settings['_metadata']['updated_by'] = 'admin'
    
    # Save back to file
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    flash('�� System settings saved successfully!', 'admin')
    return redirect(url_for('admin.admin_system'))


@admin_bp.route('/stats')
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

@admin_bp.route('/feedback')
@admin_required
def admin_feedback():
    """View user feedback on AI analysis accuracy"""
    # et filter parameters
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
    
    # et feedback with improvement notes
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


@admin_bp.route('/users/<int:user_id>/adjust-balance', methods=['POST'])
@admin_required
def admin_adjust_balance(user_id):
    """Adjust user balance with support for credits, debits, refunds, and corrections"""
    user = User.query.get_or_404(user_id)
    
    transaction_type = request.form.get('transaction_type')
    amount = Decimal(str(request.form.get('amount', 0)))
    reason = request.form.get('reason', 'Manual adjustment by admin')
    
    if amount <= 0:
        flash('�� Amount must be positive', 'danger')
        return redirect(url_for('admin.admin_user_detail', user_id=user_id))
    
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
        flash(f'�� Added ${amount:.2f} to {user.email}', 'success')
        
    elif transaction_type == 'manual_debit':
        # Remove funds (negative transaction)
        if user.balance_usd < amount:
            flash(f'�� Cannot debit ${amount:.2f} - user only has ${user.balance_usd:.2f}', 'danger')
            return redirect(url_for('admin.admin_user_detail', user_id=user_id))
        
        user.balance_usd -= amount
        transaction = Transaction(
            user_id=user_id,
            amount_usd=-amount,  # Negative for debit
            transaction_type='debit',
            description=f'Manual Debit - {reason}'
        )
        db.session.add(transaction)
        flash(f'�� Deducted ${amount:.2f} from {user.email}', 'success')
        
    elif transaction_type == 'refund':
        # Refund (cannot exceed refundable balance)
        if amount > max_refundable:
            flash(f'�� Cannot refund ${amount:.2f} - maximum refundable balance is ${max_refundable:.2f} (excluding ${bonus_total:.2f} in bonuses)', 'danger')
            return redirect(url_for('admin.admin_user_detail', user_id=user_id))
        
        if user.balance_usd < amount:
            flash(f'�� Cannot refund ${amount:.2f} - user only has ${user.balance_usd:.2f}', 'danger')
            return redirect(url_for('admin.admin_user_detail', user_id=user_id))
        
        user.balance_usd -= amount
        transaction = Transaction(
            user_id=user_id,
            amount_usd=-amount,  # Negative for refund
            transaction_type='debit',
            description=f'Refund - {reason}'
        )
        db.session.add(transaction)
        flash(f'�� Refunded ${amount:.2f} to {user.email}', 'success')
        
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
        flash(f'�� Applied bonus correction of ${amount:.2f} to {user.email}', 'success')
    
    else:
        flash('�� Invalid transaction type', 'danger')
        return redirect(url_for('admin.admin_user_detail', user_id=user_id))
    
    db.session.commit()
    return redirect(url_for('admin.admin_user_detail', user_id=user_id))


@admin_bp.route('/users/<int:user_id>/reset-password', methods=['POST'])
@admin_required
def admin_reset_password(user_id):
    """Reset user password (generate temporary password)"""
    user = User.query.get_or_404(user_id)
    
    # enerate temporary password
    import secrets
    temp_password = secrets.token_urlsafe(12)
    user.set_password(temp_password)
    db.session.commit()
    
    flash(f'�� Password reset for {user.email}. Temporary password: {temp_password}', 'admin')
    return redirect(url_for('admin.admin_user_detail', user_id=user_id))


