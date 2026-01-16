from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify, current_app
from flask_login import LoginManager, login_required, current_user, logout_user
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
from flask_wtf.csrf import CSRFProtect

# Initialize extensions
csrf = CSRFProtect()

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
def create_app(config_name=None):
    """Application factory"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Configure CSRF to accept tokens from headers (for JSON requests)
    app.config['WTF_CSRF_CHECK_DEFAULT'] = True
    app.config['WTF_CSRF_METHODS'] = ['POST', 'PUT', 'PATCH', 'DELETE']
    app.config['WTF_CSRF_HEADERS'] = ['X-CSRFToken', 'X-CSRF-Token']
    
    # Initialize extensions
    init_db(app)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page'
    login_manager.login_message_category = 'info'
    
    # Initialize CSRF Protection
    csrf.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        user = User.query.get(int(user_id))
        # Block suspended users from logging in
        if user and user.is_suspended:
            return None
        # Update last_seen timestamp for online status tracking (naive UTC)
        if user:
            user.last_seen = datetime.now(timezone.utc)
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
        from flask_login import current_user
        from database import JobQueue
        
        # Get active jobs for menu visibility
        active_jobs = []
        if current_user.is_authenticated:
            active_jobs = JobQueue.query.filter_by(
                user_id=current_user.id
            ).filter(
                JobQueue.status.in_(['pending', 'processing'])
            ).all()
        
        return {
            'now': datetime.utcnow,
            'timedelta': timedelta,
            'active_jobs': active_jobs
        }
    
    @app.template_filter('number_format')
    def number_format(value):
        """Format number with thousands separator"""
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return value
    
    # Register blueprints
    from blueprints.auth import auth_bp
    from blueprints.payments import payments_bp
    from blueprints.main import main_bp
    from blueprints.dashboard import dashboard_bp
    from blueprints.analysis import analysis_bp
    from blueprints.export import export_bp
    from blueprints.admin import admin_bp
    from blueprints.api import api_bp
    from migrate_db_web import migrate_bp
    from migrate_description_web import migrate_desc_bp
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(payments_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(migrate_bp)
    app.register_blueprint(migrate_desc_bp)
    
    # Exempt specific routes from CSRF
    csrf.exempt(analysis_bp)
    csrf.exempt(api_bp)
    csrf.exempt(payments_bp)
    csrf.exempt(export_bp)
    csrf.exempt(migrate_bp)
    csrf.exempt(migrate_desc_bp)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        flash('Page not found', 'error')
        return redirect(url_for('dashboard.dashboard') if current_user.is_authenticated else url_for('main.landing'))
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        flash('An internal error occurred', 'error')
        return redirect(url_for('dashboard.dashboard') if current_user.is_authenticated else url_for('main.landing'))
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
