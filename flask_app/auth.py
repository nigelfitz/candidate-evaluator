from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from database import db, User
from datetime import datetime
import re
import os
import json

auth_bp = Blueprint('auth', __name__)

def load_system_settings():
    """Load system settings from JSON file"""
    settings_path = os.path.join(os.path.dirname(__file__), 'config', 'system_settings.json')
    with open(settings_path, 'r') as f:
        return json.load(f)

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, ""

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Check if registration is enabled
        settings = load_system_settings()
        if not settings['registration_enabled']['value']:
            flash('We are in the final stages of development and testing. Please check back soon.', 'info')
            return render_template('register.html')
        
        email = request.form.get('email', '').strip().lower()
        name = request.form.get('name', '').strip()
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')
        
        # Validation
        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('register.html')
        
        if not validate_email(email):
            flash('Invalid email address', 'error')
            return render_template('register.html')
        
        if password != password_confirm:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        valid, msg = validate_password(password)
        if not valid:
            flash(msg, 'error')
            return render_template('register.html')
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('register.html')
        
        # Get welcome credit from system settings
        settings = load_system_settings()
        starting_balance = settings['new_user_welcome_credit']['value']
        
        # Create user
        user = User(email=email, name=name, balance_usd=starting_balance)
        user.set_password(password)
        
        # Track signup source (from URL parameter ?ref=)
        signup_source = request.args.get('ref', 'organic')
        user.signup_source = signup_source
        
        db.session.add(user)
        db.session.commit()
        
        # Log the user in immediately after registration
        login_user(user)
        flash('Welcome! Your account has been created.', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user, remember=bool(remember))
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('landing'))

@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('profile.html')
