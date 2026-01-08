"""
Main blueprint - public pages (landing, pricing, terms, etc.)
"""
from flask import render_template, redirect, url_for, request, flash, current_app
from flask_login import current_user
from . import main_bp
import json
import os

@main_bp.route('/')
def landing():
    """Landing page - shows marketing page for non-logged-in users, redirects to dashboard for logged-in"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    # Load pricing for display
    pricing_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'pricing_settings.json')
    with open(pricing_path, 'r', encoding='utf-8') as f:
        pricing_config = json.load(f)
    
    return render_template('landing.html', pricing=pricing_config)

@main_bp.route('/pricing')
def pricing():
    """Pricing page"""
    pricing_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'pricing_settings.json')
    with open(pricing_path, 'r', encoding='utf-8') as f:
        pricing_config = json.load(f)
    test_drive_price = pricing_config['standard_tier_price']['value'] - pricing_config['new_user_welcome_credit']['value']
    return render_template('pricing.html',
                         pricing=pricing_config,
                         test_drive_price=test_drive_price)

@main_bp.route('/features')
def features():
    """Features deep dive page"""
    return render_template('features.html')

@main_bp.route('/security')
def security():
    """Security and data privacy page"""
    return render_template('security.html')

@main_bp.route('/privacy')
def privacy():
    """Privacy Policy page"""
    return render_template('privacy.html')

@main_bp.route('/terms')
def terms():
    """Terms of Service page"""
    return render_template('terms.html')

@main_bp.route('/help')
def help():
    """Help and FAQ page"""
    return render_template('help.html', user=current_user)

@main_bp.route('/support', methods=['GET', 'POST'])
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
            return redirect(url_for('main.support'))
        except Exception as e:
            print(f"ERROR: Failed to send support email: {e}")
            flash('There was an error sending your message. Please try emailing us directly at contact@candidateevaluator.com', 'error')
            return render_template('support.html')
    
    return render_template('support.html')

@main_bp.route('/favicon.ico')
def favicon():
    """Return 204 No Content for favicon requests to avoid 404s"""
    return '', 204
