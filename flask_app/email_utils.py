"""
Email utilities for Candidate Evaluator
Handles sending transactional emails like welcome emails, notifications, etc.
"""
from flask import render_template, current_app
from flask_mail import Message
from threading import Thread

def send_async_email(app, msg, mail):
    """Send email asynchronously to avoid blocking the request"""
    with app.app_context():
        try:
            mail.send(msg)
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            # Log error but don't crash the app

def send_email(subject, recipients, html_body, text_body=None, mail=None):
    """
    Send an email
    
    Args:
        subject (str): Email subject line
        recipients (list): List of recipient email addresses
        html_body (str): HTML version of email
        text_body (str, optional): Plain text fallback
        mail (Mail, optional): Flask-Mail instance
    """
    if mail is None:
        from app import mail as default_mail
        mail = default_mail
    
    # Check if email is configured
    if not current_app.config.get('MAIL_USERNAME'):
        print("Email not configured - skipping send")
        return
    
    msg = Message(
        subject=subject,
        recipients=recipients,
        html=html_body,
        body=text_body
    )
    
    # Send asynchronously
    Thread(target=send_async_email, args=(current_app._get_current_object(), msg, mail)).start()

def send_welcome_email(user, mail=None):
    """
    Send welcome email to newly registered user
    
    Args:
        user: User object with email and name attributes
        mail (Mail, optional): Flask-Mail instance
    """
    html_body = render_template('emails/welcome_email.html', user_name=user.name or 'there')
    
    text_body = f"""Hi {user.name or 'there'},

Welcome to Candidate Evaluator!

Thanks for joining. We're excited to help you cut through the noise and find your top talent faster.

Ready to get started? It only takes 60 seconds:
1. Go to your Dashboard.
2. Upload your Job Description.
3. Drop in your Candidate Resumes.

Log in here to run your first analysis: https://candidateevaluator.com/dashboard

If you have any questions or feedback while you're testing the platform, just reply to this email. I read every message and I'd love to hear how we can make the tool better for you.

Happy hiring,
The Candidate Evaluator Team

---
© 2025 Candidate Evaluator. All rights reserved.
You received this because you signed up at candidateevaluator.com
"""
    
    send_email(
        subject="Welcome to Candidate Evaluator - Let's find your next hire",
        recipients=[user.email],
        html_body=html_body,
        text_body=text_body,
        mail=mail
    )
