"""
Email utilities for Candidate Evaluator
Handles sending transactional emails using SendGrid API
"""
from flask import render_template, current_app
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
import os

def send_email(subject, recipients, html_body, text_body=None):
    """
    Send an email using SendGrid API
    
    Args:
        subject (str): Email subject line
        recipients (list): List of recipient email addresses
        html_body (str): HTML version of email
        text_body (str, optional): Plain text fallback
    """
    # Check if SendGrid is configured
    api_key = current_app.config.get('SENDGRID_API_KEY')
    if not api_key:
        print("SendGrid not configured - skipping email send")
        return
    
    try:
        # Get sender from config
        sender_tuple = current_app.config.get('MAIL_DEFAULT_SENDER', ('Candidate Evaluator', 'contact@candidateevaluator.com'))
        from_email = Email(sender_tuple[1], sender_tuple[0])
        
        # Send to each recipient
        for recipient in recipients:
            to_email = To(recipient)
            
            # Create message
            message = Mail(
                from_email=from_email,
                to_emails=to_email,
                subject=subject,
                plain_text_content=text_body or html_body,
                html_content=html_body
            )
            
            # Send via SendGrid
            sg = SendGridAPIClient(api_key)
            response = sg.send(message)
            
            if response.status_code >= 200 and response.status_code < 300:
                print(f"✅ Email sent to {recipient}")
            else:
                print(f"⚠️ SendGrid returned status {response.status_code} for {recipient}")
                
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def send_welcome_email(user):
    """
    Send welcome email to newly registered user
    
    Args:
        user: User object with email and name attributes
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
        text_body=text_body
    )
