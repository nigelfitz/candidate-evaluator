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


def send_support_email(user_name, user_email, subject, message):
    """
    Send support request email to admin
    
    Args:
        user_name (str): Name of user submitting request
        user_email (str): Email address of user
        subject (str): Subject line of support request
        message (str): Support message content
    """
    # Support emails go to admin
    admin_email = current_app.config.get('ADMIN_EMAIL', 'contact@candidateevaluator.com')
    
    html_body = f"""
    <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px;">
                    New Support Request
                </h2>
                
                <div style="background: #f5f7fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <p style="margin: 5px 0;"><strong>From:</strong> {user_name}</p>
                    <p style="margin: 5px 0;"><strong>Email:</strong> <a href="mailto:{user_email}">{user_email}</a></p>
                    <p style="margin: 5px 0;"><strong>Subject:</strong> {subject}</p>
                </div>
                
                <div style="background: white; border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px;">
                    <h3 style="margin-top: 0; color: #334155;">Message:</h3>
                    <p style="white-space: pre-wrap;">{message}</p>
                </div>
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #64748b; font-size: 0.9rem;">
                    <p>Reply directly to this email to respond to the user.</p>
                </div>
            </div>
        </body>
    </html>
    """
    
    text_body = f"""New Support Request

From: {user_name}
Email: {user_email}
Subject: {subject}

Message:
{message}

---
Reply directly to this email to respond to the user.
"""
    
    send_email(
        subject=f"Support Request: {subject}",
        recipients=[admin_email],
        html_body=html_body,
        text_body=text_body
    )
