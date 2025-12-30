import os
from datetime import timedelta
import json

class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///candidate_evaluator.db'
    # Fix for Railway PostgreSQL URL (uses postgres:// instead of postgresql://)
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    
    # Debug logging to verify database connection
    print(f"DEBUG: DATABASE_URL exists: {bool(os.environ.get('DATABASE_URL'))}")
    print(f"DEBUG: Using database: {'PostgreSQL' if SQLALCHEMY_DATABASE_URI.startswith('postgresql://') else 'SQLite'}")
    print(f"DEBUG: Database URI (first 30 chars): {SQLALCHEMY_DATABASE_URI[:30]}")
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = True  # Only send cookie over HTTPS
    SESSION_COOKIE_HTTPONLY = True  # Prevent JavaScript access
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # File Upload
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    
    # OpenAI
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # Stripe
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')
    STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY')
    STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET')
    
    # Pricing (Dynamically loaded from pricing_settings.json - Single Source of Truth)
    @staticmethod
    def get_pricing():
        """Load pricing configuration from JSON file"""
        pricing_file = os.path.join(os.path.dirname(__file__), 'config', 'pricing_settings.json')
        try:
            with open(pricing_file, 'r') as f:
                pricing_data = json.load(f)
            return {
                'BASE_ANALYSIS_PRICE': pricing_data['standard_tier_price']['value'],
                'DEEP_DIVE_PRICE': pricing_data['deep_dive_price']['value'],
                'EXTRA_INSIGHT_PRICE': pricing_data['individual_insight_price']['value'],
                'HIRING_SPRINT_CHARGE': pricing_data['hiring_sprint_charge']['value'],
                'HIRING_SPRINT_CREDIT': pricing_data['hiring_sprint_credit']['value'],
                'VOLUME_BONUS_THRESHOLD': pricing_data['volume_bonus_threshold']['value'],
                'VOLUME_BONUS_PERCENTAGE': pricing_data['volume_bonus_percentage']['value'],
                'MINIMUM_TOPUP_AMOUNT': pricing_data['minimum_topup_amount']['value']
            }
        except Exception as e:
            print(f"ERROR loading pricing_settings.json: {e}")
            # Fallback to defaults if file is missing
            return {
                'BASE_ANALYSIS_PRICE': 10.0,
                'DEEP_DIVE_PRICE': 10.0,
                'EXTRA_INSIGHT_PRICE': 1.0,
                'HIRING_SPRINT_CHARGE': 45.0,
                'HIRING_SPRINT_CREDIT': 50.0,
                'VOLUME_BONUS_THRESHOLD': 50.0,
                'VOLUME_BONUS_PERCENTAGE': 15.0,
                'MINIMUM_TOPUP_AMOUNT': 5.0
            }
    
    # Suggested fund amounts for "Add Funds" page
    SUGGESTED_AMOUNTS = [10, 25, 50, 100]
    
    # Email Configuration (SendGrid API)
    SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
    MAIL_DEFAULT_SENDER = ('Candidate Evaluator', 'contact@candidateevaluator.com')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SESSION_COOKIE_SECURE = False  # Allow HTTP in development

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

# Config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
