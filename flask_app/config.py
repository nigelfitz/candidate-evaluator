import os
from datetime import timedelta

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
    
    # Pricing (in USD)
    BASE_ANALYSIS_PRICE = 4.00  # Includes JD extraction + all candidates scored/ranked + top 3 AI insights + all reports
    EXTRA_INSIGHT_PRICE = 1.00  # Per additional AI insight beyond top 3
    
    # Suggested fund amounts for "Add Funds" page
    SUGGESTED_AMOUNTS = [10, 25, 50, 100]

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
