from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication and credit management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100))
    
    # Account balance in USD
    balance_usd = db.Column(db.Numeric(10, 2), default=0.00, nullable=False)
    welcome_bonus_claimed = db.Column(db.Boolean, default=False, nullable=False)  # Track if signup bonus was given
    
    # Stripe customer ID for Invoice API
    stripe_customer_id = db.Column(db.String(255), nullable=True)
    
    # Account status
    is_suspended = db.Column(db.Boolean, default=False, nullable=False)
    suspension_reason = db.Column(db.String(500))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)
    last_seen = db.Column(db.DateTime)  # Track last activity for "online" status
    
    # Marketing & Analytics tracking
    signup_source = db.Column(db.String(100))  # e.g., 'organic', 'google-ad', 'referral', 'linkedin'
    total_analyses_count = db.Column(db.Integer, default=0, nullable=False)  # Track number of jobs analyzed
    total_revenue_usd = db.Column(db.Numeric(10, 2), default=0.00, nullable=False)  # Total they've spent
    
    # Relationships
    transactions = db.relationship('Transaction', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def add_funds(self, amount_usd, description='Funds added'):
        """Add funds to user account"""
        self.balance_usd += amount_usd
        transaction = Transaction(
            user_id=self.id,
            amount_usd=amount_usd,
            transaction_type='credit',
            description=description
        )
        db.session.add(transaction)
        return transaction
    
    def deduct_funds(self, amount_usd, description='Analysis Spend', analysis_id=None):
        """Deduct funds from user account"""
        if self.balance_usd < amount_usd:
            return False
        self.balance_usd -= amount_usd
        transaction = Transaction(
            user_id=self.id,
            amount_usd=-amount_usd,
            transaction_type='debit',
            description=description,
            analysis_id=analysis_id
        )
        db.session.add(transaction)
        return transaction
    
    def __repr__(self):
        return f'<User {self.email}>'


class Transaction(db.Model):
    """Transaction history for credit purchases and usage"""
    __tablename__ = 'transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Transaction details
    amount_usd = db.Column(db.Numeric(10, 2), nullable=False)  # Positive for deposits, negative for charges
    transaction_type = db.Column(db.String(20), nullable=False)  # 'credit' or 'debit'
    description = db.Column(db.Text)
    
    # Link to analysis (nullable since credit purchases don't have analysis)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id', ondelete='SET NULL'), nullable=True, index=True)
    analysis_deleted_at = db.Column(db.DateTime, nullable=True)  # Track when associated analysis was deleted
    
    # Payment details (for purchases)
    stripe_payment_id = db.Column(db.String(100))
    stripe_amount_cents = db.Column(db.Integer)  # Amount paid in cents
    
    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f'<Transaction {self.id}: ${self.amount_usd}>'


class Analysis(db.Model):
    """Analysis history"""
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Job details
    job_title = db.Column(db.String(255))
    job_description_text = db.Column(db.Text)  # Truncated to 5000 chars for display
    jd_full_text = db.Column(db.Text)  # Full JD text for recreating drafts
    jd_filename = db.Column(db.String(255))  # Original JD filename
    jd_bytes = db.Column(db.LargeBinary)  # Store original JD PDF bytes for preview
    num_candidates = db.Column(db.Integer, nullable=False)
    num_criteria = db.Column(db.Integer, nullable=False)
    
    # Results (stored as JSON)
    coverage_data = db.Column(db.Text)  # JSON: coverage matrix
    insights_data = db.Column(db.Text)  # JSON: candidate insights
    evidence_data = db.Column(db.Text)  # JSON: evidence snippets
    criteria_list = db.Column(db.Text)  # JSON: list of criteria
    category_map = db.Column(db.Text)  # JSON: {criterion: category} mapping
    gpt_candidates = db.Column(db.Text)  # JSON: list of candidates analyzed with GPT
    
    # Cost
    cost_usd = db.Column(db.Numeric(10, 2), nullable=False)  # What customer paid
    openai_cost_usd = db.Column(db.Numeric(10, 4))  # Total OpenAI API cost from token usage
    ranker_cost_usd = db.Column(db.Numeric(10, 4))  # Cost for ranking/scoring (3 calls per resume)
    insight_cost_usd = db.Column(db.Numeric(10, 4))  # Cost for deep insights (5 candidates)
    
    # Analysis metadata
    analysis_size = db.Column(db.String(20))  # 'small', 'medium', 'large'
    
    # Progress tracking (for live progress bar during AI pipeline)
    resumes_processed = db.Column(db.Integer, default=0)  # Number of resumes scored so far
    
    # Error tracking for failed analyses
    error_message = db.Column(db.Text)  # Full error details including stack trace
    failed_at = db.Column(db.DateTime)  # When the failure occurred
    
    # Analytics tracking fields
    completed_at = db.Column(db.DateTime)  # When analysis finished processing
    processing_duration_seconds = db.Column(db.Integer)  # Total processing time
    exceeded_resume_limit = db.Column(db.Boolean, default=False)  # Hit 200+ warning?
    user_chose_override = db.Column(db.Boolean, default=False)  # Clicked "Try All Anyway"?
    
    # Document size metrics
    jd_character_count = db.Column(db.Integer)  # Job description length
    avg_resume_character_count = db.Column(db.Integer)  # Average resume length
    min_resume_character_count = db.Column(db.Integer)  # Shortest resume
    max_resume_character_count = db.Column(db.Integer)  # Longest resume
    
    # Performance tracking metrics (for monitoring system health)
    retry_count = db.Column(db.Integer, default=0)  # Number of API retries during analysis
    json_fallback_count = db.Column(db.Integer, default=0)  # Times gpt-4o-mini was used instead of gpt-4o
    cost_multiplier = db.Column(db.Float)  # Actual cost / expected cost ratio
    api_calls_made = db.Column(db.Integer)  # Total OpenAI API calls for this analysis
    avg_api_response_ms = db.Column(db.Integer)  # Average API response time in milliseconds
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f'<Analysis {self.id}: {self.job_title}>'


class CandidateFile(db.Model):
    """Store candidate resume files for viewing/download"""
    __tablename__ = 'candidate_files'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True)
    candidate_name = db.Column(db.String(255), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_bytes = db.Column(db.LargeBinary, nullable=False)  # Original file
    extracted_text = db.Column(db.Text, nullable=False)  # Extracted text
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<CandidateFile {self.id}: {self.candidate_name}>'


class JobQueue(db.Model):
    """Background job queue for analysis processing"""
    __tablename__ = 'job_queue'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    draft_id = db.Column(db.Integer, db.ForeignKey('drafts.id'), nullable=False)
    
    # Job configuration
    insights_mode = db.Column(db.String(20))  # standard, deep_dive, full_radar
    
    # Status tracking
    status = db.Column(db.String(20), default='pending', nullable=False, index=True)  # pending, processing, completed, failed, cancelled
    progress = db.Column(db.Integer, default=0)  # Number of resumes processed
    total = db.Column(db.Integer, default=0)  # Total resumes to process
    
    # Results
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=True)
    error_message = db.Column(db.Text)  # Error details if failed
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    # Relationships
    user = db.relationship('User', backref='queued_jobs')
    draft = db.relationship('Draft', backref='queued_jobs')
    analysis = db.relationship('Analysis', backref='source_job', foreign_keys=[analysis_id])
    
    def __repr__(self):
        return f'<JobQueue {self.id}: {self.status}>'


class Draft(db.Model):
    """Draft JD and criteria storage (before analysis runs)"""
    __tablename__ = 'drafts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # JD details
    jd_filename = db.Column(db.String(255))
    jd_text = db.Column(db.Text)  # Full JD text
    jd_hash = db.Column(db.String(64))  # Hash for deduplication
    jd_bytes = db.Column(db.LargeBinary)  # Store original file bytes for PDF preview
    job_title = db.Column(db.String(255))  # Extracted/edited job title
    
    # Criteria (stored as JSON)
    criteria_data = db.Column(db.Text)  # JSON: [{criterion, category, use}, ...]
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Draft {self.id}: {self.jd_filename}>'


class DraftResume(db.Model):
    """Store uploaded resumes temporarily before analysis runs"""
    __tablename__ = 'draft_resumes'
    
    id = db.Column(db.Integer, primary_key=True)
    draft_id = db.Column(db.Integer, db.ForeignKey('drafts.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # File details
    file_name = db.Column(db.String(255), nullable=False)
    file_bytes = db.Column(db.LargeBinary, nullable=False)  # Original file
    extracted_text = db.Column(db.Text, nullable=False)  # Extracted text
    candidate_name = db.Column(db.String(255), nullable=False)  # Inferred name
    file_hash = db.Column(db.String(64), nullable=False)  # Hash for deduplication
    
    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<DraftResume {self.id}: {self.candidate_name}>'


class UserSettings(db.Model):
    """User preferences and default settings"""
    __tablename__ = 'user_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True, index=True)
    
    # Threshold preferences
    default_hi_threshold = db.Column(db.Integer, default=70, nullable=False)  # Default high coverage threshold
    default_lo_threshold = db.Column(db.Integer, default=40, nullable=False)  # Default low coverage threshold
    
    # Display preferences
    results_per_page = db.Column(db.Integer, default=10, nullable=False)  # History pagination
    show_percentages = db.Column(db.Boolean, default=True, nullable=False)  # Show % or raw scores
    
    # Export preferences
    include_evidence_by_default = db.Column(db.Boolean, default=True, nullable=False)  # Include evidence snippets
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('settings', uselist=False, cascade='all, delete-orphan'))
    
    def __repr__(self):
        return f'<UserSettings user_id={self.user_id}>'
    
    @staticmethod
    def get_or_create(user_id):
        """Get user settings or create with defaults"""
        settings = UserSettings.query.filter_by(user_id=user_id).first()
        if not settings:
            settings = UserSettings(user_id=user_id)
            db.session.add(settings)
            db.session.commit()
        return settings


class Feedback(db.Model):
    """User feedback on AI analysis accuracy"""
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Feedback data
    vote = db.Column(db.String(10), nullable=False)  # 'up' or 'down'
    improvement_note = db.Column(db.Text, nullable=True)  # Optional text for thumbs down
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    analysis = db.relationship('Analysis', backref='feedback')
    user = db.relationship('User', backref='feedback_given')
    
    def __repr__(self):
        return f'<Feedback {self.id}: Analysis {self.analysis_id} - {self.vote}>'


def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    with app.app_context():
        try:
            db.create_all()
        except Exception as e:
            # Handle race condition between gunicorn workers
            if 'duplicate key value violates unique constraint' in str(e):
                print(f"DEBUG: Tables already created by another worker, continuing...")
                
                # Special handling for feedback table sequence conflict
                if 'feedback_id_seq' in str(e):
                    print("🔧 Detected feedback table sequence conflict - attempting auto-fix...")
                    try:
                        # Drop and recreate the feedback table
                        db.session.execute(db.text("DROP TABLE IF EXISTS feedback CASCADE;"))
                        db.session.execute(db.text("DROP SEQUENCE IF EXISTS feedback_id_seq CASCADE;"))
                        db.session.commit()
                        print("✅ Dropped conflicting feedback table and sequence")
                        
                        # Recreate just the feedback table
                        from database import Feedback
                        Feedback.__table__.create(db.engine, checkfirst=True)
                        print("✅ Recreated feedback table successfully")
                    except Exception as fix_error:
                        print(f"⚠️ Auto-fix failed: {fix_error}")
                        # Continue anyway - table might already be fixed
            else:
                raise


class AdminLoginAttempt(db.Model):
    """Track admin login attempts for brute-force protection"""
    __tablename__ = 'admin_login_attempts'
    
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(45), nullable=False, index=True)  # IPv6 support
    attempted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    success = db.Column(db.Boolean, default=False, nullable=False)
    
    def __repr__(self):
        return f'<AdminLoginAttempt {self.id}: {self.ip_address} at {self.attempted_at}>'


class AdminAuditLog(db.Model):
    """Audit log for admin actions"""
    __tablename__ = 'admin_audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(100), nullable=False, index=True)  # e.g., 'user_suspended', 'settings_updated'
    details = db.Column(db.Text)  # JSON string with action details
    ip_address = db.Column(db.String(45), nullable=False)
    user_agent = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f'<AdminAuditLog {self.id}: {self.action} at {self.created_at}>'
