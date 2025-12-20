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
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)
    
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
    
    def deduct_funds(self, amount_usd, description='Analysis', analysis_id=None):
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
    description = db.Column(db.String(255))
    
    # Link to analysis (nullable since credit purchases don't have analysis)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id', ondelete='SET NULL'), nullable=True, index=True)
    
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
    cost_usd = db.Column(db.Numeric(10, 2), nullable=False)
    
    # Analysis metadata
    analysis_size = db.Column(db.String(20))  # 'small', 'medium', 'large'
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    deleted_at = db.Column(db.DateTime, nullable=True, index=True)  # Soft delete
    
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


def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
