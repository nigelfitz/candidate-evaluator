"""
Web-accessible database migration for Railway
Visit /admin/migrate-stripe-customer-id after deployment
"""
from flask import Blueprint, jsonify
from database import db
from sqlalchemy import text, inspect

migrate_bp = Blueprint('migrate', __name__)

@migrate_bp.route('/admin/migrate-stripe-customer-id', methods=['GET'])
def migrate_stripe_customer_id():
    """Add stripe_customer_id column if it doesn't exist"""
    try:
        # Check if column already exists
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('users')]
        
        if 'stripe_customer_id' in columns:
            return jsonify({
                'status': 'success',
                'message': 'Column stripe_customer_id already exists!',
                'action': 'none'
            })
        
        # Add the column
        with db.engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR(255)"
            ))
            conn.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Successfully added stripe_customer_id column!',
            'action': 'added_column'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
