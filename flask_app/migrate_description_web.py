"""
Web-accessible migration to extend Transaction.description from VARCHAR(255) to TEXT
Visit: /admin/migrate-transaction-description
"""
from flask import Blueprint, jsonify
from database import db
from sqlalchemy import text, inspect
from auth import admin_required

migrate_desc_bp = Blueprint('migrate_desc', __name__)

@migrate_desc_bp.route('/admin/migrate-transaction-description')
@admin_required
def migrate_transaction_description():
    """Extend description field from VARCHAR(255) to TEXT"""
    try:
        # Get column info
        inspector = inspect(db.engine)
        columns = inspector.get_columns('transactions')
        
        desc_column = next((col for col in columns if col['name'] == 'description'), None)
        
        if not desc_column:
            return jsonify({
                'status': 'error',
                'message': 'Description column not found'
            }), 500
        
        # Check if already TEXT
        current_type = str(desc_column['type']).upper()
        if 'TEXT' in current_type:
            return jsonify({
                'status': 'success',
                'message': 'Column already TEXT type',
                'current_type': current_type
            })
        
        # Alter the column
        db.session.execute(text(
            "ALTER TABLE transactions ALTER COLUMN description TYPE TEXT;"
        ))
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Successfully migrated description field to TEXT',
            'previous_type': current_type,
            'new_type': 'TEXT'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
