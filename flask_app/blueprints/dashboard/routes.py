from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user, logout_user
from database import db, User, Transaction, Analysis, Draft, Feedback, UserSettings
from datetime import datetime, timezone
from decimal import Decimal
import json
import os

from blueprints.dashboard import dashboard_bp


def send_balance_mismatch_alert(user, actual_balance, calculated_balance, discrepancy):
    """Send alert to admin about balance mismatch"""
    # TODO: Implement admin notification system
    pass


@dashboard_bp.route('/dashboard')
@login_required
def dashboard():
    """Dashboard - logged-in users only"""
    # Get recent analyses for the user
    recent_analyses = Analysis.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Analysis.created_at.desc()
    ).limit(5).all()
    
    # Get current draft if exists
    draft = Draft.query.filter_by(user_id=current_user.id).first()
    
    # Load pricing settings for welcome credit display
    pricing_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'pricing_settings.json')
    with open(pricing_path, 'r') as f:
        pricing_config = json.load(f)
    
    return render_template('dashboard.html', user=current_user, recent_analyses=recent_analyses, draft=draft, pricing=pricing_config)


@dashboard_bp.route('/api/get-balance')
@login_required
def get_balance():
    """API endpoint to get current user balance (for updating after Stripe payment)"""
    return jsonify({
        'balance': float(current_user.balance_usd),
        'user_id': current_user.id
    })


@dashboard_bp.route('/account')
@login_required
def account():
    """Account balance and transaction history"""
    # Get all transactions ordered newest first for display
    transactions = current_user.transactions.order_by(
        db.desc('created_at')
    ).all()
    
    # Check which analysis IDs still exist (hard delete means missing = deleted)
    analysis_ids = [t.analysis_id for t in transactions if t.analysis_id]
    existing_analyses = set()
    transactions_analyses = {}
    if analysis_ids:
        all_analyses = Analysis.query.filter(Analysis.id.in_(analysis_ids)).all()
        existing_analyses = {a.id for a in all_analyses}
        transactions_analyses = {a.id: a for a in all_analyses}
    
    # Calculate running balance for display
    # Get all transactions to calculate correct balance
    all_transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(db.asc('created_at')).all()
    
    # Calculate running balance forward from oldest to newest
    running_balance = Decimal('0')
    balance_lookup = {}  # Map transaction ID to balance after that transaction
    
    for txn in all_transactions:
        running_balance += Decimal(str(txn.amount_usd))
        balance_lookup[txn.id] = float(running_balance)
    
    # Create balances list for display transactions
    balances = [balance_lookup.get(txn.id, 0) for txn in transactions]
    
    # Verify final balance matches user's actual balance
    if balances and abs(float(current_user.balance_usd) - balances[0]) > 0.01:
        discrepancy = float(current_user.balance_usd) - balances[0]
        print(f"⚠️ WARNING: Balance mismatch for user {current_user.id} ({current_user.email}). "
              f"Calculated: ${balances[0]:.2f}, Actual: ${float(current_user.balance_usd):.2f}, "
              f"Discrepancy: ${discrepancy:.2f}")
        
        # Send alert to admin (only once per user per session to avoid spam)
        if not hasattr(current_user, '_balance_alert_sent'):
            send_balance_mismatch_alert(current_user, float(current_user.balance_usd), balances[0], discrepancy)
            current_user._balance_alert_sent = True
    
    return render_template('account.html', 
                         user=current_user,
                         transactions=transactions,
                         balances=balances,
                         existing_analyses=existing_analyses,
                         transactions_analyses=transactions_analyses)


@dashboard_bp.route('/delete-account', methods=['POST'])
@login_required
def delete_account():
    """Permanently delete user account and all associated data"""
    user_id = current_user.id
    user_email = current_user.email
    
    try:
        # Delete all user's analyses (cascades to candidate_files)
        Analysis.query.filter_by(user_id=user_id).delete()
        
        # Delete all transactions
        Transaction.query.filter_by(user_id=user_id).delete()
        
        # Delete feedback
        Feedback.query.filter_by(user_id=user_id).delete()
        
        # Delete user settings if they exist
        UserSettings.query.filter_by(user_id=user_id).delete()
        
        # Logout the user before deleting
        logout_user()
        
        # Finally delete the user account
        User.query.filter_by(id=user_id).delete()
        
        db.session.commit()
        
        flash(f'Account {user_email} has been permanently deleted.', 'success')
        return redirect(url_for('main.landing'))
        
    except Exception as e:
        db.session.rollback()
        flash('An error occurred while deleting your account. Please contact support.', 'danger')
        return redirect(url_for('dashboard.account'))


@dashboard_bp.route('/job-history')
@login_required
def job_history():
    """Job analysis history - only show completed analyses"""
    # Filter to only show analyses with coverage_data (completed jobs)
    # This excludes incomplete analyses where the job failed
    analyses = Analysis.query.filter_by(user_id=current_user.id).filter(
        Analysis.coverage_data != '',
        Analysis.coverage_data.isnot(None)
    ).order_by(db.desc(Analysis.created_at)).limit(50).all()
    
    # Calculate total deep insights generated
    total_deep_insights = 0
    for analysis in analyses:
        if analysis.gpt_candidates:
            try:
                gpt_list = json.loads(analysis.gpt_candidates)
                if isinstance(gpt_list, list):
                    total_deep_insights += len(gpt_list)
            except (json.JSONDecodeError, TypeError):
                pass
    
    return render_template('job_history.html', 
                         user=current_user,
                         analyses=analyses,
                         total_deep_insights=total_deep_insights)


@dashboard_bp.route('/delete-analysis/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    """Hard delete an analysis and its related data"""
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Verify ownership
    if analysis.user_id != current_user.id:
        flash('Unauthorized', 'error')
        return redirect(url_for('dashboard.job_history'))
    
    try:
        # Mark deletion timestamp on related transactions BEFORE deleting analysis
        deletion_time = datetime.now(timezone.utc)
        Transaction.query.filter_by(analysis_id=analysis_id).update({
            'analysis_deleted_at': deletion_time
        })
        
        # Delete related candidate files (cascade will handle this automatically due to ondelete='CASCADE')
        # Transaction records will have analysis_id set to NULL (due to ondelete='SET NULL')
        # But analysis_deleted_at will preserve when it was deleted
        
        db.session.delete(analysis)
        db.session.commit()
        
        flash('✅ Analysis deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting analysis: {str(e)}', 'error')
    
    return redirect(url_for('dashboard.job_history'))


@dashboard_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """User settings and preferences"""
    user_settings = UserSettings.get_or_create(current_user.id)
    
    if request.method == 'POST':
        try:
            # Update threshold preferences
            hi_threshold = request.form.get('hi_threshold', type=int)
            lo_threshold = request.form.get('lo_threshold', type=int)
            
            if hi_threshold and lo_threshold:
                if lo_threshold >= hi_threshold:
                    flash('Low threshold must be less than high threshold', 'error')
                    return redirect(url_for('dashboard.settings'))
                if lo_threshold < 0 or hi_threshold > 100:
                    flash('Thresholds must be between 0 and 100', 'error')
                    return redirect(url_for('dashboard.settings'))
                
                user_settings.default_hi_threshold = hi_threshold
                user_settings.default_lo_threshold = lo_threshold
            
            # Update display preferences
            results_per_page = request.form.get('results_per_page', type=int)
            if results_per_page and 5 <= results_per_page <= 100:
                user_settings.results_per_page = results_per_page
            
            show_percentages = request.form.get('show_percentages') == 'on'
            user_settings.show_percentages = show_percentages
            
            # Update export preferences
            include_evidence = request.form.get('include_evidence_by_default') == 'on'
            user_settings.include_evidence_by_default = include_evidence
            
            db.session.commit()
            flash('Settings updated successfully', 'success')
            return redirect(url_for('dashboard.settings'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating settings: {str(e)}', 'error')
            return redirect(url_for('dashboard.settings'))
    
    return render_template('settings.html', 
                         user=current_user,
                         settings=user_settings)
