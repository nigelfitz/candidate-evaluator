from flask import jsonify
from flask_login import login_required, current_user
from database import Analysis
import json

from blueprints.api import api_bp
@api_bp.route('/api/analysis-progress/<int:analysis_id>')
@login_required
def get_analysis_progress(analysis_id):
    """API endpoint to fetch real-time analysis progress"""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    # Check if analysis is complete (has coverage_data populated)
    is_complete = bool(analysis.coverage_data)
    
    return jsonify({
        'resumes_processed': analysis.resumes_processed,
        'total_candidates': analysis.num_candidates,
        'is_complete': is_complete
    })

@api_bp.route('/api/analysis-progress/latest')
@login_required
def get_latest_analysis_progress():
    """API endpoint to fetch progress of the most recent in-progress analysis"""
    # Get the most recent analysis that's not yet complete (ordered by created_at DESC)
    analysis = Analysis.query.filter_by(
        user_id=current_user.id
    ).order_by(Analysis.created_at.desc()).first()
    
    if not analysis:
        return jsonify({'error': 'No analysis found'}), 404
    
    # Check if analysis is complete (has coverage_data populated)
    is_complete = bool(analysis.coverage_data)
    
    # Determine current phase (using analysis_size field temporarily as phase indicator)
    phase = analysis.analysis_size if analysis.analysis_size in ['phase1', 'phase2', 'complete'] else 'phase1'
    
    return jsonify({
        'analysis_id': analysis.id,
        'resumes_processed': analysis.resumes_processed,
        'total_candidates': analysis.num_candidates,
        'is_complete': is_complete,
        'phase': phase
    })

