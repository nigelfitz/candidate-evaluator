import re
import os
from pathlib import Path

# Define blueprint mappings (route -> blueprint)
route_to_blueprint = {
    # Main blueprint
    'landing': 'main', 'pricing': 'main', 'features': 'main', 'security': 'main',
    'privacy': 'main', 'terms': 'main', 'help': 'main', 'support': 'main',
    # Auth blueprint
    'register': 'auth', 'login': 'auth', 'logout': 'auth',
    # Payments blueprint
    'wallet': 'payments', 'buy_credits': 'payments', 'add_test_funds': 'payments',
    'create_checkout': 'payments', 'create_checkout_session': 'payments',
    'success': 'payments', 'webhook': 'payments',
    # Dashboard blueprint
    'dashboard': 'dashboard', 'get_balance': 'dashboard', 'account': 'dashboard',
    'delete_account': 'dashboard', 'job_history': 'dashboard', 'delete_analysis': 'dashboard',
    'settings': 'dashboard',
    # Analysis blueprint (many routes)
    'analyze': 'analysis', 'clear_session': 'analysis', 'delete_resume': 'analysis',
    'load_analysis_to_draft': 'analysis', 'run_analysis': 'analysis', 'results': 'analysis',
    'submit_feedback': 'analysis', 'export_analysis': 'analysis', 'review_criteria': 'analysis',
    'export_criteria': 'analysis', 'import_criteria': 'analysis', 'view_jd_pdf': 'analysis',
    'view_candidate_file': 'analysis', 'insights': 'analysis', 'unlock_candidate': 'analysis',
    'download_candidate_pdf': 'analysis', 'criteria': 'analysis',
    'jd_length_warning': 'analysis', 'resume_length_warning': 'analysis',
    'document_warnings': 'analysis',
    # Export blueprint
    'exports': 'export', 'preview_pdf': 'export', 'preview_pdf_inline': 'export',
    'executive_pdf': 'export', 'executive_docx': 'export', 'coverage_excel': 'export',
    'coverage_csv': 'export', 'individual_pdf': 'export', 'individual_docx': 'export',
    'candidates_csv': 'export', 'criteria_csv': 'export'
}

# Find all templates
templates_dir = Path('templates')
total_changes = 0
files_with_changes = []

for template in templates_dir.glob('*.html'):
    content = template.read_text(encoding='utf-8')
    changes = 0
    
    for route, blueprint in route_to_blueprint.items():
        # Skip auth routes as they already have blueprint prefix
        if blueprint == 'auth' and f"url_for('{blueprint}.{route}'" in content:
            continue
        
        # Pattern 1: url_for('route')
        old_pattern = f"url_for('{route}'"
        new_pattern = f"url_for('{blueprint}.{route}'"
        count = content.count(old_pattern)
        if count > 0:
            changes += count
            content = content.replace(old_pattern, new_pattern)
    
    if changes > 0:
        template.write_text(content, encoding='utf-8')
        files_with_changes.append((template.name, changes))
        total_changes += changes

print(f'Fixed {total_changes} url_for patterns in {len(files_with_changes)} files:')
for filename, count in files_with_changes:
    print(f'  {filename}: {count} changes')
