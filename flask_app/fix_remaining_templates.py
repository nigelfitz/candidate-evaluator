"""
Fix remaining url_for patterns in templates that weren't caught by the first script.
Maps route names to their blueprint prefixes.
"""
import os
import re

# Map route names to blueprint prefixes
ROUTE_MAPPINGS = {
    # Analysis routes
    'clear_session': 'analysis',
    'run_analysis_route': 'analysis',
    'import_criteria': 'analysis',
    'review_criteria': 'analysis',
    'analyze': 'analysis',
    'update_criteria': 'analysis',
    
    # Export routes
    'export_executive_pdf': 'export',
    'export_executive_docx': 'export',
    'preview_executive_pdf_inline': 'export',
    'export_coverage_excel': 'export',
    'export_coverage_csv': 'export',
    'export_candidates_csv': 'export',
    'export_criteria_csv': 'export',
    'export_individual_pdf': 'export',
    'export_individual_docx': 'export',
    
    # Admin routes
    'admin_system_save': 'admin',
    'admin_user_detail': 'admin',
    'admin_balance_adjustment': 'admin',
    'admin_refund': 'admin',
    'admin_reset_password': 'admin',
    'admin_suspend_user': 'admin',
    'admin_unsuspend_user': 'admin',
    'admin_users': 'admin',
    'admin_stats': 'admin',
    'admin_balance_audit': 'admin',
    'admin_save_prompts': 'admin',
    'admin_save_pricing': 'admin',
    'admin_pricing': 'admin',
    'admin_gpt_settings': 'admin',
    'admin_prompts': 'admin',
    'admin_system': 'admin',
    'admin_business_health': 'admin',
    'admin_analytics': 'admin',
    'admin_feedback': 'admin',
    'admin_audit_logs': 'admin',
    'admin_failed_jobs': 'admin',
    'admin_save_settings': 'admin',
    'admin_save_business_health': 'admin',
}

def fix_template(file_path):
    """Fix url_for patterns in a single template file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = 0
    
    # Fix url_for patterns
    for route_name, blueprint in ROUTE_MAPPINGS.items():
        # Pattern 1: url_for('route_name')
        pattern1 = f"url_for('{route_name}'"
        replacement1 = f"url_for('{blueprint}.{route_name}'"
        if pattern1 in content:
            count = content.count(pattern1)
            content = content.replace(pattern1, replacement1)
            changes += count
            print(f"  Fixed {count}x: {route_name} -> {blueprint}.{route_name}")
        
        # Pattern 2: url_for("route_name")
        pattern2 = f'url_for("{route_name}"'
        replacement2 = f'url_for("{blueprint}.{route_name}"'
        if pattern2 in content:
            count = content.count(pattern2)
            content = content.replace(pattern2, replacement2)
            changes += count
            print(f"  Fixed {count}x: {route_name} -> {blueprint}.{route_name}")
    
    if changes > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return changes
    return 0

def main():
    """Fix all templates in the templates directory."""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    total_changes = 0
    files_changed = 0
    
    print("Fixing remaining url_for patterns in templates...")
    print("=" * 60)
    
    for filename in os.listdir(templates_dir):
        if filename.endswith('.html'):
            file_path = os.path.join(templates_dir, filename)
            changes = fix_template(file_path)
            if changes > 0:
                files_changed += 1
                total_changes += changes
                print(f"✓ {filename}: {changes} changes")
    
    print("=" * 60)
    print(f"Total: {total_changes} url_for patterns fixed in {files_changed} files")

if __name__ == '__main__':
    main()
