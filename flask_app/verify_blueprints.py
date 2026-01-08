"""
Verification script to test all blueprint endpoints after refactoring
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

def test_blueprints():
    """Test that all blueprints are registered and routes are accessible"""
    app = create_app('development')
    
    print("=" * 60)
    print("BLUEPRINT VERIFICATION TEST")
    print("=" * 60)
    print()
    
    # Get all registered blueprints
    print("Registered Blueprints:")
    print("-" * 60)
    for blueprint_name, blueprint in app.blueprints.items():
        print(f"✓ {blueprint_name:20s} (prefix: {blueprint.url_prefix or '/'})")
    print()
    
    # Test key routes from each blueprint
    test_routes = [
        # Main blueprint
        ('/', 'main.landing'),
        ('/pricing', 'main.pricing'),
        ('/features', 'main.features'),
        ('/support', 'main.support'),
        
        # Dashboard blueprint
        ('/dashboard', 'dashboard.dashboard'),
        ('/account', 'dashboard.account'),
        ('/settings', 'dashboard.settings'),
        ('/job-history', 'dashboard.job_history'),
        
        # Analysis blueprint
        ('/analyze', 'analysis.analyze'),
        ('/run_analysis', 'analysis.run_analysis_route'),
        
        # Export blueprint
        # (These require analysis_id parameter, so we'll skip detailed testing)
        
        # Admin blueprint
        ('/admin/login', 'admin.admin_login'),
        ('/admin/business-health', 'admin.admin_business_health'),
        
        # Auth blueprint (existing)
        ('/auth/login', 'auth.login'),
        ('/auth/register', 'auth.register'),
        
        # Payments blueprint (existing)
        ('/payments/buy-credits', 'payments.buy_credits'),
    ]
    
    print("Testing Routes:")
    print("-" * 60)
    
    with app.test_client() as client:
        passed = 0
        failed = 0
        
        for route, endpoint in test_routes:
            try:
                response = client.get(route, follow_redirects=False)
                # Accept 200, 302 (redirect), or 401 (auth required) as success
                if response.status_code in [200, 302, 401]:
                    print(f"✓ {route:30s} [{response.status_code}] OK")
                    passed += 1
                else:
                    print(f"✗ {route:30s} [{response.status_code}] FAILED")
                    failed += 1
            except Exception as e:
                print(f"✗ {route:30s} ERROR: {str(e)}")
                failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All blueprint routes are working correctly!")
        return True
    else:
        print(f"\n⚠️  {failed} route(s) failed verification")
        return False

if __name__ == '__main__':
    success = test_blueprints()
    sys.exit(0 if success else 1)
