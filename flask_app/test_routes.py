"""
Simple verification script to test all blueprint endpoints
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

def test_routes():
    """Test that routes from each blueprint are accessible"""
    app = create_app('development')
    
    print("=" * 70)
    print("BLUEPRINT ROUTE VERIFICATION")
    print("=" * 70)
    print()
    
    # Test routes from each blueprint
    test_cases = [
        # Main blueprint
        ('/', 'Main - Landing'),
        ('/pricing', 'Main - Pricing'),
        ('/features', 'Main - Features'),
        
        # Dashboard blueprint
        ('/dashboard', 'Dashboard - Home'),
        ('/account', 'Dashboard - Account'),
        ('/settings', 'Dashboard - Settings'),
        
        # Analysis blueprint
        ('/analyze', 'Analysis - Start'),
        
        # Export blueprint (will need auth)
        # Skipping as they require analysis_id
        
        # Admin blueprint
        ('/admin/login', 'Admin - Login'),
        
        # Auth blueprint
        ('/auth/login', 'Auth - Login'),
        ('/auth/register', 'Auth - Register'),
        
        # Payments blueprint
        ('/payments/wallet', 'Payments - Wallet'),
    ]
    
    with app.test_client() as client:
        passed = 0
        failed = 0
        
        for route, description in test_cases:
            try:
                response = client.get(route, follow_redirects=False)
                # Accept 200 (OK), 302 (redirect), or 401 (auth required) as success
                if response.status_code in [200, 302, 401]:
                    status_emoji = "✅"
                    passed += 1
                else:
                    status_emoji = "❌"
                    failed += 1
                print(f"{status_emoji} [{response.status_code}] {description:30s} {route}")
            except Exception as e:
                print(f"❌ [ERR] {description:30s} {route} - {str(e)}")
                failed += 1
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 All blueprint routes are working correctly!")
        return 0
    else:
        print(f"\n⚠️  {failed} route(s) failed verification")
        return 1

if __name__ == '__main__':
    sys.exit(test_routes())
