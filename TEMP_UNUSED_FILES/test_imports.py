# Test script for analysis functionality
# Run this to test if imports work correctly

import sys
import os

# Add flask_app to path
sys.path.insert(0, os.path.dirname(__file__))

print("Testing analysis module imports...")
try:
    from analysis import (
        read_file_bytes, hash_bytes, Candidate, JD,
        extract_jd_sections_with_gpt, build_criteria_from_sections,
        analyse_candidates, infer_candidate_name
    )
    print("✓ All analysis imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\nTesting app.py imports...")
try:
    # Check if app can be created
    from app import create_app
    app = create_app('development')
    print("✓ Flask app created successfully")
except Exception as e:
    print(f"✗ App creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All imports successful! Ready to run the app.")
