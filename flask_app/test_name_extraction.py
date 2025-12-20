"""Test the new candidate name extraction logic"""

from analysis import infer_candidate_name

# Test cases
test_cases = [
    # (filename, text, expected_output_pattern)
    ("resume_john_smith.pdf", "Name: John A. Smith\nAddress: 123 Main St", "John A. Smith"),
    ("cv.pdf", "JIGNESH M DUSARA\nSoftware Engineer", "Jignesh M Dusara"),
    ("resume.pdf", "Mary-Jane O'Connor\nExperience:", "Mary-Jane O'Connor"),
    ("JohnSmith_CV_2024.pdf", "", "Johnsmith"),  # Fallback to filename
    ("resume.pdf", "Professional Summary\nExperienced developer", "Candidate"),  # No name found
]

print("Testing candidate name extraction:\n")
print("="*80)

for filename, text, expected_pattern in test_cases:
    result = infer_candidate_name(filename, text)
    status = "✓" if expected_pattern.lower() in result.lower() else "✗"
    print(f"{status} File: {filename:30} | Text: {text[:30]:30} | Result: {result}")

print("="*80)
print("\nTest complete!")
