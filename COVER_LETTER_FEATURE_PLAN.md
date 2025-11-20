# Cover Letter Feature - Implementation Plan

**Status:** Postponed  
**Date:** November 12, 2025  
**Current Clean Backup:** appv36-restored-clean.py

---

## Problem Statement

Candidates often provide or are required to submit cover letters alongside their resumes. Currently, the app only analyzes resumes, missing valuable context from cover letters that could improve candidate assessment.

---

## Proposed Solution (Option 2 - Agreed Upon)

**Separate Upload with Auto-Matching**

### User Workflow
1. Bulk upload resumes (existing functionality)
2. Bulk upload cover letters (new)
3. System auto-matches cover letters to candidates by filename similarity
4. User reviews matches and manually adjusts if needed via dropdown
5. Click "Apply Cover Letters" to associate them with candidates
6. Option to add individual cover letters later in Candidate Insights tab

### Key Benefits
- Keeps cover letters and resumes separate for transparency
- Auto-matching reduces manual work
- Flexible: supports bulk and individual uploads
- Clear UI showing which candidates have cover letters

---

## Technical Implementation Plan

### 1. Data Structure Changes

**Candidate Class Extension:**
```python
@dataclass
class Candidate:
    name: str
    file_name: str
    text: str
    hash: str
    raw_bytes: Optional[bytes] = None
    # NEW FIELDS:
    cover_letter_file_name: Optional[str] = None
    cover_letter_text: Optional[str] = None
    cover_letter_raw_bytes: Optional[bytes] = None
    
    @property
    def combined_text(self) -> str:
        """Combine cover letter and resume for analysis"""
        if self.cover_letter_text:
            return f"{self.cover_letter_text}\n\n--- RESUME/CV ---\n\n{self.text}"
        return self.text
```

### 2. Auto-Matching Algorithm

**Fuzzy Name Matching Function:**
```python
def match_cover_letter_to_candidates(
    cover_letter_filename: str, 
    candidates: List[Candidate]
) -> Tuple[Optional[Candidate], float]:
    """
    Match cover letter to candidate using filename similarity
    Returns: (best_match_candidate, confidence_score)
    """
    # Extract name from filename (remove extension, special chars)
    # Use Jaccard similarity on word sets
    # Consider word overlap scoring
    # Return match with confidence >= 0.4
```

**Matching Strategy:**
- Clean filenames: remove extensions, underscores, special characters
- Tokenize into words
- Calculate Jaccard similarity: intersection / union of word sets
- Consider word overlap percentage
- Threshold: 0.4 confidence minimum for auto-match
- If below threshold or tie: require manual selection

### 3. UI Components

#### Upload & Analyse Tab

**New Section (after candidate upload):**
```
📄 Cover Letters (Optional)
- File uploader (multiple files, PDF/DOCX)
- Only shows if candidates already uploaded
- Processing:
  1. Extract text from each cover letter
  2. Auto-match to candidates
  3. Display matching table with confidence scores
  4. Dropdowns for manual adjustment
  5. "Apply Cover Letters" button
- Status metric: "💌 X candidates with cover letters"
```

**Critical Logic Order:**
```python
# MUST process in this order:
1. Process uploaded resumes → create candidates
2. Store candidates in session state
3. THEN show cover letter section (if candidates exist)
4. Process cover letters and match
```

#### Candidate Insights Tab

**Resume/CV Expander Updates:**
```
📄 Resume / CV
├── Cover Letter Section (if exists)
│   ├── 📄 [filename]
│   ├── PDF Viewer (inline, 600px)
│   ├── Remove button
│   └── Extracted text (collapsible)
│
└── Resume Section
    ├── 📄 [filename]
    ├── PDF Viewer (inline, 600px)
    └── Extracted text (collapsible)

[+ Upload Cover Letter] button (if none exists)
```

### 4. Additional Features

**Editable Candidate Names:**
- Replace static name display with `st.text_input`
- Allow user to correct names for better matching
- Update session state on change

**Remove Candidates Tab:**
- Functionality consolidated into Candidate Insights
- Remove from navigation array

---

## Issues Encountered in First Implementation

### Primary Issues

1. **Logic Order Bug**
   - Cover letter section checked `if existing_candidates:` 
   - But candidates were processed AFTER in the code flow
   - Result: Section never displayed

2. **Session State Persistence**
   - Cover letters not surviving `st.rerun()` calls
   - File uploaders clear after rerun (normal Streamlit behavior)
   - Need more robust state management

3. **UI Regressions**
   - Candidate Insights layout changed unexpectedly
   - Possible file sync issues during rapid edits
   - Multiple simultaneous changes increased risk

4. **Complexity**
   - Too many changes at once (data model + UI + matching logic)
   - Difficult to isolate which change caused which issue

### Root Causes

- **File Upload Ordering:** Streamlit processes widgets top-to-bottom; conditional sections must come after data they depend on
- **State Management:** Complex interactions between file uploads, session state, and reruns
- **Change Scope:** Attempted too much in one implementation cycle

---

## Revised Implementation Strategy

### Phase 1: Data Structure Only (Minimal Risk)
- Add cover letter fields to Candidate class
- Add `combined_text` property
- Test that existing functionality still works
- **NO UI CHANGES YET**

### Phase 2: Manual Upload (Low Risk)
- Add individual cover letter upload to Candidate Insights Resume/CV expander
- Simple file upload → extract text → store in session state
- Display cover letter separately from resume
- Test thoroughly with 1-2 candidates

### Phase 3: Bulk Upload (Medium Risk)
- Add cover letter section to Upload & Analyse tab
- **CRITICAL:** Place AFTER candidate processing in code
- Simple manual matching first (just dropdowns, no auto-match)
- Test state persistence

### Phase 4: Auto-Matching (Higher Risk)
- Implement fuzzy matching algorithm
- Add confidence scores
- Allow manual override
- Test with various filename patterns

### Alternative: Two-Phase Upload
Instead of mixing uploads on one page:
1. Upload & Analyse tab: Resumes only
2. New "Add Cover Letters" button/section on Candidate Insights tab
3. Simpler state management (candidates already exist)
4. Lower risk of logic ordering issues

---

## Testing Checklist

Before considering feature complete:

- [ ] Upload JD + resumes → verify candidates created
- [ ] Upload cover letters → verify matching works
- [ ] Manual adjustment → verify dropdowns update correctly
- [ ] Apply button → verify cover letters persist to candidates
- [ ] Navigate to Candidate Insights → verify cover letter displays
- [ ] Verify PDF viewer shows cover letter correctly
- [ ] Add individual cover letter → verify it persists
- [ ] Remove cover letter → verify it's removed from candidate
- [ ] Re-run analysis → verify combined_text used
- [ ] Check all tabs → verify no UI regressions
- [ ] Test with various filename patterns
- [ ] Test with missing cover letters (some candidates without)
- [ ] Test session state survival across tab changes

---

## Known Constraints

1. **Streamlit File Uploaders:** Clear after `st.rerun()` - must store data in session state immediately
2. **Widget Order:** Widgets process top-to-bottom; conditionals must come after dependencies
3. **Emoji Corruption:** `replace_string_in_file` tool may corrupt UTF-8 emojis - manual fix needed
4. **Re-analysis Cost:** Adding cover letters individually requires re-running OpenAI analysis (API cost)

---

## Future Enhancements

- **Single-Candidate Re-analysis:** Add function to re-analyze just one candidate (when cover letter added individually)
- **Cover Letter Insights:** Separate GPT analysis of cover letter (motivation, writing quality, fit)
- **Match History:** Log auto-match decisions for debugging
- **Bulk Remove:** Remove all cover letters at once
- **Cover Letter Required Flag:** Mark certain roles as requiring cover letters

---

## Backup Files for Reference

- **appv35-before-cover-letters.py** - Clean state before implementation
- **appv36-restored-clean.py** - Current clean state (post-restoration)
- **app.py** - Current working version (identical to appv36)

---

## Code Snippets for Reference

### Fuzzy Matching Implementation (Attempted)
```python
def match_cover_letter_to_candidates(cl_filename: str, candidates: List[Candidate]) -> Tuple[Optional[Candidate], float]:
    """Match cover letter filename to candidate using Jaccard similarity"""
    import re
    
    # Clean filename
    cl_name = re.sub(r'[_\-.]', ' ', cl_filename.lower())
    cl_name = re.sub(r'\s+', ' ', cl_name).strip()
    cl_words = set(cl_name.split())
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        cand_name = candidate.name.lower()
        cand_words = set(cand_name.split())
        
        # Jaccard similarity
        intersection = cl_words & cand_words
        union = cl_words | cand_words
        
        if union:
            jaccard = len(intersection) / len(union)
            
            # Word overlap bonus
            overlap = len(intersection) / len(cl_words) if cl_words else 0
            
            # Combined score
            score = (jaccard * 0.6) + (overlap * 0.4)
            
            if score > best_score:
                best_score = score
                best_match = candidate
    
    # Threshold
    if best_score >= 0.4:
        return best_match, best_score
    
    return None, 0.0
```

---

## Success Criteria

Feature is complete when:
1. Users can bulk upload cover letters with auto-matching
2. Users can review and adjust matches before applying
3. Cover letters persist correctly in session state
4. Candidate Insights shows cover letters separately from resumes
5. Users can add/remove individual cover letters
6. Analysis uses combined text (cover letter + resume)
7. No regressions to existing functionality
8. All tests pass
9. Code is clean and well-documented

---

**Next Steps When Resuming:**
1. Review this document
2. Start with Phase 1 (data structure only)
3. Create new backup before changes
4. Implement incrementally with testing at each phase
5. Don't rush - test thoroughly after each change
