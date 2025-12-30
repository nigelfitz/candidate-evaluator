# Export Reports Updated: Reasoning + Evidence Model

**Date:** December 30, 2025  
**Status:** ✅ COMPLETE

## Overview
The PDF and DOCX export functionality has been fully updated to align with the new 4-tuple evidence structure and implement tier enforcement for locked candidates.

---

## Changes Implemented

### 1. **Evidence Structure Mapping** ✅
**File:** `flask_app/export_candidate.py`

Updated both PDF and DOCX export functions to properly extract from the new 4-tuple:
- **Index [0]**: `raw_evidence` - Verbatim quotes from resume
- **Index [1]**: `justification` - AI-generated reasoning
- **Index [2]**: `score` - Match score (0.0-1.0)
- **Index [3]**: `density` - Number of evidence occurrences

**Before:**
```python
snippet = evidence_tuple[0]  # Old: just grabbed first element
```

**After:**
```python
raw_evidence = evidence_tuple[0] if len(evidence_tuple) > 0 else ''
justification = evidence_tuple[1] if len(evidence_tuple) > 1 else ''
```

---

### 2. **PDF Export Updates** ✅

#### Function: `to_individual_candidate_pdf()`
**Lines:** 36-305

**Key Changes:**
1. **Section Renamed**: "Evidence Snippets" → "Detailed Analysis by Criterion"
2. **Two-Part Display** (for unlocked candidates):
   - **AI Justification** section with reasoning text
   - **Raw Source Text** section with verbatim resume quotes
3. **Multi-Source Handling**: Properly splits evidence by `\n\n` (double line breaks)
4. **Increased Capacity**: No truncation limits (handles max_tokens=300 from AI)

**Tier Enforcement:**
- **Unlocked Candidates**: Shows full AI justification + raw evidence
- **Locked Candidates**: Displays locked message:
  > 🔒 **Detailed analysis available in Deep Dive tier.**  
  > Unlock to view AI-generated justifications and verbatim evidence quotes.

**Styling:**
- Locked messages use gray background (#F8FAFC) with border
- Evidence quotes in italic monospace for readability

---

### 3. **DOCX Export Updates** ✅

#### Function: `to_individual_candidate_docx()`
**Lines:** 308-540

**Key Changes:**
- Mirrors PDF structure exactly
- Uses Word paragraph styles for formatting:
  - Bold headings for "AI Justification:" and "Raw Source Text:"
  - Intense Quote style for evidence text
  - Standard paragraphs for justifications

**Tier Enforcement:**
- Same locked/unlocked logic as PDF
- Locked message with 🔒 emoji

---

### 4. **Function Signatures Updated** ✅

Both export functions now declare:
```python
evidence_map: Dict[Tuple[str,str], Tuple[str,str,float,int]]
```

**Tuple Structure:**
- Key: `(candidate_name, criterion_name)`
- Value: `(raw_evidence, justification, score, density)`

---

## Technical Details

### Evidence Splitting
**Multi-source quotes** are properly handled:
```python
evidence_parts = raw_evidence.split('\n\n')
for evidence_part in evidence_parts:
    clean_evidence = evidence_part.strip()
    if clean_evidence:
        # Display each source separately
```

### Text Wrapping
- **Removed truncation**: No more 300-character limits
- **Full display**: AI justifications and evidence shown in full
- **ReportLab auto-wrap**: Paragraph styles handle long text automatically

### Backward Compatibility
Code still handles old 2-tuple format gracefully:
```python
if len(evidence_tuple) > 1 else ''
```
This prevents crashes on legacy analyses.

---

## Tier Enforcement Logic

### Unlocked Candidates
**Condition:** `has_ai_insights = candidate_name in gpt_candidates`

When **True:**
1. Show "AI Justification" heading + justification text
2. Show "Raw Source Text" heading + verbatim quotes
3. Display multiple sources if present

### Locked Candidates
When **False:**
1. Show locked card with message
2. No justifications visible
3. No raw evidence visible
4. Professional upgrade prompt

---

## Files Modified

| File | Lines Changed | Changes |
|------|--------------|---------|
| `export_candidate.py` | 36-305 | PDF export function + evidence section |
| `export_candidate.py` | 308-540 | DOCX export function + evidence section |

---

## Testing Checklist

- [x] PDF exports compile without errors
- [x] DOCX exports compile without errors
- [x] Function signatures updated
- [x] 4-tuple extraction works
- [x] Multi-source evidence splits correctly
- [ ] **Manual Test Needed:** Generate PDF for unlocked candidate
- [ ] **Manual Test Needed:** Generate PDF for locked candidate
- [ ] **Manual Test Needed:** Verify justifications display correctly
- [ ] **Manual Test Needed:** Verify raw evidence displays correctly
- [ ] **Manual Test Needed:** Test DOCX export formatting

---

## Routes Using Updated Exports

**PDF Download:**
- Route: `/download-candidate-pdf/<analysis_id>/<candidate_name>`
- Function: `download_candidate_pdf()` in `app.py` (line 1841)
- Status: ✅ Already passes evidence_map correctly

**Individual PDF Export:**
- Route: `/export/<analysis_id>/individual-pdf`
- Function: `export_individual_pdf()` in `app.py` (line 2558)
- Status: ✅ Compatible with new format

**Individual DOCX Export:**
- Route: `/export/<analysis_id>/individual-docx`
- Function: `export_individual_docx()` in `app.py` (line 2692)
- Status: ✅ Compatible with new format

---

## Example Output

### Unlocked Candidate PDF:
```
Detailed Analysis by Criterion
Showing AI justifications and source evidence for top-scored criteria

Python Experience (Score: 92%)

AI Justification:
The candidate demonstrates strong Python proficiency with 5+ years of 
professional experience in data science and web development frameworks.

Raw Source Text:
"Developed and maintained Python-based data pipelines using pandas, NumPy, 
and scikit-learn for predictive modeling."

"Built RESTful APIs with Flask and Django, serving 10M+ requests/day with 
99.9% uptime."
```

### Locked Candidate PDF:
```
Detailed Analysis by Criterion
Deep Insights locked. Upgrade to view AI justifications and detailed evidence.

Python Experience (Score: 92%)

┌─────────────────────────────────────────────────────┐
│ 🔒 Detailed analysis available in Deep Dive tier.  │
│ Unlock to view AI-generated justifications and     │
│ verbatim evidence quotes.                          │
└─────────────────────────────────────────────────────┘
```

---

## Benefits

1. **Professional Quality**: Exports now match the high-quality web interface
2. **Verbatim Evidence**: Real resume quotes instead of summaries
3. **Clear Reasoning**: AI justifications explain the "why" behind scores
4. **Tier Respect**: Locked candidates can't bypass paywall via export
5. **Multi-Source**: Handles evidence from multiple jobs/experiences
6. **No Truncation**: Full context preserved (handles 300 tokens)

---

## Next Steps

### Immediate
1. **Test exports manually** on live system
2. **Verify formatting** in Adobe Reader and Word
3. **Check multi-source splitting** with candidates who have multiple jobs

### Future Enhancements
- Add "Export All Unlocked Candidates" bulk PDF option
- Include criterion-by-criterion evidence in Executive Summary PDF
- Add Excel export with evidence columns (optional)

---

## Notes

- **CSV Exports**: No changes needed (don't include evidence data)
- **Excel Exports**: No changes needed (coverage matrix only)
- **Executive Summary PDF**: Uses insights data (top/gaps/notes), not evidence_map
- **Backward Compatibility**: Old analyses still export without errors

---

**Status:** Ready for production testing ✅
