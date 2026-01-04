# Phase 2 Fallback Implementation

## Problem Summary
When analyzing resumes with many criteria (21+), the Insight Agent (GPT-4o) was hitting output token limits during Phase 2 (deep insights generation), causing:
- JSON responses with spaces stripped ("Thecandidatedemonstratedexperience...")
- JSONDecodeError: "Unterminated string starting at: line 34 column 2441"
- Complete analysis rollback, causing users to lose all their work

## Root Cause (Diagnosed by Gemini)
**"Token Collapse"** - When GPT-4o runs out of output tokens while generating JSON, it sacrifices formatting to complete the response:
- 21 criteria × ~200 chars per justification = ~4200 chars
- This can approach or exceed max_tokens limit
- Model removes spaces and truncates strings to fit
- Result: Malformed JSON that cannot be parsed

## Solution Architecture

### Phase 1: Prompt Update (COMPLETED)
**File**: `flask_app/config/prompts.json` (insight_generation.system_prompt)

**Changes**:
1. Added "OUTPUT INSTRUCTIONS" section explicitly allowing whitespace
2. Added CRITICAL constraint: "Each polished justification MUST be under 150 characters"
3. Added guidance: "If a justification is already clear, do not lengthen it"

**Purpose**: Prevent token overflow by capping individual justification length

**Status**: ✅ Tested locally, deployed to Railway

---

### Phase 2: Retry/Fallback Logic (IMPLEMENTED)
**File**: `flask_app/ai_service.py` (generate_deep_insights function)

**Changes**:

#### 1. JSON Parsing Error Handling Enhancement
**Location**: Lines 333-481

**Workflow**:
```
1. Try: Parse JSON response
2. Catch JSONDecodeError:
   a. Try sanitization (smart quotes, extract JSON, fix newlines)
   b. If sanitization fails → RETRY with reduced max_tokens
   c. If retry fails → FALLBACK to Phase 1 justifications
3. Return result (never raise exception)
```

**Key Code**:
```python
# After all sanitization strategies fail:
try:
    # RETRY: Call GPT-4o again with 20% lower max_tokens
    reduced_max_tokens = int(INSIGHT_MAX_TOKENS * 0.8)
    retry_response = await self.client.chat.completions.create(...)
    result = json.loads(retry_content)
    return result
except Exception as retry_error:
    # FALLBACK: Use Phase 1 justifications
    fallback_result = {
        "refined_justifications": {
            score.criterion: score.justification
            for score in evaluation.criterion_scores
        },
        "top_strengths": [...],
        "key_gaps": [...],
        "interview_questions": [...]
    }
    return fallback_result
```

#### 2. Rate Limit Error Handling Enhancement
**Location**: Lines 483-549

**Changes**:
- **Before**: Raised exception after max retries, causing rollback
- **After**: Returns fallback response using Phase 1 data

**Key Code**:
```python
# All rate limit retries exhausted:
print(f"WARNING: Rate limit error after {max_retries} retries")
print(f"FALLBACK: Using Phase 1 justifications due to rate limiting")
fallback_result = {...}  # Same structure as above
return fallback_result
```

#### 3. General Error Handling Enhancement
**Location**: Lines 551-end of except block

**Changes**:
- **Before**: Re-raised any non-rate-limit exception, causing rollback
- **After**: Returns fallback response for ANY error

**Key Code**:
```python
# Non-rate-limit error:
print(f"WARNING: Insights generation failed")
print(f"FALLBACK: Using Phase 1 justifications due to unexpected error")
fallback_result = {...}  # Same structure as above
return fallback_result
```

---

## Fallback Response Structure

The fallback response uses Phase 1 data (from RANKER agent) to construct a complete insights object:

```python
{
    "refined_justifications": {
        "criterion_name": "Phase 1 raw justification",
        ...  # All criteria included
    },
    "top_strengths": [
        "Strong performance in [top criterion] (95/100)",
        ...  # Top 3 by score
    ],
    "key_gaps": [
        "Opportunity for growth in [weak criterion] (45/100)",
        ...  # Bottom 2 by score
    ],
    "interview_questions": [
        "Can you elaborate on your experience with [weak criterion]?",
        ...  # Bottom 3 by score
    ]
}
```

---

## User Experience Changes

### Before (BROKEN)
1. User submits analysis with 21 criteria
2. Phase 1 completes successfully (all scores calculated)
3. Phase 2 fails with JSONDecodeError
4. **Entire analysis rolls back**
5. User charged nothing but loses all work
6. Must re-run analysis from scratch

### After (FIXED)
1. User submits analysis with 21 criteria
2. Phase 1 completes successfully (all scores calculated)
3. Phase 2 attempts deep insights:
   - Try: Generate polished insights
   - Sanitize: Fix common JSON issues
   - Retry: Reduce max_tokens and try once more
   - **Fallback: Use Phase 1 justifications as "Safe Mode"**
4. **Analysis completes successfully**
5. User sees complete ranking with all 21 criteria scores
6. User charged for work completed (Phase 1 always delivered)
7. User can make informed hiring decision immediately

---

## Business Impact

### Critical Guarantee
**"The Heavy Lifting is Always Delivered"**

Even if the Insight Agent fails to polish the justifications:
- ✅ All candidate rankings are preserved
- ✅ All criterion scores (0-100) are available
- ✅ Phase 1 justifications provide solid evidence
- ✅ User can identify top candidates immediately
- ✅ No work is lost, no rollback occurs

### Quality Levels
1. **Best Case**: Polished insights from Insight Agent (GPT-4o)
2. **Good Case**: Retry succeeds with reduced tokens
3. **Safe Mode**: Phase 1 justifications from RANKER (GPT-4o-mini)
   - Still factually accurate
   - Still tied to specific evidence
   - Still enable informed decisions

---

## Testing Strategy

### Test Case 1: Normal Operation
- Resume: 5-10 criteria
- Expected: Phase 2 succeeds, polished insights returned
- Verify: No fallback messages in logs

### Test Case 2: Token Pressure
- Resume: Nigel Fitzgerald with 21 criteria
- Expected: Retry or fallback triggers
- Verify: Analysis completes, all scores visible

### Test Case 3: Rate Limiting
- Simulate: Multiple rapid analyses
- Expected: Fallback after max retries
- Verify: User sees results, not error page

### Test Case 4: API Error
- Simulate: OpenAI service disruption
- Expected: Immediate fallback to Phase 1
- Verify: Graceful degradation, no crash

---

## Deployment Notes

### Local Testing
```bash
cd flask_app
python run.py
# Test with Nigel Fitzgerald resume + 21 criteria
# Watch console for "FALLBACK" messages
```

### Railway Deployment
```bash
git add flask_app/ai_service.py
git commit -m "Implement Phase 2 retry/fallback logic for Insight Agent failures"
git push origin main
# Railway auto-deploys
# Monitor Railway logs for "FALLBACK" messages
```

### Monitoring
**Success Indicators**:
- "SUCCESS: Retry with reduced max_tokens succeeded!"
- "SUCCESS: Fallback response constructed with X justifications"

**Warning Indicators** (non-critical):
- "WARNING: All JSON sanitization strategies failed"
- "FALLBACK: Using Phase 1 justifications as Safe Mode"

**Error Indicators** (requires investigation):
- None - all paths now lead to successful completion

---

## Future Enhancements

### Option 1: Progressive Insight Generation
Instead of generating all insights at once:
1. Split criteria into batches of 7
2. Generate insights for each batch separately
3. Combine results into final response
4. **Benefit**: Never hit token limits

### Option 2: Streaming Response
Use OpenAI's streaming API:
1. Process JSON as it arrives
2. Stop when nearing token limit
3. Use partial results + fallback for remainder
4. **Benefit**: Optimal balance of polish and reliability

### Option 3: Adaptive Max Tokens
Dynamically calculate max_tokens:
```python
criteria_count = len(evaluation.criterion_scores)
tokens_per_criterion = 50  # Conservative estimate
max_tokens = min(4000, criteria_count * tokens_per_criterion * 2)
```
4. **Benefit**: Right-sized limits per job

---

## Summary

**Problem**: Insight Agent failures caused complete analysis rollback
**Solution**: Three-layer defense (prompt limit + retry + fallback)
**Result**: Users always get complete rankings, never lose work
**Status**: Phase 1 deployed, Phase 2 ready for local testing

---

## Credits

- **Gemini AI**: Diagnosed "token collapse" root cause
- **Claude AI**: Implemented retry/fallback logic
- **User Requirement**: "Your users will see the scores for all 21 criteria, ensuring they never feel like they 'skipped the right person' because of a technical glitch."
