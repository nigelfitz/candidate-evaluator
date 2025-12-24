# Admin Configuration Panel - User Guide

## Overview

The Admin Configuration Panel allows you to adjust GPT insights generation parameters **without editing any code**. All settings are stored in `/config/gpt_settings.json` and take effect immediately after saving.

## Accessing the Admin Panel

1. Navigate to: `http://localhost:5000/admin/login`
2. Enter the admin password (default: `admin123`)
3. **IMPORTANT:** Change the default password by setting the `ADMIN_PASSWORD` environment variable

### Setting Custom Admin Password

**Windows PowerShell:**
```powershell
$env:ADMIN_PASSWORD = "YourSecurePassword123"
```

**Linux/Mac:**
```bash
export ADMIN_PASSWORD="YourSecurePassword123"
```

**Or add to `.env` file:**
```
ADMIN_PASSWORD=YourSecurePassword123
```

## Settings Explained

### 1. Core GPT Settings

#### GPT Model
- **Current Default:** `gpt-4o`
- **Options:** gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4
- **What it does:** Selects which OpenAI model generates insights
- **Cost Impact:**
  - `gpt-4o`: $2.50 per 1M input tokens, $10 per 1M output (**RECOMMENDED**)
  - `gpt-4o-mini`: 60% cheaper but lower quality (~$0.007 per insight vs $0.018)
  - `gpt-4-turbo`: Same cost as gpt-4o, similar quality
  - `gpt-4`: 3× more expensive, no benefit for this task
- **Quality Impact:** gpt-4o provides best balance - personalized insights mentioning specific companies, roles, years of experience
- **When to change:** Use gpt-4o-mini for cost savings if budget-constrained (test quality first)

#### Temperature
- **Current Default:** `0.2`
- **Range:** 0.0 (deterministic) to 2.0 (very creative)
- **What it does:** Controls randomness in GPT responses
- **Recommended:** 0.1-0.3 for professional insights
- **Cost Impact:** None - only affects output style
- **Quality Impact:**
  - **0.0-0.3:** Consistent, focused, professional (BEST for hiring)
  - **0.4-0.7:** More varied, slightly creative
  - **0.8-2.0:** Very creative, may introduce fluff or inconsistency
- **When to change:** 
  - Increase to 0.3-0.4 if insights feel too robotic
  - Decrease to 0.1 if you want maximum consistency across candidates

#### Max Response Tokens
- **Current Default:** `1000`
- **Range:** 500-2000
- **What it does:** Limits how long GPT's response can be
- **Recommended:** 1000 (insights typically use 400-600 tokens)
- **Cost Impact:** Linear - 2000 tokens = double cost, but we only pay for what's used
- **Quality Impact:**
  - **<500:** May truncate insights mid-sentence
  - **500-1000:** Perfect for 3-6 bullets + paragraph
  - **>1500:** Wastes money - GPT won't use it
- **When to change:** Only reduce to 500-700 if trying to force very brief insights

### 2. Context & Evidence Settings

#### Evidence Snippet Characters
- **Current Default:** `600`
- **Range:** 200-1000
- **What it does:** How many characters of resume text to include per criterion as "evidence"
- **Recommended:** 600 (sweet spot)
- **Cost Impact:** With 20 criteria and top 10 evidence items:
  - **400 chars:** ~4000 input tokens (~$0.010 per insight)
  - **600 chars:** ~5000 input tokens (~$0.012 per insight)
  - **1000 chars:** ~8000 input tokens (~$0.020 per insight)
- **Quality Impact:**
  - **200-400:** Often cuts mid-sentence, missing context
  - **500-700:** Complete thoughts, good context
  - **800-1000:** Diminishing returns - more noise than signal
- **When to change:**
  - Increase to 700-800 if insights seem too vague
  - Reduce to 400-500 for cost savings (accept slightly less context)

#### Candidate Resume Text Characters
- **Current Default:** `3000`
- **Range:** 1000-10000
- **What it does:** How much of the resume to send as overall context (separate from evidence snippets)
- **Recommended:** 3000 (captures header, summary, recent experience)
- **Cost Impact:**
  - **3000 chars:** ~750 tokens (~$0.002 per insight)
  - **5000 chars:** ~1250 tokens (~$0.003 per insight)
  - **10000 chars:** ~2500 tokens (~$0.006 per insight)
- **Quality Impact:**
  - **1000-2000:** May miss candidate's name, contact info, professional summary
  - **3000-5000:** Captures critical overview
  - **>5000:** Minimal improvement - evidence snippets provide specifics
- **When to change:**
  - Increase to 5000 only if resumes have very long headers/summaries
  - Rarely worth changing from 3000

#### Job Description Text Characters
- **Current Default:** `3000`
- **Range:** 1000-10000
- **What it does:** How much of JD to send as context
- **Recommended:** 3000
- **Cost Impact:** Same as candidate text
- **Quality Impact:** 3000 chars captures JD overview and key requirements
- **When to change:** Rarely - 3000 is optimal for most JDs

#### Number of Evidence Items
- **Current Default:** `10`
- **Range:** 5-20
- **What it does:** How many criteria (sorted by score) to include detailed evidence for
- **Recommended:** 10 (top 10 criteria get evidence, rest only scored)
- **Cost Impact:** **MULTIPLIES** evidence snippet cost
  - **10 items × 600 chars:** ~6000 chars evidence
  - **20 items × 600 chars:** ~12000 chars evidence (double cost)
- **Quality Impact:**
  - **5-8:** May miss important strengths/gaps
  - **10-12:** Good coverage of top criteria
  - **15-20:** Too much data, GPT may get confused
- **When to change:**
  - Increase to 15 if you have many (30+) criteria
  - Reduce to 8 for fewer (<15) criteria

### 3. Score Thresholds

#### High/Strong Threshold
- **Current Default:** `0.75`
- **Range:** 0.6-0.9
- **What it does:** Score above this = "High" or "Strong" match (green in UI)
- **Recommended:** 0.75 (75% similarity required)
- **Cost Impact:** None - only affects UI colors and what GPT considers "strong"
- **Quality Impact:**
  - **0.6-0.7:** More lenient - more criteria marked "strong"
  - **0.75-0.8:** Balanced - clear strengths
  - **0.85-0.9:** Very strict - only exceptional matches
- **When to change:**
  - Lower to 0.65-0.70 for hard-to-fill roles (fewer perfect matches expected)
  - Raise to 0.80-0.85 for highly competitive roles (only best candidates)

#### Low/Weak Threshold
- **Current Default:** `0.35`
- **Range:** 0.2-0.5
- **What it does:** Score below this = "Low" or "Weak" match (red in UI)
- **Recommended:** 0.35
- **Cost Impact:** None
- **Quality Impact:**
  - **0.2-0.3:** Strict - only major gaps flagged
  - **0.35-0.4:** Balanced - clear weaknesses highlighted
  - **0.45-0.5:** Lenient - more criteria marked as gaps
- **When to change:**
  - Lower to 0.25-0.30 to only show critical gaps
  - Raise to 0.40-0.45 to be more conservative (flag more potential concerns)

## Current Cost Per Insight

With default settings (gpt-4o, 600 char evidence, 3000 char context):
- **Input:** ~4500 tokens @ $2.50/1M = $0.011
- **Output:** ~500 tokens @ $10.00/1M = $0.005
- **Total:** ~$0.016-0.018 per insight

**Compared to pricing:**
- Base analysis: $4.00 (includes top 3 insights)
- Extra insights: $1.00 each
- **Profit margin:** ~98% per extra insight ($1.00 - $0.018 = $0.982 profit)

## Optimization Strategies

### For Cost Savings (Lower Quality)
1. Switch model to `gpt-4o-mini` (-60% cost)
2. Reduce evidence snippets to 400-500 chars
3. Reduce evidence items to 8
4. Keep temperature at 0.2
5. **New cost:** ~$0.007 per insight

### For Maximum Quality (Higher Cost)
1. Keep model at `gpt-4o`
2. Increase evidence snippets to 700-800 chars
3. Increase evidence items to 12-15
4. Keep temperature at 0.2
5. Increase candidate/JD text to 5000 chars
6. **New cost:** ~$0.025-0.030 per insight

### For Best Balance (Recommended)
1. Model: `gpt-4o`
2. Temperature: `0.2`
3. Evidence snippets: `600` chars
4. Evidence items: `10`
5. Candidate/JD text: `3000` chars
6. Max tokens: `1000`
7. **Cost:** ~$0.018 per insight

## Testing Changes

**ALWAYS test with small batches first!**

1. Make changes in admin panel
2. Save settings
3. Run analysis with **"Top 3"** insights only
4. Check quality in Insights page:
   - Are insights personalized (mention names, companies)?
   - Do strengths/gaps make sense?
   - Is evidence cited correctly?
5. If quality good, try larger batch
6. Monitor costs in transaction history

## Troubleshooting

### Insights are too generic
- ✅ Increase evidence_snippet_chars to 700-800
- ✅ Ensure candidate_text_chars is at least 3000
- ✅ Check that model is gpt-4o (not gpt-4o-mini)
- ✅ Keep temperature low (0.2-0.3)

### Insights cut off mid-sentence
- ✅ Increase max_tokens to 1200-1500
- ✅ Reduce evidence items if >15

### Costs too high
- ✅ Switch to gpt-4o-mini model
- ✅ Reduce evidence_snippet_chars to 400-500
- ✅ Reduce evidence_items to 8
- ✅ Lower max_tokens to 700-800

### Insights inconsistent between candidates
- ✅ Lower temperature to 0.1
- ✅ Ensure using gpt-4o model

## Security Notes

1. **Change default password** immediately in production
2. Admin panel is session-based (logout when done)
3. Settings file is NOT encrypted (don't store secrets there)
4. Only you should have access to `/admin` route
5. Consider adding IP whitelist if hosting publicly

## File Locations

- **Settings file:** `flask_app/config/gpt_settings.json`
- **Admin template:** `flask_app/templates/admin.html`
- **Admin login:** `flask_app/templates/admin_login.html`
- **Admin routes:** `flask_app/app.py` (lines 749-850)
- **Settings loader:** `flask_app/analysis.py` (lines 13-45)

## Support

Questions? Check:
1. This README
2. Comments in `gpt_settings.json` (detailed explanations)
3. Usage notes section in admin panel
4. Cost estimate calculator in admin panel (updates live)

---

**Last Updated:** December 18, 2025
**Version:** 1.0
