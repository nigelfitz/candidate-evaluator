# Project Improvement Ideas

This document tracks potential enhancements and features for future development.

---

## 1. AI-Generated Follow-Up Questions for Candidates

**Category:** Candidate Evaluation  
**Priority:** Medium-High  
**Added:** 2025-11-11

### Description
Enable users to generate a list of follow-up/clarification questions to ask selected candidate(s) based on their coverage analysis.

### Rationale
- Candidates may have relevant skills/experience that weren't mentioned in their resume
- GPT analysis might miss or misinterpret certain qualifications
- Helps interviewers probe gaps or ambiguous areas identified in the coverage matrix
- Provides structured way to verify weak scores before dismissing a candidate

### Proposed Implementation
- Add feature to Coverage or Insights tab
- User selects one or more candidates
- System generates 5-10 targeted questions based on:
  - Criteria with medium/low scores (potential gaps)
  - Ambiguous evidence snippets
  - Missing but important qualifications from JD
- Questions should be specific and actionable
- Option to export questions as part of interview prep document

### Technical Considerations
- Use GPT-4o with structured output (JSON schema)
- Input: candidate resume, JD, coverage scores, evidence snippets
- Prompt engineering to generate open-ended, non-leading questions
- Could leverage existing `evidence_map` for context

### Example Output
For candidate "John Smith" with low score on "Agile/Scrum experience":
- "Can you describe your experience working in Agile environments? What ceremonies did you participate in?"
- "Your resume mentions project management—did this involve sprint planning or iterative development?"

---

## 2. Deep Analysis Mode (GPT Re-scoring for Finalists)

**Category:** Analysis Enhancement  
**Priority:** Medium  
**Added:** 2025-11-11 (from original ideas file)

### Description
Add optional GPT-powered re-scoring for top 3-5 candidates with detailed reasoning, after initial NLP filtering.

### Rationale
- Use traditional NLP for initial fast filtering of all candidates
- Then apply GPT for nuanced analysis only on finalists
- Gives recruiters confidence in close decisions without the cost/speed penalty of scoring all candidates with GPT
- Best of both worlds: speed + depth where it matters

### Proposed Implementation
- Add "Deep Analysis" button on Coverage tab (enabled after initial scoring)
- User selects top N candidates (default 3-5)
- GPT re-evaluates selected candidates with more detailed prompts
- Provides reasoning for each score, not just the number
- Could compare candidates head-to-head for key criteria

### Technical Considerations
- Requires careful prompt design to get consistent, actionable reasoning
- May need to increase token limits for comprehensive analysis
- Should cache results to avoid re-running expensive operations

---

## 3. Scale for Large Candidate Pools

**Category:** Performance & Scalability  
**Priority:** High  
**Added:** 2025-11-11 (from original ideas file)

### Description
Add "Skip GPT insights" toggle to handle 50+ candidates faster by making GPT analysis optional.

### Rationale
- Currently GPT insights are generated for every candidate (bottleneck for large batches)
- Processing hundreds of resumes hits rate limits and causes long wait times
- Many use cases only need semantic scoring, not full GPT insights

### Proposed Implementation
- Add checkbox on Upload & Analyse: "Generate GPT insights (slower)"
- Default: OFF for batches > 10 candidates
- Run fast NLP-only scoring first
- User can selectively generate GPT insights for top N candidates afterward
- Add batch queueing with progress tracking for large uploads
- Consider pagination/lazy loading for Coverage matrix with 50+ candidates

### Technical Considerations
- Separate scoring pipeline from insights generation
- Add progress bar with ETA for large batches
- Implement retry logic for API rate limits
- Consider streaming results as they complete rather than waiting for all

---

## 4. Candidate-Facing Version (Separate Project)

**Category:** New Product  
**Priority:** Low (Future)  
**Added:** 2025-11-11 (from original ideas file)

### Description
Create separate app for **candidates** (not recruiters) to self-assess their resume against job descriptions.

### Rationale
- Helps candidates identify gaps before applying
- Could prompt candidates with questions to surface unlisted experience
- May improve resume quality by suggesting missing keywords/skills
- Different UX focus: coaching vs. evaluation

### Proposed Implementation
- Fork this codebase or start fresh with shared libraries
- User uploads: their resume + target JD
- App shows strengths/weaknesses from candidate perspective
- Interactive Q&A: "Do you have experience with X that's not on your resume?"
- Based on answers, suggest resume improvements
- Possibly generate updated resume sections

### Technical Considerations
- Requires different tone/messaging (helpful vs. evaluative)
- Privacy: candidate data should not be retained
- Could monetize as a B2C SaaS product
- May need resume editing features (beyond scope of current app)

### Notes
- This is a distinct product—should be separate repository
- Consider user research to validate demand
- Overlap with existing resume optimization tools (Jobscan, etc.)

---

## Notes
- Ideas should be clearly scoped before implementation
- Consider user feedback and pain points when prioritizing
- Keep the app simple and focused—not every idea needs to be implemented
- New ideas can be added using the template format above
