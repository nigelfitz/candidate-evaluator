# Complete Analytics System Implementation Plan

**Status**: Future Enhancement
**Estimated Effort**: 8-12 hours
**Current Status**: Basic analytics live with 4 fields

---

## Phase 1: Database Schema Enhancements (2-3 hours)

### Analysis Table - Add Comprehensive Metrics
```sql
-- Document size metrics
ALTER TABLE analyses ADD COLUMN jd_file_size_bytes INTEGER;
ALTER TABLE analyses ADD COLUMN jd_character_count INTEGER;
ALTER TABLE analyses ADD COLUMN total_resume_file_size_bytes BIGINT;
ALTER TABLE analyses ADD COLUMN avg_resume_character_count INTEGER;
ALTER TABLE analyses ADD COLUMN min_resume_character_count INTEGER;
ALTER TABLE analyses ADD COLUMN max_resume_character_count INTEGER;

-- API usage tracking
ALTER TABLE analyses ADD COLUMN api_calls_made INTEGER DEFAULT 0;
ALTER TABLE analyses ADD COLUMN total_tokens_used INTEGER DEFAULT 0;
ALTER TABLE analyses ADD COLUMN total_api_cost NUMERIC(10, 4) DEFAULT 0.00;
ALTER TABLE analyses ADD COLUMN rate_limit_hits INTEGER DEFAULT 0;
ALTER TABLE analyses ADD COLUMN rate_limit_retry_count INTEGER DEFAULT 0;

-- Score distribution
ALTER TABLE analyses ADD COLUMN avg_overall_score NUMERIC(5, 4);
ALTER TABLE analyses ADD COLUMN median_overall_score NUMERIC(5, 4);
ALTER TABLE analyses ADD COLUMN min_overall_score NUMERIC(5, 4);
ALTER TABLE analyses ADD COLUMN max_overall_score NUMERIC(5, 4);
ALTER TABLE analyses ADD COLUMN candidates_above_70_percent INTEGER DEFAULT 0;
ALTER TABLE analyses ADD COLUMN candidates_50_to_70_percent INTEGER DEFAULT 0;
ALTER TABLE analyses ADD COLUMN candidates_below_50_percent INTEGER DEFAULT 0;

-- AI Insights tracking
ALTER TABLE analyses ADD COLUMN insights_generated_count INTEGER DEFAULT 0;
ALTER TABLE analyses ADD COLUMN insights_generation_time_seconds INTEGER;
ALTER TABLE analyses ADD COLUMN insights_tokens_used INTEGER DEFAULT 0;
ALTER TABLE analyses ADD COLUMN insights_api_cost NUMERIC(10, 4) DEFAULT 0.00;

-- Performance metrics
ALTER TABLE analyses ADD COLUMN text_extraction_total_time INTEGER;
ALTER TABLE analyses ADD COLUMN embedding_generation_time INTEGER;
ALTER TABLE analyses ADD COLUMN similarity_calculation_time INTEGER;
ALTER TABLE analyses ADD COLUMN database_write_time INTEGER;

-- Cost breakdown
ALTER TABLE analyses ADD COLUMN cost_per_candidate NUMERIC(10, 4);
ALTER TABLE analyses ADD COLUMN cost_per_criterion NUMERIC(10, 4);
ALTER TABLE analyses ADD COLUMN cost_breakdown TEXT;  -- JSON

-- Success tracking
ALTER TABLE analyses ADD COLUMN status VARCHAR(20) DEFAULT 'completed';
ALTER TABLE analyses ADD COLUMN error_count INTEGER DEFAULT 0;
ALTER TABLE analyses ADD COLUMN error_details TEXT;  -- JSON
```

### CandidateFile Table - Individual Metrics
```sql
ALTER TABLE candidate_files ADD COLUMN resume_file_size_bytes INTEGER;
ALTER TABLE candidate_files ADD COLUMN resume_character_count INTEGER;
ALTER TABLE candidate_files ADD COLUMN resume_file_type VARCHAR(10);
ALTER TABLE candidate_files ADD COLUMN extraction_duration_seconds INTEGER;
ALTER TABLE candidate_files ADD COLUMN analysis_duration_seconds INTEGER;
ALTER TABLE candidate_files ADD COLUMN chunk_count INTEGER;
ALTER TABLE candidate_files ADD COLUMN processing_error BOOLEAN DEFAULT FALSE;
ALTER TABLE candidate_files ADD COLUMN error_message TEXT;
```

### User Table - Aggregate Stats
```sql
ALTER TABLE users ADD COLUMN total_jobs_run INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN total_candidates_analyzed INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN total_api_spend NUMERIC(10, 2) DEFAULT 0.00;
ALTER TABLE users ADD COLUMN avg_candidates_per_job NUMERIC(6, 2);
ALTER TABLE users ADD COLUMN largest_batch_processed INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN last_active_date TIMESTAMP;
```

### New Table: DocumentWarnings
```sql
CREATE TABLE document_warnings (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    warning_type VARCHAR(50) NOT NULL,  -- jd_too_long, resume_too_long, batch_too_large
    warning_shown_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_action VARCHAR(50),  -- continued, abandoned, reduced_batch, override
    document_count INTEGER,
    document_size INTEGER,
    additional_data TEXT,  -- JSON for extra context
    INDEX idx_warnings_analysis (analysis_id),
    INDEX idx_warnings_user (user_id),
    INDEX idx_warnings_type (warning_type)
);
```

### New Table: SystemHealthLog
```sql
CREATE TABLE system_health_log (
    id SERIAL PRIMARY KEY,
    log_date DATE NOT NULL UNIQUE,
    jobs_completed INTEGER DEFAULT 0,
    jobs_failed INTEGER DEFAULT 0,
    avg_processing_time INTEGER,
    total_api_spend NUMERIC(10, 2) DEFAULT 0.00,
    rate_limit_incidents INTEGER DEFAULT 0,
    error_types TEXT,  -- JSON distribution
    peak_concurrent_jobs INTEGER DEFAULT 1,
    total_candidates_processed INTEGER DEFAULT 0,
    total_api_calls INTEGER DEFAULT 0,
    INDEX idx_health_date (log_date)
);
```

---

## Phase 2: Code Instrumentation (3-4 hours)

### 2.1 Update app.py - Upload Routes

**Location**: Lines ~350-550 (JD upload, resume upload)

```python
# Capture document sizes during upload
jd_file_size = len(jd_bytes)
jd_char_count = len(jd_text)

# Store in draft
draft.jd_file_size_bytes = jd_file_size
draft.jd_character_count = jd_char_count
```

```python
# Track resume file sizes
for resume in resumes:
    resume_file_size = len(resume_bytes)
    draft_resume.file_size_bytes = resume_file_size
    draft_resume.character_count = len(extracted_text)
```

**Location**: Lines ~760-780 (Analysis creation)

```python
# Calculate aggregate resume stats before creating Analysis
total_resume_size = sum(len(c.raw_bytes) for c in candidates)
resume_char_counts = [len(c.text) for c in candidates]

analysis = Analysis(
    # ... existing fields ...
    jd_file_size_bytes=len(jd_bytes),
    jd_character_count=len(jd_text),
    total_resume_file_size_bytes=total_resume_size,
    avg_resume_character_count=int(np.mean(resume_char_counts)),
    min_resume_character_count=min(resume_char_counts),
    max_resume_character_count=max(resume_char_counts),
)
```

### 2.2 Update analysis.py - AI Pipeline

**Location**: Add timing wrapper function at top
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(metric_name):
    """Context manager for timing operations"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[METRIC] {metric_name}: {elapsed:.2f}s")
    # Return elapsed for storage
    return elapsed
```

**Location**: Wrap major operations
```python
# Text extraction timing
extraction_times = []
for candidate in candidates:
    start = time.time()
    text = extract_text(candidate.file_bytes)
    extraction_times.append(time.time() - start)

analysis.text_extraction_total_time = int(sum(extraction_times))
```

```python
# API call tracking
api_calls = 0
tokens_used = 0
rate_limits = 0

# In AI service calls:
response = await openai_client.chat.completions.create(...)
api_calls += 1
tokens_used += response.usage.total_tokens

# After rate limit retry:
rate_limits += 1
```

### 2.3 Update ai_service.py - Token Tracking

**Location**: Every OpenAI API call
```python
def track_api_call(response):
    """Track API usage for analytics"""
    return {
        'tokens': response.usage.total_tokens,
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'model': response.model,
        'cost': calculate_cost(response)
    }
```

### 2.4 Score Distribution Calculation

**Location**: After coverage matrix is built (app.py ~860-880)
```python
# Calculate score distribution
overall_scores = [eval_obj.overall_score / 100.0 for eval_obj in evaluations]

analysis.avg_overall_score = np.mean(overall_scores)
analysis.median_overall_score = np.median(overall_scores)
analysis.min_overall_score = min(overall_scores)
analysis.max_overall_score = max(overall_scores)

analysis.candidates_above_70_percent = sum(1 for s in overall_scores if s >= 0.70)
analysis.candidates_50_to_70_percent = sum(1 for s in overall_scores if 0.50 <= s < 0.70)
analysis.candidates_below_50_percent = sum(1 for s in overall_scores if s < 0.50)
```

### 2.5 Warning Event Tracking

**Location**: When showing resume limit modal (analyze.html JavaScript)
```javascript
function showResumeLimitModal(fileCount, maxLimit) {
    // Show modal...
    
    // Track warning event via AJAX
    fetch('/api/track-warning', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            warning_type: 'batch_too_large',
            document_count: fileCount,
            limit: maxLimit
        })
    });
}
```

**Location**: New API endpoint in app.py
```python
@app.route('/api/track-warning', methods=['POST'])
@login_required
def track_warning():
    """Track warning events for analytics"""
    data = request.json
    
    warning = DocumentWarning(
        user_id=current_user.id,
        warning_type=data['warning_type'],
        document_count=data['document_count'],
        document_size=data.get('document_size'),
        user_action='shown'  # Will be updated on user choice
    )
    db.session.add(warning)
    db.session.commit()
    
    return jsonify({'status': 'tracked', 'warning_id': warning.id})
```

---

## Phase 3: Enhanced Analytics Dashboard (2-3 hours)

### 3.1 Advanced Charts

**Add to admin_analytics.html:**

1. **Cost Breakdown Pie Chart**
   - Extraction vs Embeddings vs Similarity vs Insights
   - Shows where API spend goes

2. **Resume Quality Heatmap**
   - File size vs Character count vs Processing time
   - Identifies problematic resumes

3. **User Cohort Analysis**
   - New vs Returning users
   - Batch size growth over time

4. **Rate Limiting Timeline**
   - When rate limits occur (time of day)
   - Correlation with batch sizes

5. **Error Type Distribution**
   - Pie chart of error categories
   - Trend over time

### 3.2 Advanced Filters

```html
<div class="analytics-filters">
    <select id="dateRange">
        <option value="7">Last 7 days</option>
        <option value="30" selected>Last 30 days</option>
        <option value="90">Last 90 days</option>
        <option value="365">Last year</option>
    </select>
    
    <select id="userFilter">
        <option value="all">All Users</option>
        <option value="new">New Users (< 30 days)</option>
        <option value="active">Active Users</option>
        <option value="power">Power Users (10+ jobs)</option>
    </select>
    
    <select id="batchSize">
        <option value="all">All Batch Sizes</option>
        <option value="small">Small (1-50)</option>
        <option value="medium">Medium (51-150)</option>
        <option value="large">Large (151+)</option>
    </select>
</div>
```

### 3.3 Export Functionality

```python
@app.route('/admin/analytics/export/<format>')
@admin_required
def export_analytics(format):
    """Export analytics data as CSV or JSON"""
    data = gather_analytics_data()
    
    if format == 'csv':
        return send_csv(data)
    elif format == 'json':
        return jsonify(data)
```

---

## Phase 4: Automated Insights (1-2 hours)

### 4.1 Smart Alerts

```python
def generate_analytics_insights():
    """Generate automated insights from data"""
    insights = []
    
    # Check override rate trend
    recent_rate = calculate_override_rate(days=7)
    historical_rate = calculate_override_rate(days=30)
    
    if recent_rate > historical_rate * 1.5:
        insights.append({
            'type': 'warning',
            'title': 'Increasing Override Rate',
            'message': f'Override rate jumped from {historical_rate:.1f}% to {recent_rate:.1f}%',
            'action': 'Consider raising default limit or improving batch guidance'
        })
    
    # Check for rate limiting spikes
    rate_limits_today = Analysis.query.filter(
        func.date(Analysis.created_at) == datetime.utcnow().date(),
        Analysis.rate_limit_hits > 0
    ).count()
    
    if rate_limits_today > 5:
        insights.append({
            'type': 'alert',
            'title': 'Elevated Rate Limiting',
            'message': f'{rate_limits_today} jobs hit rate limits today',
            'action': 'Monitor API usage or implement request queuing'
        })
    
    return insights
```

### 4.2 Daily Health Check

```python
@app.cli.command()
def daily_health_check():
    """Run daily analytics aggregation (cron job)"""
    today = datetime.utcnow().date()
    
    # Aggregate today's data
    health = SystemHealthLog(
        log_date=today,
        jobs_completed=Analysis.query.filter(
            func.date(Analysis.created_at) == today
        ).count(),
        jobs_failed=Analysis.query.filter(
            func.date(Analysis.created_at) == today,
            Analysis.status == 'failed'
        ).count(),
        # ... other aggregations ...
    )
    
    db.session.add(health)
    db.session.commit()
    
    # Send alert if anomalies detected
    insights = generate_analytics_insights()
    if any(i['type'] == 'alert' for i in insights):
        send_admin_alert(insights)
```

---

## Phase 5: Performance Optimizations (1 hour)

### 5.1 Database Indexes

```sql
-- Speed up analytics queries
CREATE INDEX idx_analyses_completed_at ON analyses(completed_at);
CREATE INDEX idx_analyses_status ON analyses(status);
CREATE INDEX idx_analyses_processing_duration ON analyses(processing_duration_seconds);
CREATE INDEX idx_users_total_jobs ON users(total_jobs_run);
CREATE INDEX idx_users_last_active ON users(last_active_date);
```

### 5.2 Query Optimization

Use SQLAlchemy subqueries for complex aggregations:
```python
# Instead of loading all Analysis objects:
# BAD: analyses = Analysis.query.all()

# GOOD: Use aggregations
stats = db.session.query(
    func.count(Analysis.id).label('count'),
    func.avg(Analysis.num_candidates).label('avg_candidates'),
    func.sum(Analysis.cost_usd).label('total_cost')
).first()
```

### 5.3 Caching

```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=128)
def get_cached_analytics(date_key):
    """Cache analytics for performance"""
    # Regenerate cache every hour
    return generate_analytics_data()
```

---

## Phase 6: Testing & Validation (1 hour)

### 6.1 Test Scenarios

1. **Small Job (5 candidates)**
   - Verify all timings captured
   - Check score distribution calculated
   - Confirm no rate limiting

2. **Large Job (250 candidates)**
   - Warning event logged
   - Override tracked correctly
   - Rate limiting handled
   - All metrics captured

3. **Failed Job**
   - Error details stored
   - Status set correctly
   - User stats not corrupted

4. **Edge Cases**
   - Zero candidates (should fail gracefully)
   - Duplicate resumes (should be deduplicated)
   - Very large files (track size correctly)

### 6.2 Data Validation Queries

```sql
-- Check for orphaned warnings
SELECT COUNT(*) FROM document_warnings 
WHERE analysis_id NOT IN (SELECT id FROM analyses);

-- Verify timing data consistency
SELECT COUNT(*) FROM analyses 
WHERE processing_duration_seconds IS NOT NULL 
AND (completed_at IS NULL OR created_at IS NULL);

-- Check score distribution math
SELECT id, 
    candidates_above_70_percent + 
    candidates_50_to_70_percent + 
    candidates_below_50_percent as total,
    num_candidates
FROM analyses 
WHERE total != num_candidates;
```

---

## Implementation Checklist

- [ ] Run comprehensive database migration
- [ ] Update app.py upload routes with size tracking
- [ ] Add timing wrappers to analysis.py
- [ ] Instrument ai_service.py for token tracking
- [ ] Calculate score distributions after analysis
- [ ] Add warning tracking API endpoint
- [ ] Update frontend to track warning events
- [ ] Build enhanced analytics dashboard
- [ ] Add export functionality
- [ ] Implement automated insights
- [ ] Set up daily health check cron job
- [ ] Add database indexes
- [ ] Optimize queries with aggregations
- [ ] Implement caching layer
- [ ] Create comprehensive test suite
- [ ] Run validation queries
- [ ] Document all metrics for future reference

---

## Maintenance & Monitoring

### Weekly Tasks
- Review analytics insights
- Check for data quality issues
- Monitor dashboard performance

### Monthly Tasks
- Analyze trends and patterns
- Update recommendations based on data
- Archive old SystemHealthLog data

### Quarterly Tasks
- Review which metrics are most valuable
- Remove unused metrics to reduce overhead
- Add new metrics based on business needs

---

## Estimated Benefits

**Decision Making**:
- Data-driven limit adjustments
- Identify performance bottlenecks
- Understand user behavior patterns

**Cost Optimization**:
- Track API spend per feature
- Identify expensive operations
- Optimize based on actual usage

**User Experience**:
- Spot issues before users complain
- Understand pain points
- Validate UX improvements

**System Health**:
- Early warning for problems
- Capacity planning insights
- Performance regression detection

---

## Notes

- Start with Phase 1-2 for core instrumentation
- Phase 3-4 can be added incrementally
- Phase 5-6 important for production scale
- All phases build on the basic analytics already implemented

**Current Status**: Basic analytics live with 4 fields capturing essential patterns. Full implementation ready when needed.
