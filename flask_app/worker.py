"""
Background worker for processing analysis jobs
Polls database for pending jobs and processes them sequentially
"""

import time
import sys
import os
import json
import asyncio

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass
from datetime import datetime, timezone
from decimal import Decimal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from database import db, JobQueue, Draft, DraftResume, Analysis, Transaction, CandidateFile, User
from ai_service import run_global_ranking, run_deep_insights
from analysis import Candidate, load_gpt_settings
from config import Config
from email_utils import send_email

# Create Flask app context
app = create_app()

def send_completion_email(user, analysis, job):
    """Send email notification when job completes"""
    try:
        subject = f"✅ Analysis Complete - {analysis.job_title}"
        
        # Calculate processing time
        processing_time = "Unknown"
        if job.started_at and job.completed_at:
            delta = job.completed_at - job.started_at
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            if hours > 0:
                processing_time = f"{hours}h {minutes}m"
            else:
                processing_time = f"{minutes} minutes"
        
        # Get top 3 candidates
        coverage_data = json.loads(analysis.coverage_data)
        sorted_candidates = sorted(coverage_data, key=lambda x: x.get('Overall', 0), reverse=True)
        top_3 = sorted_candidates[:3]
        
        # Build HTML for top candidates
        top_candidates_html = ""
        for i, c in enumerate(top_3):
            score = int(c.get('Overall', 0) * 100)
            # Color based on score
            if score >= 80:
                color = "#10b981"  # Green
                badge_bg = "#d1fae5"
            elif score >= 60:
                color = "#f59e0b"  # Orange
                badge_bg = "#fef3c7"
            else:
                color = "#6b7280"  # Gray
                badge_bg = "#f3f4f6"
            
            top_candidates_html += f"""
                <tr>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                        <div style="font-weight: 600; color: #1f2937; margin-bottom: 4px;">{i+1}. {c['Candidate']}</div>
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: right;">
                        <span style="background: {badge_bg}; color: {color}; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 14px;">{score}/100</span>
                    </td>
                </tr>
            """
        
        # HTML email body
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f9fafb;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f9fafb; padding: 40px 20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); overflow: hidden;">
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 32px; text-align: center;">
                            <div style="font-size: 48px; margin-bottom: 8px;">✅</div>
                            <h1 style="margin: 0; color: white; font-size: 28px; font-weight: 700;">Analysis Complete!</h1>
                        </td>
                    </tr>
                    
                    <!-- Greeting -->
                    <tr>
                        <td style="padding: 32px;">
                            <p style="margin: 0 0 24px 0; font-size: 16px; color: #374151; line-height: 1.6;">
                                Hi <strong>{user.name or user.email.split('@')[0]}</strong>,
                            </p>
                            <p style="margin: 0 0 24px 0; font-size: 16px; color: #374151; line-height: 1.6;">
                                Your candidate analysis is ready to view! 🎉
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Job Details -->
                    <tr>
                        <td style="padding: 0 32px 24px 32px;">
                            <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f9fafb; border-radius: 8px; padding: 20px;">
                                <tr>
                                    <td style="padding: 8px 0;">
                                        <span style="color: #6b7280; font-size: 14px;">Job Title</span><br>
                                        <strong style="color: #1f2937; font-size: 16px;">{analysis.job_title}</strong>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0;">
                                        <table width="100%" cellpadding="0" cellspacing="0">
                                            <tr>
                                                <td width="50%" style="padding-right: 10px;">
                                                    <span style="color: #6b7280; font-size: 14px;">Candidates Analyzed</span><br>
                                                    <strong style="color: #1f2937; font-size: 16px;">{analysis.num_candidates}</strong>
                                                </td>
                                                <td width="50%">
                                                    <span style="color: #6b7280; font-size: 14px;">Processing Time</span><br>
                                                    <strong style="color: #1f2937; font-size: 16px;">{processing_time}</strong>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0;">
                                        <span style="color: #6b7280; font-size: 14px;">Cost</span><br>
                                        <strong style="color: #1f2937; font-size: 16px;">${analysis.cost_usd:.2f}</strong>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Top Candidates -->
                    <tr>
                        <td style="padding: 0 32px 24px 32px;">
                            <h2 style="margin: 0 0 16px 0; color: #1f2937; font-size: 20px; font-weight: 700;">🏆 Top Candidates</h2>
                            <table width="100%" cellpadding="0" cellspacing="0" style="border-radius: 8px; overflow: hidden; border: 1px solid #e5e7eb;">
                                {top_candidates_html}
                            </table>
                        </td>
                    </tr>
                    
                    <!-- CTA Button -->
                    <tr>
                        <td style="padding: 0 32px 32px 32px; text-align: center;">
                            <a href="https://candidateevaluator.com/results/{analysis.id}" 
                               style="display: inline-block; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; text-decoration: none; padding: 16px 32px; border-radius: 8px; font-weight: 600; font-size: 16px; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">
                                📊 View Full Results
                            </a>
                        </td>
                    </tr>
                    
                    <!-- Balance Info -->
                    <tr>
                        <td style="padding: 0 32px 32px 32px;">
                            <div style="background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 4px;">
                                <p style="margin: 0; color: #1e40af; font-size: 14px;">
                                    💰 <strong>Your remaining balance:</strong> ${user.balance_usd:.2f}
                                </p>
                            </div>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px 32px; background-color: #f9fafb; border-top: 1px solid #e5e7eb; text-align: center;">
                            <p style="margin: 0 0 8px 0; color: #6b7280; font-size: 14px;">
                                Thanks for using <strong style="color: #1f2937;">CandidateEvaluator</strong>!
                            </p>
                            <p style="margin: 0; color: #9ca3af; font-size: 12px;">
                                Need help? Visit our <a href="https://candidateevaluator.com/help" style="color: #3b82f6; text-decoration: none;">Help Center</a>
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
        
        # Plain text fallback
        text_body = f"""Hi {user.name or user.email},

Your analysis is ready!

Job Details:
- Job Title: {analysis.job_title}
- Candidates Analyzed: {analysis.num_candidates}
- Processing Time: {processing_time}
- Cost: ${analysis.cost_usd:.2f}

Top Candidates:
{chr(10).join([f"{i+1}. {c['Candidate']} - {int(c.get('Overall', 0) * 100)}/100" for i, c in enumerate(top_3)])}

View full results: https://candidateevaluator.com/analysis/results/{analysis.id}

Your remaining balance: ${user.balance_usd:.2f}

Thanks for using CandidateEvaluator!
"""
        
        email_sent = send_email(
            subject=subject,
            recipients=[user.email],
            html_body=html_body,
            text_body=text_body
        )
        
        if email_sent:
            print(f"✅ Sent completion email to {user.email}")
        
    except Exception as e:
        print(f"⚠️  Failed to send completion email: {str(e)}")


def send_failure_email(user, job, error_message):
    """Send email notification when job fails"""
    try:
        subject = f"❌ Analysis Failed - Job #{job.id}"
        
        body = f"""Hi {user.name or user.email},

Unfortunately, your analysis job failed to complete.

Job Details:
- Job ID: #{job.id}
- Status: Failed
- Error: {error_message[:200]}

You have NOT been charged for this failed analysis.

What to try:
1. Check if any resumes have unusual formatting
2. Try with a smaller batch size
3. Contact support if the issue persists

View your jobs: https://candidateevaluator.com/dashboard/jobs

Sorry for the inconvenience!
CandidateEvaluator Team
"""
        
        send_email(
            subject=subject,
            recipients=[user.email],
            html_body=body,
            text_body=body
        )
        print(f"✅ Sent failure email to {user.email}")
        
    except Exception as e:
        print(f"⚠️  Failed to send failure email: {str(e)}")


def process_job(job):
    """Process a single analysis job"""
    print(f"\n{'='*60}")
    print(f"🔄 Processing Job #{job.id}")
    print(f"   User: {job.user.email}")
    print(f"   Created: {job.created_at}")
    print(f"{'='*60}\n")
    
    try:
        # Mark as processing
        job.status = 'processing'
        job.started_at = datetime.now(timezone.utc)
        db.session.commit()
        
        # Load draft data
        draft = job.draft
        if not draft or not draft.criteria_data:
            raise Exception("Draft or criteria data not found")
        
        # Load resumes
        draft_resumes = DraftResume.query.filter_by(draft_id=draft.id).all()
        if not draft_resumes:
            raise Exception("No resumes found in draft")
        
        job.total = len(draft_resumes)
        db.session.commit()
        
        print(f"📊 Job Configuration:")
        print(f"   Job Title: {draft.job_title}")
        print(f"   Resumes: {len(draft_resumes)}")
        print(f"   Insights Mode: {job.insights_mode}")
        print(f"")
        
        # Convert to Candidate objects
        candidates = []
        for dr in draft_resumes:
            candidates.append(Candidate(
                name=dr.candidate_name,
                file_name=dr.file_name,
                text=dr.extracted_text,
                hash=dr.file_hash,
                raw_bytes=dr.file_bytes
            ))
        
        # Get configuration
        jd_text = draft.jd_text
        criteria_list = json.loads(draft.criteria_data)
        criteria = [c['criterion'] for c in criteria_list if c.get('use', True)]
        job_title = draft.job_title or "Position Not Specified"
        
        # Calculate cost - EXACT SAME LOGIC AS routes.py
        pricing = Config.get_pricing()
        standard_price = Decimal(str(pricing['BASE_ANALYSIS_PRICE']))
        deep_dive_price = Decimal(str(pricing['DEEP_DIVE_PRICE']))
        individual_insight_price = Decimal(str(pricing['EXTRA_INSIGHT_PRICE']))
        
        num_candidates = len(candidates)
        insights_mode = job.insights_mode or 'standard'
        
        # Initialize cost variable
        estimated_cost = standard_price  # Default to standard tier
        
        # NEW PRICING MODEL: Calculate cost based on tier
        if insights_mode == 'standard':
            # Standard: base price (includes Top 5 insights)
            estimated_cost = standard_price
            num_insights = min(5, num_candidates)
        elif insights_mode == 'deep_dive':
            # Deep Dive: standard + deep dive price (includes Top 15 insights)
            estimated_cost = standard_price + deep_dive_price
            num_insights = min(15, num_candidates)
        elif insights_mode == 'full_radar':
            # Full Radar: standard price + individual price per candidate beyond 5
            extra_candidates = max(0, num_candidates - 5)
            estimated_cost = standard_price + (individual_insight_price * extra_candidates)
            num_insights = num_candidates
        # Legacy support for old values (map to new tiers)
        elif insights_mode in ['top3', 'top5']:
            estimated_cost = standard_price
            num_insights = min(5, num_candidates)
        elif insights_mode == 'top10':
            estimated_cost = standard_price + deep_dive_price
            num_insights = min(15, num_candidates)
        elif insights_mode == 'all':
            extra_candidates = max(0, num_candidates - 5)
            estimated_cost = standard_price + (individual_insight_price * extra_candidates)
            num_insights = num_candidates
        else:
            estimated_cost = standard_price
            num_insights = 0
        
        # Check funds
        if job.user.balance_usd < estimated_cost:
            raise Exception(f"Insufficient funds: need ${estimated_cost:.2f}, have ${job.user.balance_usd:.2f}")
        
        print(f"💰 Cost: ${estimated_cost:.2f}")
        print(f"💳 User balance: ${job.user.balance_usd:.2f}")
        print(f"")
        
        # Track start time
        analysis_start_time = datetime.now(timezone.utc)
        
        # Calculate document metrics
        jd_char_count = len(jd_text)
        resume_char_counts = [len(c.text) for c in candidates]
        avg_resume_chars = int(sum(resume_char_counts) / len(resume_char_counts)) if resume_char_counts else 0
        min_resume_chars = min(resume_char_counts) if resume_char_counts else 0
        max_resume_chars = max(resume_char_counts) if resume_char_counts else 0
        
        # Create Analysis record
        analysis = Analysis(
            user_id=job.user_id,
            job_title=job_title,
            job_description_text=jd_text[:5000],
            jd_full_text=jd_text,
            jd_filename=draft.jd_filename,
            jd_bytes=draft.jd_bytes,
            num_candidates=len(candidates),
            num_criteria=len(criteria),
            coverage_data='',
            insights_data='',
            evidence_data='',
            criteria_list=json.dumps(criteria_list),
            cost_usd=estimated_cost,
            analysis_size='phase1',
            resumes_processed=0,
            jd_character_count=jd_char_count,
            avg_resume_character_count=avg_resume_chars,
            min_resume_character_count=min_resume_chars,
            max_resume_character_count=max_resume_chars
        )
        db.session.add(analysis)
        db.session.flush()
        db.session.commit()
        
        print(f"📝 Created Analysis #{analysis.id}")
        print(f"")
        
        # Progress callback
        def update_progress(completed, total):
            job.progress = completed
            db.session.commit()
            print(f"Progress: {completed}/{total} candidates scored ({int(completed/total*100)}%)")
        
        # PHASE 1: Global Ranking
        print(f"🤖 Phase 1: AI scoring all candidates...")
        candidate_tuples = [(c.name, c.text) for c in candidates]
        
        gpt_settings = load_gpt_settings()
        jd_text_limit = gpt_settings.get('jd_text_chars', 15000)
        
        criteria_for_ai = [c for c in criteria_list if c.get('use', True)]
        
        evaluations = asyncio.run(
            run_global_ranking(
                candidates=candidate_tuples,
                jd_text=jd_text[:jd_text_limit],
                criteria=criteria_for_ai,
                progress_callback=update_progress
            )
        )
        
        print(f"✅ Phase 1 complete")
        print(f"")
        
        # PHASE 2: Deep Insights
        insights_data = {}
        gpt_candidates_list = []
        
        if num_insights > 0:
            print(f"🌟 Phase 2: Generating deep insights for top {num_insights}...")
            analysis.analysis_size = 'phase2'
            db.session.commit()
            
            insights_data = asyncio.run(
                run_deep_insights(
                    candidates=candidate_tuples,
                    jd_text=jd_text,
                    evaluations=evaluations,
                    top_n=num_insights
                )
            )
            
            gpt_candidates_list = list(insights_data.keys())
            
            # Overwrite Phase 1 justifications with Phase 2 refined versions
            for candidate_name, insights in insights_data.items():
                eval_obj = next((e for e in evaluations if e.candidate_name == candidate_name), None)
                if eval_obj and 'justifications' in insights:
                    for criterion, refined_justification in insights['justifications'].items():
                        score_obj = next((s for s in eval_obj.criterion_scores if s.criterion == criterion), None)
                        if score_obj:
                            score_obj.justification = refined_justification
            
            analysis.analysis_size = 'complete'
            db.session.commit()
            print(f"✅ Phase 2 complete")
            print(f"")
        else:
            analysis.analysis_size = 'complete'
            db.session.commit()
        
        # Build coverage DataFrame
        import pandas as pd
        coverage_records = []
        evidence_map = {}
        
        for eval_obj in evaluations:
            row = {"Candidate": eval_obj.candidate_name}
            for score in eval_obj.criterion_scores:
                row[score.criterion] = score.score / 100.0
                evidence_map[(eval_obj.candidate_name, score.criterion)] = (
                    score.raw_evidence,
                    score.justification,
                    score.score / 100.0,
                    1
                )
            row["Overall"] = eval_obj.overall_score / 100.0
            coverage_records.append(row)
        
        coverage = pd.DataFrame(coverage_records)
        category_map = {c['criterion']: c.get('category', 'Other Requirements') for c in criteria_list if c.get('use', True)}
        
        # Update analysis with results
        analysis.coverage_data = coverage.to_json(orient='records')
        analysis.insights_data = json.dumps(insights_data)
        analysis.evidence_data = json.dumps({f"{k[0]}|||{k[1]}": v for k, v in evidence_map.items()})
        analysis.category_map = json.dumps(category_map)
        analysis.gpt_candidates = json.dumps(gpt_candidates_list)
        analysis.completed_at = datetime.now(timezone.utc)
        analysis.processing_duration_seconds = int((analysis.completed_at - analysis_start_time).total_seconds())
        
        # Store candidate files
        for candidate in candidates:
            candidate_file = CandidateFile(
                analysis_id=analysis.id,
                candidate_name=candidate.name,
                file_name=candidate.file_name,
                file_bytes=candidate.raw_bytes,
                extracted_text=candidate.text
            )
            db.session.add(candidate_file)
        
        # Charge user
        job.user.balance_usd -= estimated_cost
        
        # Create transaction
        transaction = Transaction(
            user_id=job.user_id,
            amount_usd=-estimated_cost,
            transaction_type='debit',
            description=f'[Job #{analysis.id:04d}] - {job_title}',
            analysis_id=analysis.id
        )
        db.session.add(transaction)
        
        # Update user analytics
        job.user.total_analyses_count += 1
        job.user.total_revenue_usd += estimated_cost
        
        # Mark job complete
        job.status = 'completed'
        job.completed_at = datetime.now(timezone.utc)
        job.analysis_id = analysis.id
        
        db.session.commit()
        
        print(f"")
        print(f"✅ Job #{job.id} COMPLETED successfully")
        print(f"   Analysis ID: #{analysis.id}")
        print(f"   Duration: {analysis.processing_duration_seconds}s")
        print(f"   Cost: ${estimated_cost:.2f}")
        print(f"")
        
        # Send email notification
        send_completion_email(job.user, analysis, job)
        
        return True
        
    except Exception as e:
        # Job failed - rollback and save error
        db.session.rollback()
        
        error_msg = str(e)
        print(f"")
        print(f"❌ Job #{job.id} FAILED")
        print(f"   Error: {error_msg}")
        print(f"")
        
        job.status = 'failed'
        job.completed_at = datetime.now(timezone.utc)
        job.error_message = error_msg
        db.session.commit()
        
        # Send failure notification
        send_failure_email(job.user, job, error_msg)
        
        return False


def main():
    """Main worker loop"""
    print(f"")
    print(f"{'='*60}")
    print(f"🚀 Background Worker Starting")
    print(f"{'='*60}")
    print(f"")
    
    with app.app_context():
        while True:
            try:
                # Find next pending job
                job = JobQueue.query.filter_by(status='pending')\
                                   .order_by(JobQueue.created_at)\
                                   .first()
                
                if job:
                    process_job(job)
                else:
                    # No pending jobs - sleep for a bit
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Worker shutting down...")
                break
            except Exception as e:
                print(f"\n❌ Worker error: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(5)


if __name__ == '__main__':
    main()
