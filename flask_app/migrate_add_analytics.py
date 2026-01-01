"""
Migration Script: Add Comprehensive Analytics & Tracking
=========================================================
Adds extensive tracking fields to existing tables and creates new analytics tables.

This enables detailed analysis of:
- Job processing metrics (volume, timing, costs)
- Document size and quality patterns
- User behavior and warning interactions
- System health and performance
- API usage and rate limiting

Run this after backing up your database.
"""

from app import app
from database import db
from sqlalchemy import text
import sys

def run_migration():
    """Execute all migration steps"""
    with app.app_context():
        print("\n" + "="*70)
        print("🔧 ANALYTICS MIGRATION - Adding Comprehensive Tracking")
        print("="*70 + "\n")
        
        try:
            # STEP 1: Add fields to Analysis table
            print("📊 Step 1: Enhancing Analysis table...")
            analysis_fields = [
                # Processing timing
                ("completed_at", "TIMESTAMP"),
                ("processing_duration_seconds", "INTEGER"),
                
                # Document metrics
                ("jd_file_size_bytes", "INTEGER"),
                ("jd_character_count", "INTEGER"),
                ("total_resume_file_size_bytes", "BIGINT"),
                ("avg_resume_character_count", "INTEGER"),
                ("min_resume_character_count", "INTEGER"),
                ("max_resume_character_count", "INTEGER"),
                
                # Batch warning tracking
                ("exceeded_resume_limit", "BOOLEAN DEFAULT FALSE"),
                ("user_chose_override", "BOOLEAN DEFAULT FALSE"),
                ("batch_size_warning_shown", "BOOLEAN DEFAULT FALSE"),
                
                # API usage metrics
                ("api_calls_made", "INTEGER DEFAULT 0"),
                ("total_tokens_used", "INTEGER DEFAULT 0"),
                ("total_api_cost", "NUMERIC(10, 4) DEFAULT 0.00"),
                ("rate_limit_hits", "INTEGER DEFAULT 0"),
                ("rate_limit_retry_count", "INTEGER DEFAULT 0"),
                
                # Success metrics
                ("status", "VARCHAR(20) DEFAULT 'completed'"),  # completed, failed, partial_failure
                ("error_count", "INTEGER DEFAULT 0"),
                ("error_details", "TEXT"),  # JSON
                
                # Score distribution
                ("avg_overall_score", "NUMERIC(5, 4)"),
                ("median_overall_score", "NUMERIC(5, 4)"),
                ("min_overall_score", "NUMERIC(5, 4)"),
                ("max_overall_score", "NUMERIC(5, 4)"),
                ("candidates_above_70_percent", "INTEGER DEFAULT 0"),
                ("candidates_50_to_70_percent", "INTEGER DEFAULT 0"),
                ("candidates_below_50_percent", "INTEGER DEFAULT 0"),
                
                # AI Insights tracking
                ("insights_generated_count", "INTEGER DEFAULT 0"),
                ("insights_generation_time_seconds", "INTEGER"),
                ("insights_tokens_used", "INTEGER DEFAULT 0"),
                ("insights_api_cost", "NUMERIC(10, 4) DEFAULT 0.00"),
                
                # Performance metrics
                ("text_extraction_total_time", "INTEGER"),  # seconds
                ("embedding_generation_time", "INTEGER"),   # seconds
                ("similarity_calculation_time", "INTEGER"), # seconds
                ("database_write_time", "INTEGER"),         # seconds
                
                # Cost breakdown
                ("cost_per_candidate", "NUMERIC(10, 4)"),
                ("cost_per_criterion", "NUMERIC(10, 4)"),
                ("cost_breakdown", "TEXT"),  # JSON with detailed breakdown
            ]
            
            for field_name, field_type in analysis_fields:
                try:
                    db.session.execute(text(
                        f"ALTER TABLE analyses ADD COLUMN {field_name} {field_type};"
                    ))
                    print(f"  ✅ Added: analyses.{field_name}")
                except Exception as e:
                    if "already exists" in str(e) or "duplicate column" in str(e):
                        print(f"  ⏭️  Skip: analyses.{field_name} (already exists)")
                    else:
                        print(f"  ❌ Error adding analyses.{field_name}: {e}")
            
            db.session.commit()
            print("✅ Analysis table enhanced\n")
            
            # STEP 2: Add fields to CandidateFile table (represents individual candidates)
            print("📄 Step 2: Enhancing CandidateFile table...")
            candidate_fields = [
                ("resume_file_size_bytes", "INTEGER"),
                ("resume_character_count", "INTEGER"),
                ("resume_file_type", "VARCHAR(10)"),  # pdf, docx, doc, txt
                ("extraction_duration_seconds", "INTEGER"),
                ("analysis_duration_seconds", "INTEGER"),
                ("chunk_count", "INTEGER"),
                ("processing_error", "BOOLEAN DEFAULT FALSE"),
                ("error_message", "TEXT"),
            ]
            
            for field_name, field_type in candidate_fields:
                try:
                    db.session.execute(text(
                        f"ALTER TABLE candidate_files ADD COLUMN {field_name} {field_type};"
                    ))
                    print(f"  ✅ Added: candidate_files.{field_name}")
                except Exception as e:
                    if "already exists" in str(e) or "duplicate column" in str(e):
                        print(f"  ⏭️  Skip: candidate_files.{field_name} (already exists)")
                    else:
                        print(f"  ❌ Error adding candidate_files.{field_name}: {e}")
            
            db.session.commit()
            print("✅ CandidateFile table enhanced\n")
            
            # STEP 3: Add fields to User table
            print("👤 Step 3: Enhancing User table...")
            user_fields = [
                ("total_jobs_run", "INTEGER DEFAULT 0"),
                ("total_candidates_analyzed", "INTEGER DEFAULT 0"),
                ("total_api_spend", "NUMERIC(10, 2) DEFAULT 0.00"),
                ("avg_candidates_per_job", "NUMERIC(6, 2)"),
                ("largest_batch_processed", "INTEGER DEFAULT 0"),
                ("last_active_date", "TIMESTAMP"),
            ]
            
            for field_name, field_type in user_fields:
                try:
                    db.session.execute(text(
                        f"ALTER TABLE users ADD COLUMN {field_name} {field_type};"
                    ))
                    print(f"  ✅ Added: users.{field_name}")
                except Exception as e:
                    if "already exists" in str(e) or "duplicate column" in str(e):
                        print(f"  ⏭️  Skip: users.{field_name} (already exists)")
                    else:
                        print(f"  ❌ Error adding users.{field_name}: {e}")
            
            db.session.commit()
            print("✅ User table enhanced\n")
            
            # STEP 4: Create DocumentWarnings table
            print("⚠️  Step 4: Creating DocumentWarnings table...")
            try:
                db.session.execute(text("""
                    CREATE TABLE IF NOT EXISTS document_warnings (
                        id SERIAL PRIMARY KEY,
                        analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        warning_type VARCHAR(50) NOT NULL,
                        warning_shown_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        user_action VARCHAR(50),
                        document_count INTEGER,
                        document_size INTEGER,
                        additional_data TEXT,
                        INDEX idx_warnings_analysis (analysis_id),
                        INDEX idx_warnings_user (user_id),
                        INDEX idx_warnings_type (warning_type),
                        INDEX idx_warnings_shown_at (warning_shown_at)
                    );
                """))
                print("  ✅ DocumentWarnings table created")
            except Exception as e:
                if "already exists" in str(e):
                    print("  ⏭️  Skip: DocumentWarnings table (already exists)")
                else:
                    print(f"  ❌ Error creating DocumentWarnings: {e}")
            
            db.session.commit()
            print("✅ DocumentWarnings ready\n")
            
            # STEP 5: Create SystemHealthLog table
            print("🏥 Step 5: Creating SystemHealthLog table...")
            try:
                db.session.execute(text("""
                    CREATE TABLE IF NOT EXISTS system_health_log (
                        id SERIAL PRIMARY KEY,
                        log_date DATE NOT NULL UNIQUE,
                        jobs_completed INTEGER DEFAULT 0,
                        jobs_failed INTEGER DEFAULT 0,
                        avg_processing_time INTEGER,
                        total_api_spend NUMERIC(10, 2) DEFAULT 0.00,
                        rate_limit_incidents INTEGER DEFAULT 0,
                        error_types TEXT,
                        peak_concurrent_jobs INTEGER DEFAULT 1,
                        total_candidates_processed INTEGER DEFAULT 0,
                        total_api_calls INTEGER DEFAULT 0,
                        INDEX idx_health_date (log_date)
                    );
                """))
                print("  ✅ SystemHealthLog table created")
            except Exception as e:
                if "already exists" in str(e):
                    print("  ⏭️  Skip: SystemHealthLog table (already exists)")
                else:
                    print(f"  ❌ Error creating SystemHealthLog: {e}")
            
            db.session.commit()
            print("✅ SystemHealthLog ready\n")
            
            # STEP 6: Create indexes for better query performance
            print("🚀 Step 6: Creating performance indexes...")
            indexes = [
                ("idx_analyses_completed_at", "analyses", "completed_at"),
                ("idx_analyses_status", "analyses", "status"),
                ("idx_analyses_processing_duration", "analyses", "processing_duration_seconds"),
                ("idx_users_total_jobs", "users", "total_jobs_run"),
                ("idx_users_last_active", "users", "last_active_date"),
            ]
            
            for idx_name, table, column in indexes:
                try:
                    db.session.execute(text(
                        f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column});"
                    ))
                    print(f"  ✅ Index created: {idx_name}")
                except Exception as e:
                    if "already exists" in str(e):
                        print(f"  ⏭️  Skip: {idx_name} (already exists)")
                    else:
                        print(f"  ❌ Error creating {idx_name}: {e}")
            
            db.session.commit()
            print("✅ Performance indexes ready\n")
            
            print("="*70)
            print("✅ MIGRATION COMPLETE!")
            print("="*70)
            print("\n📊 Analytics tracking is now active!")
            print("🎯 Data will be automatically captured on all new jobs")
            print("📈 Visit /admin/analytics to view comprehensive stats\n")
            
        except Exception as e:
            print(f"\n❌ MIGRATION FAILED: {e}")
            print("Rolling back changes...")
            db.session.rollback()
            sys.exit(1)


if __name__ == '__main__':
    print("\n⚠️  WARNING: This will modify your database schema!")
    print("Make sure you have a recent backup before proceeding.\n")
    
    response = input("Continue with migration? (yes/no): ").strip().lower()
    if response == 'yes':
        run_migration()
    else:
        print("\n❌ Migration cancelled by user")
        sys.exit(0)
