import json

# New two-agent configuration
new_config = {
  "_metadata": {
    "description": "Two-Agent AI Configuration for Candidate Evaluation System",
    "last_updated": "2025-12-30 00:00:00 UTC",
    "version": "2.0",
    "warning": "Changes to these settings affect both quality and cost. Test with small batches first.",
    "architecture": "RANKER_AGENT handles bulk scoring (all candidates × all criteria). INSIGHT_AGENT generates deep insights for selected candidates only."
  },
  
  "ranker_model": {
    "value": "gpt-4o-mini",
    "description": "Model for RANKER_AGENT - handles bulk scoring and criteria extraction",
    "options": [
      "gpt-4o-mini",
      "gpt-4o",
      "gpt-4-turbo",
      "gpt-4"
    ],
    "explanation": "gpt-4o-mini: RECOMMENDED for ranker. 10x cheaper than gpt-4o, sufficient quality for scoring. gpt-4o: Higher quality but expensive for bulk work. Use only if quality issues with mini.",
    "cost_impact": "gpt-4o-mini: $0.15 per 1M input, $0.60 per 1M output. gpt-4o: $2.50/$10. For 50 candidates × 20 criteria (1000 calls), mini=$0.30, gpt-4o=$3.00.",
    "quality_impact": "gpt-4o-mini provides reliable 0-100 scores and justifications. gpt-4o offers marginally better evidence extraction but rarely worth 10x cost for bulk scoring.",
    "usage": "Called (# candidates × # criteria) times per analysis. This is your highest-volume operation."
  },
  
  "insight_model": {
    "value": "gpt-4o",
    "description": "Model for INSIGHT_AGENT - generates deep insights for selected candidates",
    "options": [
      "gpt-4o",
      "gpt-4o-mini",
      "gpt-4-turbo",
      "gpt-4"
    ],
    "explanation": "gpt-4o: RECOMMENDED for insights. Best readability and personalization. gpt-4o-mini: 60% cheaper but generic phrasing. gpt-4-turbo: Same cost as gpt-4o, similar quality. gpt-4: More expensive, no benefit.",
    "cost_impact": "gpt-4o: ~$0.03 per insight (3000 tokens in, 800 tokens out). gpt-4o-mini: ~$0.01 per insight. For 5 insights, difference is $0.10.",
    "quality_impact": "gpt-4o produces professional, specific insights with natural phrasing. gpt-4o-mini works but tends toward generic language ('demonstrates strong skills...').",
    "usage": "Called only for candidates you select for 'Deep Insights' (typically top 3-10). Low volume, high value."
  },
  
  "ranker_temperature": {
    "value": 0.1,
    "description": "Temperature for RANKER_AGENT (0.0 = deterministic, 2.0 = creative)",
    "range": [0.0, 2.0],
    "explanation": "RANKER needs consistency across thousands of scoring calls. 0.1 ensures same resume+criterion always gets same score. Higher values introduce randomness.",
    "recommended": 0.1,
    "cost_impact": "No cost impact - only affects consistency",
    "quality_impact": "0.1 = highly consistent scoring. 0.3+ = scores may vary on re-run. Keep LOW for ranker."
  },
  
  "insight_temperature": {
    "value": 0.4,
    "description": "Temperature for INSIGHT_AGENT (0.0 = deterministic, 2.0 = creative)",
    "range": [0.0, 2.0],
    "explanation": "INSIGHT benefits from readability and natural phrasing. 0.4 balances consistency with engaging language. Too low (0.1) = robotic. Too high (0.8) = inconsistent.",
    "recommended": 0.4,
    "cost_impact": "No cost impact - only affects writing style",
    "quality_impact": "0.4 produces professional, readable insights without excessive creativity. Sweet spot for hiring assessments."
  },
  
  "advanced_api_settings": {
    "presence_penalty": {
      "value": 0.4,
      "description": "Reduces repetitive phrasing across insights (0.0 = no penalty, 2.0 = max penalty)",
      "range": [0.0, 2.0],
      "explanation": "Penalizes using the same phrases repeatedly across multiple candidates. Helps avoid 'demonstrates strong...' appearing in every insight.",
      "recommended": 0.4,
      "applies_to": "INSIGHT_AGENT only",
      "cost_impact": "No cost impact - only affects output style",
      "quality_impact": "0.3-0.6 produces more personalized, varied insights. Too high (>1.0) may produce awkward phrasing."
    },
    "frequency_penalty": {
      "value": 0.3,
      "description": "Discourages repeating words within a single response (0.0 = no penalty, 2.0 = max penalty)",
      "range": [0.0, 2.0],
      "explanation": "Reduces word repetition within each insight. Prevents 'Python experience... experienced with Python... Python skills' redundancy.",
      "recommended": 0.3,
      "applies_to": "INSIGHT_AGENT only",
      "cost_impact": "No cost impact - only affects output style",
      "quality_impact": "0.2-0.4 produces more natural, varied language. Too high (>0.8) may lose important keyword emphasis."
    },
    "ranker_max_tokens": {
      "value": 300,
      "description": "Max tokens for RANKER responses (score + justification + evidence)",
      "range": [200, 500],
      "explanation": "RANKER returns: score (integer), justification (1 sentence), raw_evidence (verbatim quotes). 300 tokens = ~225 words, sufficient for detailed evidence.",
      "recommended": 300,
      "cost_impact": "Each token costs $0.0000006 (gpt-4o-mini). 300 tokens = $0.00018 per call. For 1000 calls, difference between 200 and 300 tokens is $0.06.",
      "quality_impact": "300 tokens ensures complete evidence extraction. 200 may truncate quotes. 500 is overkill and wastes money."
    },
    "insight_max_tokens": {
      "value": 1000,
      "description": "Max tokens for INSIGHT responses (strengths + gaps + notes)",
      "range": [500, 2000],
      "explanation": "INSIGHT returns: 3-6 strength bullets, 3-6 gap bullets, 2-4 sentence notes. Typical output: 400-600 tokens. 1000 provides safety margin.",
      "recommended": 1000,
      "cost_impact": "Each token costs $0.00001 (gpt-4o). 1000 tokens = $0.01 per insight. 2000 tokens = $0.02. We typically only use ~500.",
      "quality_impact": "Too low (<500) may truncate insights. 1000 is sufficient for detailed output. 2000 wastes money with no benefit."
    }
  },
  
  "score_thresholds": {
    "high_threshold": {
      "value": 0.75,
      "description": "Score ≥75% = 'Strong Match' (green in UI)",
      "range": [0.6, 0.9],
      "explanation": "Criteria scoring 75+ are highlighted as strengths. Lower threshold = more 'strengths'. Higher = more selective.",
      "recommended": 0.75
    },
    "low_threshold": {
      "value": 0.35,
      "description": "Score <35% = 'Weak Match' (red in UI)",
      "range": [0.2, 0.5],
      "explanation": "Criteria scoring below 35 are flagged as gaps. Lower threshold = fewer gaps. Higher = more critical assessment.",
      "recommended": 0.35
    }
  },
  
  "pricing_config": {
    "standard_tier_price": {
      "value": 4.0,
      "description": "Price for Standard tier analysis (full scoring + top 3 insights)",
      "explanation": "Covers: JD extraction, scoring all candidates × all criteria, generating 3 insights. Includes profit margin over OpenAI costs."
    },
    "deep_dive_unlock_price": {
      "value": 1.0,
      "description": "Price to unlock Deep Insights for one additional candidate",
      "explanation": "Each additional insight costs ~$0.03 in OpenAI fees. $1.00 provides healthy profit margin while staying affordable."
    },
    "model_costs": {
      "gpt-4o": {
        "input_per_million": 2.5,
        "output_per_million": 10.0
      },
      "gpt-4o-mini": {
        "input_per_million": 0.15,
        "output_per_million": 0.6
      },
      "gpt-4-turbo": {
        "input_per_million": 10.0,
        "output_per_million": 30.0
      },
      "gpt-4": {
        "input_per_million": 30.0,
        "output_per_million": 60.0
      }
    }
  },
  
  "_usage_notes": {
    "current_architecture": "TWO-AGENT SYSTEM: RANKER (bulk scoring) + INSIGHT (deep analysis for selected candidates)",
    "typical_analysis_cost": "50 candidates, 20 criteria: RANKER=$0.30 (1000 calls), INSIGHT=$0.15 (5 insights) = $0.45 total. Revenue=$4.00. Profit=$3.55 (79% margin)",
    "recommended_workflow": "1. Test changes with small job (10 candidates). 2. Check quality in Insights page. 3. Monitor costs in Stats page. 4. Adjust gradually.",
    "optimization_priority": "1. Keep RANKER=gpt-4o-mini (already optimized). 2. Keep INSIGHT=gpt-4o (quality matters here). 3. Adjust temperatures only if quality issues. 4. Don't change max_tokens unless truncation occurs.",
    "cost_saving_tips": "For budget jobs: Use gpt-4o-mini for both agents (saves 60% on insights). Reduce ranker_max_tokens to 200 (saves 33%). Skip Deep Insights for weak candidates.",
    "quality_improvement_tips": "If scores seem inconsistent: Lower ranker_temperature to 0.05. If insights seem generic: Increase insight_temperature to 0.5. If evidence incomplete: Raise ranker_max_tokens to 400."
  }
}

# Write to file
with open('flask_app/config/gpt_settings.json', 'w', encoding='utf-8') as f:
    json.dump(new_config, f, indent=2, ensure_ascii=False)

print("✅ gpt_settings.json updated successfully with two-agent architecture!")
