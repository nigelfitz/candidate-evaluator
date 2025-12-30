# Generate new admin_gpt.html with two-agent architecture UI

html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT Settings - Admin Panel</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .admin-container { max-width: 1200px; margin: 0 auto; }
        .settings-card { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); padding: 30px; margin-bottom: 20px; }
        .setting-item { border-bottom: 1px solid #f0f0f0; padding: 20px 0; }
        .setting-item:last-child { border-bottom: none; }
        .setting-name { font-weight: 700; color: #1a1a1a; font-size: 1.15rem; margin-bottom: 12px; }
        .setting-value { font-family: 'Courier New', monospace; background: #667eea; color: white; padding: 6px 14px; border-radius: 6px; display: inline-block; font-weight: 600; }
        .setting-desc { color: #555; margin-top: 10px; font-size: 0.95rem; line-height: 1.5; }
        .setting-explanation { background: #f0f4ff; padding: 14px 16px; border-radius: 8px; margin-top: 12px; font-size: 0.9rem; border-left: 3px solid #667eea; }
        .agent-badge { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 8px; }
        .ranker-badge { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); }
        .insight-badge { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
        .profit-card { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 25px; border-radius: 15px; margin-bottom: 20px; }
        .profit-card h3 { margin: 0; font-size: 1.4rem; }
        .profit-value { font-size: 3rem; font-weight: bold; margin: 15px 0; }
        .profit-breakdown { background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; margin-top: 15px; }
        .profit-breakdown div { margin: 5px 0; font-size: 0.95rem; }
        .section-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 18px 24px; border-radius: 12px; margin-bottom: 0; cursor: pointer; display: flex; justify-content: space-between; align-items: center; transition: all 0.3s; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
        .section-header:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4); }
        .section-header h3 { margin: 0; font-size: 1.3rem; font-weight: 600; }
        .section-header .toggle-icon { font-size: 1.5rem; transition: transform 0.3s; }
        .section-header.collapsed .toggle-icon { transform: rotate(-90deg); }
        .section-content { padding: 25px; border: 2px solid #f0f0f0; border-top: none; border-radius: 0 0 12px 12px; background: white; }
        .section-content.collapsed { display: none; }
        .section-badge { background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 12px; }
        .warning { background: #f8d7da; padding: 12px; border-radius: 8px; border-left: 4px solid #dc3545; margin-bottom: 20px; }
        .success-msg { background: #d4edda; padding: 12px; border-radius: 8px; border-left: 4px solid #28a745; margin-bottom: 20px; }
        .btn-save { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; padding: 12px 30px; font-size: 1.1rem; }
        .btn-save:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        .form-range { margin-top: 10px; }
        .range-labels { display: flex; justify-content: space-between; font-size: 0.85rem; color: #666; margin-top: 5px; }
        .logout-btn { position: fixed; top: 20px; right: 20px; z-index: 1000; }
        .admin-tabs { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); padding: 15px 30px; margin-bottom: 20px; }
        .admin-tabs a { color: #666; text-decoration: none; padding: 12px 24px; display: inline-block; border-bottom: 3px solid transparent; font-weight: 500; transition: all 0.3s; }
        .admin-tabs a:hover { color: #667eea; }
        .admin-tabs a.active { color: #667eea; border-bottom-color: #667eea; }
        .session-info { position: fixed; top: 20px; right: 180px; background: rgba(255,255,255,0.95); padding: 8px 16px; border-radius: 8px; font-size: 0.85rem; color: #666; box-shadow: 0 2px 8px rgba(0,0,0,0.1); z-index: 1000; }
        .session-info .timeout { font-weight: 600; color: #667eea; }
    </style>
</head>
<body>
    <div class="session-info">
        ⏱️ Session timeout: <span class="timeout">30 min inactivity</span>
    </div>
    <a href="{{ url_for('admin_logout') }}" class="btn btn-danger logout-btn">🔒 Logout</a>
    
    <div class="admin-container">
        <!-- Tab Navigation -->
        <div class="admin-tabs">
            <a href="{{ url_for('admin_gpt_settings') }}" class="{% if active_tab == 'gpt' %}active{% endif %}">🤖 GPT Settings</a>
            <a href="{{ url_for('admin_prompts') }}" class="{% if active_tab == 'prompts' %}active{% endif %}">💬 Prompts</a>
            <a href="{{ url_for('admin_users') }}" class="{% if active_tab == 'users' %}active{% endif %}">👥 Users</a>
            <a href="{{ url_for('admin_system') }}" class="{% if active_tab == 'system' %}active{% endif %}">⚙️ System</a>
            <a href="{{ url_for('admin_stats') }}" class="{% if active_tab == 'stats' %}active{% endif %}">📊 Stats</a>
            <a href="{{ url_for('admin_feedback') }}" class="{% if active_tab == 'feedback' %}active{% endif %}">💬 Feedback</a>
            <a href="{{ url_for('admin_audit_logs') }}" class="{% if active_tab == 'audit' %}active{% endif %}">🔍 Audit Logs</a>
        </div>
        
        <div class="settings-card">
            <h1 class="text-center mb-4">🎯 Two-Agent AI Configuration</h1>
            
            {% if message %}
            <div class="success-msg">
                ✅ {{ message }}
            </div>
            {% endif %}
            
            <div class="warning">
                <strong>⚠️ Two-Agent System:</strong> RANKER handles bulk scoring (all candidates × all criteria). 
                INSIGHT generates deep analysis for selected candidates only. Changes take effect immediately after saving.
            </div>
            
            <!-- Profit Margin Dashboard -->
            <div class="profit-card">
                <h3>💰 Business Health Monitor</h3>
                <div class="row">
                    <div class="col-md-4">
                        <small>Analysis Cost (50 cand, 20 crit)</small>
                        <div class="profit-value" id="analysisCost">$0.45</div>
                    </div>
                    <div class="col-md-4">
                        <small>Revenue (Standard Tier)</small>
                        <div class="profit-value">$4.00</div>
                    </div>
                    <div class="col-md-4">
                        <small>Profit Margin</small>
                        <div class="profit-value" id="profitMargin">79%</div>
                    </div>
                </div>
                <div class="profit-breakdown">
                    <strong>Cost Breakdown:</strong>
                    <div id="rankerCost">• RANKER: $0.30 (1000 scoring calls)</div>
                    <div id="insightCost">• INSIGHT: $0.15 (5 deep insights)</div>
                    <div id="profitAmount">• <strong>Profit: $3.55</strong></div>
                </div>
            </div>
        </div>
        
        <form method="POST" action="{{ url_for('admin_save_settings') }}">
            <!-- Core Two-Agent Settings -->
            <div class="settings-card">
                <div class="section-header" onclick="toggleSection(this)">
                    <div>
                        <h3 class="mb-0">🤖 Core Agent Configuration<span class="section-badge">4 settings</span></h3>
                    </div>
                    <span class="toggle-icon">▼</span>
                </div>
                <div class="section-content">
                
                <!-- RANKER Model -->
                <div class="setting-item">
                    <div class="setting-name">RANKER Model<span class="agent-badge ranker-badge">BULK SCORING</span></div>
                    <select name="ranker_model" class="form-select mt-2" onchange="updateCostEstimate()">
                        {% for option in settings.ranker_model.options %}
                        <option value="{{ option }}" {% if option == settings.ranker_model.value %}selected{% endif %}>
                            {{ option }}
                        </option>
                        {% endfor %}
                    </select>
                    <div class="setting-desc">{{ settings.ranker_model.description }}</div>
                    <div class="setting-explanation">
                        <strong>Usage:</strong> {{ settings.ranker_model.usage }}<br>
                        <strong>Explanation:</strong> {{ settings.ranker_model.explanation }}
                    </div>
                    <div class="setting-explanation" style="background: #fff9e6; border-left-color: #ffc107;">
                        <strong>💰 Cost:</strong> {{ settings.ranker_model.cost_impact }}
                    </div>
                </div>
                
                <!-- INSIGHT Model -->
                <div class="setting-item">
                    <div class="setting-name">INSIGHT Model<span class="agent-badge insight-badge">DEEP ANALYSIS</span></div>
                    <select name="insight_model" class="form-select mt-2" onchange="updateCostEstimate()">
                        {% for option in settings.insight_model.options %}
                        <option value="{{ option }}" {% if option == settings.insight_model.value %}selected{% endif %}>
                            {{ option }}
                        </option>
                        {% endfor %}
                    </select>
                    <div class="setting-desc">{{ settings.insight_model.description }}</div>
                    <div class="setting-explanation">
                        <strong>Usage:</strong> {{ settings.insight_model.usage }}<br>
                        <strong>Explanation:</strong> {{ settings.insight_model.explanation }}
                    </div>
                    <div class="setting-explanation" style="background: #fff9e6; border-left-color: #ffc107;">
                        <strong>💰 Cost:</strong> {{ settings.insight_model.cost_impact }}
                    </div>
                </div>
                
                <!-- RANKER Temperature -->
                <div class="setting-item">
                    <div class="setting-name">RANKER Temperature<span class="agent-badge ranker-badge">CONSISTENCY</span></div>
                    <div class="d-flex align-items-center gap-3 mt-2">
                        <input type="range" name="ranker_temperature" class="form-range flex-grow-1" 
                               min="0" max="1" step="0.05" value="{{ settings.ranker_temperature.value }}"
                               oninput="this.nextElementSibling.value = this.value; updateCostEstimate()">
                        <output class="setting-value" style="min-width: 60px; text-align: center;">{{ settings.ranker_temperature.value }}</output>
                    </div>
                    <div class="range-labels">
                        <span>0.0 (Deterministic)</span>
                        <span>0.1 (Recommended)</span>
                        <span>1.0 (Creative)</span>
                    </div>
                    <div class="setting-desc">{{ settings.ranker_temperature.description }}</div>
                    <div class="setting-explanation">
                        <strong>Explanation:</strong> {{ settings.ranker_temperature.explanation }}
                    </div>
                </div>
                
                <!-- INSIGHT Temperature -->
                <div class="setting-item">
                    <div class="setting-name">INSIGHT Temperature<span class="agent-badge insight-badge">READABILITY</span></div>
                    <div class="d-flex align-items-center gap-3 mt-2">
                        <input type="range" name="insight_temperature" class="form-range flex-grow-1" 
                               min="0" max="1" step="0.05" value="{{ settings.insight_temperature.value }}"
                               oninput="this.nextElementSibling.value = this.value">
                        <output class="setting-value" style="min-width: 60px; text-align: center;">{{ settings.insight_temperature.value }}</output>
                    </div>
                    <div class="range-labels">
                        <span>0.0 (Robotic)</span>
                        <span>0.4 (Recommended)</span>
                        <span>1.0 (Creative)</span>
                    </div>
                    <div class="setting-desc">{{ settings.insight_temperature.description }}</div>
                    <div class="setting-explanation">
                        <strong>Explanation:</strong> {{ settings.insight_temperature.explanation }}
                    </div>
                </div>
                </div>
            </div>
            
            <!-- Advanced API Settings (Collapsed by Default) -->
            <div class="settings-card">
                <div class="section-header collapsed" onclick="toggleSection(this)">
                    <div>
                        <h3 class="mb-0">⚙️ Advanced API Settings<span class="section-badge">Expert Controls</span></h3>
                    </div>
                    <span class="toggle-icon">▼</span>
                </div>
                <div class="section-content collapsed">
                
                <!-- Presence Penalty -->
                <div class="setting-item">
                    <div class="setting-name">Presence Penalty<span class="agent-badge insight-badge">INSIGHT ONLY</span></div>
                    <div class="d-flex align-items-center gap-3 mt-2">
                        <input type="range" name="presence_penalty" class="form-range flex-grow-1" 
                               min="0" max="2" step="0.1" value="{{ settings.advanced_api_settings.presence_penalty.value }}"
                               oninput="this.nextElementSibling.value = this.value">
                        <output class="setting-value" style="min-width: 60px; text-align: center;">{{ settings.advanced_api_settings.presence_penalty.value }}</output>
                    </div>
                    <div class="range-labels">
                        <span>0.0 (None)</span>
                        <span>0.4 (Recommended)</span>
                        <span>2.0 (Maximum)</span>
                    </div>
                    <div class="setting-desc">{{ settings.advanced_api_settings.presence_penalty.description }}</div>
                    <div class="setting-explanation">
                        <strong>Explanation:</strong> {{ settings.advanced_api_settings.presence_penalty.explanation }}
                    </div>
                </div>
                
                <!-- Frequency Penalty -->
                <div class="setting-item">
                    <div class="setting-name">Frequency Penalty<span class="agent-badge insight-badge">INSIGHT ONLY</span></div>
                    <div class="d-flex align-items-center gap-3 mt-2">
                        <input type="range" name="frequency_penalty" class="form-range flex-grow-1" 
                               min="0" max="2" step="0.1" value="{{ settings.advanced_api_settings.frequency_penalty.value }}"
                               oninput="this.nextElementSibling.value = this.value">
                        <output class="setting-value" style="min-width: 60px; text-align: center;">{{ settings.advanced_api_settings.frequency_penalty.value }}</output>
                    </div>
                    <div class="range-labels">
                        <span>0.0 (None)</span>
                        <span>0.3 (Recommended)</span>
                        <span>2.0 (Maximum)</span>
                    </div>
                    <div class="setting-desc">{{ settings.advanced_api_settings.frequency_penalty.description }}</div>
                    <div class="setting-explanation">
                        <strong>Explanation:</strong> {{ settings.advanced_api_settings.frequency_penalty.explanation }}
                    </div>
                </div>
                
                <!-- RANKER Max Tokens -->
                <div class="setting-item">
                    <div class="setting-name">RANKER Max Tokens<span class="agent-badge ranker-badge">OUTPUT LIMIT</span></div>
                    <div class="d-flex align-items-center gap-3 mt-2">
                        <input type="range" name="ranker_max_tokens" class="form-range flex-grow-1" 
                               min="200" max="500" step="50" value="{{ settings.advanced_api_settings.ranker_max_tokens.value }}"
                               oninput="this.nextElementSibling.value = this.value; updateCostEstimate()">
                        <output class="setting-value" style="min-width: 80px; text-align: center;">{{ settings.advanced_api_settings.ranker_max_tokens.value }}</output>
                    </div>
                    <div class="range-labels">
                        <span>200 (Minimal)</span>
                        <span>300 (Recommended)</span>
                        <span>500 (Detailed)</span>
                    </div>
                    <div class="setting-desc">{{ settings.advanced_api_settings.ranker_max_tokens.description }}</div>
                    <div class="setting-explanation">
                        <strong>Explanation:</strong> {{ settings.advanced_api_settings.ranker_max_tokens.explanation }}
                    </div>
                </div>
                
                <!-- INSIGHT Max Tokens -->
                <div class="setting-item">
                    <div class="setting-name">INSIGHT Max Tokens<span class="agent-badge insight-badge">OUTPUT LIMIT</span></div>
                    <div class="d-flex align-items-center gap-3 mt-2">
                        <input type="range" name="insight_max_tokens" class="form-range flex-grow-1" 
                               min="500" max="2000" step="100" value="{{ settings.advanced_api_settings.insight_max_tokens.value }}"
                               oninput="this.nextElementSibling.value = this.value; updateCostEstimate()">
                        <output class="setting-value" style="min-width: 100px; text-align: center;">{{ settings.advanced_api_settings.insight_max_tokens.value }}</output>
                    </div>
                    <div class="range-labels">
                        <span>500 (Brief)</span>
                        <span>1000 (Recommended)</span>
                        <span>2000 (Detailed)</span>
                    </div>
                    <div class="setting-desc">{{ settings.advanced_api_settings.insight_max_tokens.description }}</div>
                    <div class="setting-explanation">
                        <strong>Explanation:</strong> {{ settings.advanced_api_settings.insight_max_tokens.explanation }}
                    </div>
                </div>
                </div>
            </div>
            
            <!-- Score Thresholds -->
            <div class="settings-card">
                <div class="section-header" onclick="toggleSection(this)">
                    <div>
                        <h3 class="mb-0">🎯 Score Thresholds<span class="section-badge">UI Display</span></h3>
                    </div>
                    <span class="toggle-icon">▼</span>
                </div>
                <div class="section-content">
                
                <!-- High Threshold -->
                <div class="setting-item">
                    <div class="setting-name">High/Strong Threshold</div>
                    <div class="d-flex align-items-center gap-3 mt-2">
                        <input type="range" name="high_threshold" class="form-range flex-grow-1" 
                               min="0.6" max="0.9" step="0.05" value="{{ settings.score_thresholds.high_threshold.value }}"
                               oninput="this.nextElementSibling.value = this.value">
                        <output class="setting-value" style="min-width: 60px; text-align: center;">{{ settings.score_thresholds.high_threshold.value }}</output>
                    </div>
                    <div class="range-labels">
                        <span>0.6 (Lenient)</span>
                        <span>0.75 (Recommended)</span>
                        <span>0.9 (Strict)</span>
                    </div>
                    <div class="setting-desc">{{ settings.score_thresholds.high_threshold.description }}</div>
                </div>
                
                <!-- Low Threshold -->
                <div class="setting-item">
                    <div class="setting-name">Low/Weak Threshold</div>
                    <div class="d-flex align-items-center gap-3 mt-2">
                        <input type="range" name="low_threshold" class="form-range flex-grow-1" 
                               min="0.2" max="0.5" step="0.05" value="{{ settings.score_thresholds.low_threshold.value }}"
                               oninput="this.nextElementSibling.value = this.value">
                        <output class="setting-value" style="min-width: 60px; text-align: center;">{{ settings.score_thresholds.low_threshold.value }}</output>
                    </div>
                    <div class="range-labels">
                        <span>0.2 (Strict)</span>
                        <span>0.35 (Recommended)</span>
                        <span>0.5 (Lenient)</span>
                    </div>
                    <div class="setting-desc">{{ settings.score_thresholds.low_threshold.description }}</div>
                </div>
                </div>
            </div>
            
            <!-- Usage Notes -->
            <div class="settings-card" style="background: #f8f9fa;">
                <h4>📚 Business Intelligence</h4>
                <div class="mt-3">
                    <p><strong>Current Architecture:</strong> {{ settings._usage_notes.current_architecture }}</p>
                    <p><strong>Typical Analysis Cost:</strong> {{ settings._usage_notes.typical_analysis_cost }}</p>
                    <p><strong>Optimization Priority:</strong> {{ settings._usage_notes.optimization_priority }}</p>
                    <p><strong>Cost Saving Tips:</strong> {{ settings._usage_notes.cost_saving_tips }}</p>
                    <p><strong>Quality Improvement:</strong> {{ settings._usage_notes.quality_improvement_tips }}</p>
                </div>
            </div>
            
            <div class="text-center">
                <button type="submit" class="btn btn-primary btn-save">💾 Save Configuration</button>
            </div>
        </form>
    </div>
    
    <script>
        // Toggle section collapse
        function toggleSection(header) {
            const content = header.nextElementSibling;
            const icon = header.querySelector('.toggle-icon');
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                header.classList.remove('collapsed');
            } else {
                content.classList.add('collapsed');
                header.classList.add('collapsed');
            }
        }
        
        function updateCostEstimate() {
            // Get current values
            const rankerModel = document.querySelector('select[name="ranker_model"]').value;
            const insightModel = document.querySelector('select[name="insight_model"]').value;
            const rankerMaxTokens = parseInt(document.querySelector('input[name="ranker_max_tokens"]').value);
            const insightMaxTokens = parseInt(document.querySelector('input[name="insight_max_tokens"]').value);
            
            // Model costs (per million tokens)
            const costs = {
                'gpt-4o': { input: 2.5, output: 10.0 },
                'gpt-4o-mini': { input: 0.15, output: 0.6 },
                'gpt-4-turbo': { input: 10.0, output: 30.0 },
                'gpt-4': { input: 30.0, output: 60.0 }
            };
            
            // Typical analysis: 50 candidates, 20 criteria, 5 insights
            const numCandidates = 50;
            const numCriteria = 20;
            const numInsights = 5;
            const rankerCalls = numCandidates * numCriteria; // 1000 calls
            
            // RANKER cost calculation
            const rankerInputTokens = 500; // ~500 tokens per scoring call
            const rankerOutputTokens = rankerMaxTokens * 0.5; // Typically uses ~50% of max
            const rankerInputCost = (rankerInputTokens * rankerCalls / 1000000) * costs[rankerModel].input;
            const rankerOutputCost = (rankerOutputTokens * rankerCalls / 1000000) * costs[rankerModel].output;
            const rankerTotalCost = rankerInputCost + rankerOutputCost;
            
            // INSIGHT cost calculation
            const insightInputTokens = 3000; // ~3000 tokens per insight call
            const insightOutputTokens = insightMaxTokens * 0.5; // Typically uses ~50% of max
            const insightInputCost = (insightInputTokens * numInsights / 1000000) * costs[insightModel].input;
            const insightOutputCost = (insightOutputTokens * numInsights / 1000000) * costs[insightModel].output;
            const insightTotalCost = insightInputCost + insightOutputCost;
            
            // Total cost and profit
            const totalCost = rankerTotalCost + insightTotalCost;
            const revenue = 4.00; // Standard tier price
            const profit = revenue - totalCost;
            const profitMargin = (profit / revenue) * 100;
            
            // Update display
            document.getElementById('analysisCost').textContent = '$' + totalCost.toFixed(2);
            document.getElementById('rankerCost').textContent = '• RANKER: $' + rankerTotalCost.toFixed(2) + ' (' + rankerCalls + ' scoring calls)';
            document.getElementById('insightCost').textContent = '• INSIGHT: $' + insightTotalCost.toFixed(2) + ' (' + numInsights + ' deep insights)';
            document.getElementById('profitAmount').textContent = '• Profit: $' + profit.toFixed(2);
            document.getElementById('profitMargin').textContent = profitMargin.toFixed(0) + '%';
            
            // Color code profit margin
            const profitCard = document.querySelector('.profit-card');
            if (profitMargin < 50) {
                profitCard.style.background = 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)';
            } else if (profitMargin < 70) {
                profitCard.style.background = 'linear-gradient(135deg, #f39c12 0%, #e67e22 100%)';
            } else {
                profitCard.style.background = 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)';
            }
        }
        
        // Update cost estimate on page load
        updateCostEstimate();
    </script>
</body>
</html>
'''

# Write to file
with open('flask_app/templates/admin_gpt.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ admin_gpt.html updated with two-agent architecture UI!")
print("📊 Includes profit margin dashboard and business health monitor")
print("⚙️ Advanced settings collapsed by default for clean interface")
