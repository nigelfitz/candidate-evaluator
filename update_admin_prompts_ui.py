# Generate new admin_prompts.html with Prompts Manager "Control Tower"

html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompts Manager - Admin Panel</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/theme/monokai.min.css">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .admin-container { max-width: 1400px; margin: 0 auto; }
        .prompts-card { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); padding: 30px; margin-bottom: 20px; }
        .logout-btn { position: fixed; top: 20px; right: 20px; z-index: 1000; }
        .admin-tabs { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); padding: 15px 30px; margin-bottom: 20px; }
        .admin-tabs a { color: #666; text-decoration: none; padding: 12px 24px; display: inline-block; border-bottom: 3px solid transparent; font-weight: 500; transition: all 0.3s; }
        .admin-tabs a:hover { color: #667eea; }
        .admin-tabs a.active { color: #667eea; border-bottom-color: #667eea; }
        .session-info { position: fixed; top: 20px; right: 180px; background: rgba(255,255,255,0.95); padding: 8px 16px; border-radius: 8px; font-size: 0.85rem; color: #666; box-shadow: 0 2px 8px rgba(0,0,0,0.1); z-index: 1000; }
        .session-info .timeout { font-weight: 600; color: #667eea; }
        .prompt-tabs { border-bottom: 2px solid #e0e0e0; margin-bottom: 30px; }
        .prompt-tab { display: inline-block; padding: 12px 24px; margin-right: 10px; border: 2px solid transparent; border-bottom: none; border-radius: 8px 8px 0 0; cursor: pointer; font-weight: 600; transition: all 0.3s; }
        .prompt-tab:hover { background: #f5f5f5; }
        .prompt-tab.active { background: white; border-color: #667eea; color: #667eea; position: relative; top: 2px; }
        .prompt-tab.ranker { border-color: #3498db; color: #3498db; }
        .prompt-tab.insight { border-color: #e74c3c; color: #e74c3c; }
        .prompt-tab.jd { border-color: #2ecc71; color: #2ecc71; }
        .prompt-content { display: none; }
        .prompt-content.active { display: block; }
        .dev-notes-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
        .dev-notes-box h4 { margin: 0 0 15px 0; font-size: 1.2rem; }
        .dev-note-item { margin-bottom: 12px; padding-left: 20px; position: relative; }
        .dev-note-item::before { content: "→"; position: absolute; left: 0; font-weight: bold; }
        .pro-tip-box { background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); color: white; padding: 18px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(243, 156, 18, 0.3); }
        .pro-tip-box h5 { margin: 0 0 10px 0; font-size: 1.1rem; }
        .variable-docs { background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 25px; border-left: 4px solid #667eea; }
        .variable-docs h5 { color: #667eea; margin: 0 0 15px 0; }
        .variable-item { background: white; padding: 12px; border-radius: 6px; margin-bottom: 10px; font-family: 'Courier New', monospace; }
        .variable-name { color: #e74c3c; font-weight: bold; }
        .variable-desc { color: #555; font-size: 0.9rem; margin-top: 5px; }
        .prompt-editor { border: 2px solid #e0e0e0; border-radius: 8px; margin-bottom: 20px; }
        .prompt-editor h5 { background: #f8f9fa; padding: 12px 16px; margin: 0; border-bottom: 2px solid #e0e0e0; border-radius: 6px 6px 0 0; }
        .prompt-editor textarea { width: 100%; min-height: 200px; padding: 15px; border: none; font-family: 'Courier New', monospace; font-size: 0.95rem; resize: vertical; }
        .btn-save { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; padding: 12px 30px; font-size: 1.1rem; color: white; }
        .btn-save:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); color: white; }
        .cost-badge { background: #f39c12; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; margin-left: 10px; }
        .success-msg { background: #d4edda; padding: 12px; border-radius: 8px; border-left: 4px solid #28a745; margin-bottom: 20px; }
        .CodeMirror { height: auto; min-height: 200px; border-radius: 0 0 8px 8px; }
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
        
        <div class="prompts-card">
            <h1 class="text-center mb-4">🎯 Prompts Manager - Control Tower</h1>
            <p class="text-center text-muted mb-4">Monitor and modify what your AI agents are being told to do</p>
            
            {% if message %}
            <div class="success-msg">
                ✅ {{ message }}
            </div>
            {% endif %}
            
            <!-- Prompt Agent Tabs -->
            <div class="prompt-tabs">
                <div class="prompt-tab ranker active" onclick="switchPromptTab('ranker')">
                    🎯 RANKER AGENT
                </div>
                <div class="prompt-tab insight" onclick="switchPromptTab('insight')">
                    💡 INSIGHT AGENT
                </div>
                <div class="prompt-tab jd" onclick="switchPromptTab('jd')">
                    📄 JD EXTRACTION
                </div>
            </div>
            
            <form method="POST" action="{{ url_for('admin_save_prompts') }}">
                <!-- RANKER AGENT TAB -->
                <div id="ranker-content" class="prompt-content active">
                    <div class="dev-notes-box">
                        <h4>🧠 Developer Notes - RANKER AGENT</h4>
                        <div class="dev-note-item">
                            <strong>Purpose:</strong> {{ prompts.ranker_scoring.developer_notes.purpose }}
                        </div>
                        <div class="dev-note-item">
                            <strong>Model Used:</strong> {{ prompts.ranker_scoring.developer_notes.model_used }}
                        </div>
                        <div class="dev-note-item">
                            <strong>Temperature:</strong> {{ prompts.ranker_scoring.developer_notes.temperature }}
                        </div>
                        <div class="dev-note-item">
                            <strong>Cost Per Call:</strong> <span class="cost-badge">${{ prompts.ranker_scoring.developer_notes.cost_per_call }}</span> (1000 calls per analysis)
                        </div>
                    </div>
                    
                    <div class="pro-tip-box">
                        <h5>💡 Pro Tip</h5>
                        <p style="margin: 0;">{{ prompts.ranker_scoring.developer_notes.pro_tip }}</p>
                    </div>
                    
                    <div class="variable-docs">
                        <h5>📋 Variable Dependencies</h5>
                        {% for var, desc in prompts.ranker_scoring.variable_details.items() %}
                        <div class="variable-item">
                            <div class="variable-name">{{ var }}</div>
                            <div class="variable-desc">{{ desc }}</div>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="prompt-editor">
                        <h5>System Prompt (RANKER)</h5>
                        <textarea name="ranker_system_prompt" id="ranker_system_prompt">{{ prompts.ranker_scoring.system_prompt }}</textarea>
                    </div>
                    
                    <div class="prompt-editor">
                        <h5>User Prompt Template (RANKER)</h5>
                        <textarea name="ranker_user_prompt" id="ranker_user_prompt">{{ prompts.ranker_scoring.user_prompt_template }}</textarea>
                    </div>
                </div>
                
                <!-- INSIGHT AGENT TAB -->
                <div id="insight-content" class="prompt-content">
                    <div class="dev-notes-box">
                        <h4>🧠 Developer Notes - INSIGHT AGENT</h4>
                        <div class="dev-note-item">
                            <strong>Purpose:</strong> {{ prompts.insight_generation.developer_notes.purpose }}
                        </div>
                        <div class="dev-note-item">
                            <strong>Model Used:</strong> {{ prompts.insight_generation.developer_notes.model_used }}
                        </div>
                        <div class="dev-note-item">
                            <strong>Temperature:</strong> {{ prompts.insight_generation.developer_notes.temperature }}
                        </div>
                        <div class="dev-note-item">
                            <strong>Cost Per Call:</strong> <span class="cost-badge">${{ prompts.insight_generation.developer_notes.cost_per_call }}</span> (5 calls per analysis)
                        </div>
                    </div>
                    
                    <div class="pro-tip-box">
                        <h5>💡 Pro Tip</h5>
                        <p style="margin: 0;">{{ prompts.insight_generation.developer_notes.pro_tip }}</p>
                    </div>
                    
                    <div class="variable-docs">
                        <h5>📋 Variable Dependencies</h5>
                        {% for var, desc in prompts.insight_generation.variable_details.items() %}
                        <div class="variable-item">
                            <div class="variable-name">{{ var }}</div>
                            <div class="variable-desc">{{ desc }}</div>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="prompt-editor">
                        <h5>System Prompt (INSIGHT)</h5>
                        <textarea name="insight_system_prompt" id="insight_system_prompt">{{ prompts.insight_generation.system_prompt }}</textarea>
                    </div>
                    
                    <div class="prompt-editor">
                        <h5>User Prompt Template (INSIGHT)</h5>
                        <textarea name="insight_user_prompt" id="insight_user_prompt">{{ prompts.insight_generation.user_prompt_template }}</textarea>
                    </div>
                </div>
                
                <!-- JD EXTRACTION TAB -->
                <div id="jd-content" class="prompt-content">
                    <div class="dev-notes-box">
                        <h4>🧠 Developer Notes - JD EXTRACTION</h4>
                        <div class="dev-note-item">
                            <strong>Purpose:</strong> Extracts structured criteria and requirements from job description text
                        </div>
                        <div class="dev-note-item">
                            <strong>Model Used:</strong> Same as RANKER (gpt-4o-mini for cost efficiency)
                        </div>
                        <div class="dev-note-item">
                            <strong>Temperature:</strong> 0.1 (consistency is critical)
                        </div>
                        <div class="dev-note-item">
                            <strong>Cost Per Call:</strong> <span class="cost-badge">$0.0003</span> (1 call per analysis)
                        </div>
                    </div>
                    
                    <div class="pro-tip-box">
                        <h5>💡 Pro Tip</h5>
                        <p style="margin: 0;">If JD extraction returns inconsistent criteria formats, lower temperature to 0.0 for maximum determinism.</p>
                    </div>
                    
                    <div class="variable-docs">
                        <h5>📋 Variable Dependencies</h5>
                        <div class="variable-item">
                            <div class="variable-name">{jd_text}</div>
                            <div class="variable-desc">Full text of the job description provided by the user</div>
                        </div>
                    </div>
                    
                    <div class="prompt-editor">
                        <h5>System Prompt (JD Extraction)</h5>
                        <textarea name="jd_system_prompt" id="jd_system_prompt">{{ prompts.jd_extraction.system_prompt }}</textarea>
                    </div>
                    
                    <div class="prompt-editor">
                        <h5>User Prompt Template (JD Extraction)</h5>
                        <textarea name="jd_user_prompt" id="jd_user_prompt">{{ prompts.jd_extraction.user_prompt_template }}</textarea>
                    </div>
                </div>
                
                <div class="text-center mt-4">
                    <button type="submit" class="btn btn-save">💾 Save All Prompts</button>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        function switchPromptTab(tabName) {
            // Hide all content
            document.querySelectorAll('.prompt-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Deactivate all tabs
            document.querySelectorAll('.prompt-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected content
            document.getElementById(tabName + '-content').classList.add('active');
            
            // Activate selected tab
            document.querySelector('.prompt-tab.' + tabName).classList.add('active');
        }
    </script>
</body>
</html>
'''

# Write to file
with open('flask_app/templates/admin_prompts.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ admin_prompts.html updated with Prompts Manager Control Tower!")
print("🎯 Includes RANKER, INSIGHT, and JD Extraction tabs")
print("💡 Developer notes and pro-tips displayed prominently")
print("📋 Variable dependencies documented for each agent")
