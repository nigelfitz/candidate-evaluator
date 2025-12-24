# Admin Panel - Quick Reference Card

## Access
**URL:** http://localhost:5000/admin/login
**Default Password:** admin123
**Change Password:** Set `ADMIN_PASSWORD` environment variable

## What You Can Configure

### 🤖 **Model Selection**
- `gpt-4o` - Best quality, $0.018/insight (RECOMMENDED)
- `gpt-4o-mini` - 60% cheaper, lower quality
- `gpt-4-turbo` - Same cost, similar quality
- `gpt-4` - 3× cost, no benefit

### 🎯 **Quality Settings**
- **Temperature (0.0-2.0):** `0.2` = consistent, professional
- **Evidence Chars (200-1000):** `600` = good context without waste
- **Evidence Items (5-20):** `10` = top criteria with evidence
- **Candidate Text (1000-10000):** `3000` = captures overview
- **JD Text (1000-10000):** `3000` = captures requirements

### 💰 **Cost Control**
- **Max Tokens (500-2000):** `1000` = enough for 3-6 bullets + paragraph
- Current cost per insight: **~$0.018**
- Profit margin per extra insight: **98%** ($1.00 - $0.018)

### 🎨 **UI Thresholds**
- **High/Strong (0.6-0.9):** `0.75` = green in UI
- **Low/Weak (0.2-0.5):** `0.35` = red in UI

## Common Adjustments

### Want cheaper insights?
1. Switch model to `gpt-4o-mini`
2. Reduce evidence chars to `400-500`
3. Reduce evidence items to `8`
**New cost:** ~$0.007 per insight

### Want better insights?
1. Keep model at `gpt-4o`
2. Increase evidence chars to `700-800`
3. Increase evidence items to `12-15`
**New cost:** ~$0.025 per insight

### Insights too generic?
1. Increase evidence chars to `700-800`
2. Ensure model is `gpt-4o`
3. Keep temperature at `0.2`

### Insights cut off?
1. Increase max tokens to `1200-1500`

## Testing Workflow
1. Make changes in admin panel
2. Click "Save All Settings"
3. Run analysis with **Top 3** insights
4. Check quality in Insights page
5. Adjust if needed
6. Scale up to more insights

## No Server Restart Required!
Changes take effect **immediately** after saving.

---
**Full Guide:** See ADMIN_GUIDE.md
