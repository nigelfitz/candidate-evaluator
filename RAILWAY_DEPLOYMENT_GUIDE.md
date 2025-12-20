# Railway Deployment Guide - Candidate Evaluator

## ✅ Files Created for Deployment
- `Procfile` - Tells Railway how to start your app
- `railway.toml` - Railway configuration
- `flask_app/requirements.txt` - Updated with PostgreSQL support
- `.gitignore` - Updated to exclude database files

## Step-by-Step Deployment Process

### Step 1: Push Code to GitHub ✅ (Do this first)

1. **Check Git status:**
   ```powershell
   git status
   ```

2. **Add all files:**
   ```powershell
   git add .
   ```

3. **Commit changes:**
   ```powershell
   git commit -m "Prepare for Railway deployment with user tracking"
   ```

4. **Push to GitHub:**
   ```powershell
   git push origin main
   ```
   
   Note: If you don't have a GitHub repository yet, create one at github.com and link it:
   ```powershell
   git remote add origin https://github.com/YOUR-USERNAME/candidate-evaluator.git
   git branch -M main
   git push -u origin main
   ```

---

### Step 2: Create Railway Account & Project

1. Go to https://railway.app
2. Sign up with GitHub (easiest option)
3. Click "New Project"
4. Choose "Deploy from GitHub repo"
5. Select your `candidate-evaluator` repository
6. Railway will automatically detect Flask and start deploying

---

### Step 3: Add PostgreSQL Database

1. In your Railway project dashboard, click "New"
2. Select "Database" → "Add PostgreSQL"
3. Railway will automatically:
   - Create the database
   - Set the `DATABASE_URL` environment variable
   - Your app will use it automatically

---

### Step 4: Set Environment Variables

In Railway project settings, add these variables:

**Required:**
```
SECRET_KEY=<generate-a-random-32-character-string>
OPENAI_API_KEY=<your-openai-api-key>
STRIPE_SECRET_KEY=<your-stripe-secret-key>
STRIPE_PUBLISHABLE_KEY=<your-stripe-publishable-key>
STRIPE_WEBHOOK_SECRET=<your-stripe-webhook-secret>
```

**To generate SECRET_KEY:**
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

---

### Step 5: Domain Setup

1. **In Railway:**
   - Go to Settings → Domains
   - Click "Generate Domain" (you'll get a free railway.app subdomain)
   - Copy this URL for testing

2. **For your Cloudflare domain (candidateevaluator.com):**
   - Get the Railway domain (e.g., candidate-evaluator-production.up.railway.app)
   - In Cloudflare DNS settings:
     - Add CNAME record: `@` → `candidate-evaluator-production.up.railway.app`
     - Add CNAME record: `www` → `candidate-evaluator-production.up.railway.app`
   - Wait 5-10 minutes for DNS propagation

---

### Step 6: Initialize Database

After first deployment, Railway will automatically create all database tables when your app starts. No manual migration needed!

---

### Step 7: Test Your Live Site

1. Visit your Railway URL
2. Create a test account
3. Try a small analysis ($4)
4. Check everything works

---

### Step 8: Set Up Stripe Webhooks (Important!)

1. In Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://candidateevaluator.com/webhook/stripe`
3. Select events:
   - `checkout.session.completed`
   - `payment_intent.succeeded`
4. Copy the webhook secret
5. Add to Railway environment variables as `STRIPE_WEBHOOK_SECRET`

---

## Monitoring & Costs

**Railway Pricing:**
- $5/month base (covers small PostgreSQL database)
- Pay-as-you-go for compute (~$0.000463/GB-hour)
- Estimated total: $5-10/month for low traffic

**View Logs:**
- Railway dashboard → Click your service → View logs
- See real-time errors and requests

**Database Backups:**
- Railway automatically backs up PostgreSQL daily
- Can restore from dashboard if needed

---

## Troubleshooting

**App won't start:**
- Check Railway logs for errors
- Verify all environment variables are set
- Make sure DATABASE_URL is set by PostgreSQL addon

**Database errors:**
- Ensure PostgreSQL addon is added
- Check DATABASE_URL in environment variables
- Verify requirements.txt includes psycopg2-binary

**Stripe payments not working:**
- Verify webhook endpoint in Stripe dashboard
- Check STRIPE_WEBHOOK_SECRET matches
- View Railway logs during test payment

---

## Next: Google Analytics Setup (After Live)

Once your site is live at candidateevaluator.com:

1. Create GA4 property at https://analytics.google.com
2. Get measurement ID (G-XXXXXXXXXX)
3. Add tracking code to `flask_app/templates/base.html`
4. Wait 24 hours for data to appear

---

## Need Help?

- Railway docs: https://docs.railway.app
- Railway Discord: Very active community
- Check Railway logs first - they're very detailed
