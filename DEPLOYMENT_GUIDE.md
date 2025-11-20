# Candidate Analyser - Deployment Guide

## Phase 1: Streamlit Community Cloud Deployment

### Prerequisites
1. GitHub account
2. Streamlit Community Cloud account (free, sign up at https://streamlit.io/cloud)
3. OpenAI API key

### Step 1: Prepare Local Files

✅ Already completed:
- Password protection added
- `.gitignore` created
- `.streamlit/secrets.toml` created (for local testing)

### Step 2: Create GitHub Repository

1. Go to https://github.com and create a new repository
   - Name it something like `candidate-analyser`
   - Make it **private** (recommended for beta testing)
   - Don't initialize with README (we already have files)

2. Push your code to GitHub:
   ```bash
   cd "c:\Users\nigel\OneDrive\Documents\Nigels Documents\ChatGPT\candidate-summariser"
   git init
   git add .
   git commit -m "Initial commit - ready for deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/candidate-analyser.git
   git push -u origin main
   ```

### Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your GitHub repository
4. Set main file path: `app.py`
5. Click "Advanced settings" and add secrets:
   ```toml
   password = "YOUR_SECURE_PASSWORD"
   OPENAI_API_KEY = "your-openai-api-key"
   ```
6. Click "Deploy"

### Step 4: Share with Testers

Once deployed, you'll get a URL like: `https://your-app-name.streamlit.app`

Share this URL and the password with your testers.

### Testing Credentials

**Default password (change in secrets):** `testpass123`

### Important Notes

- **Password:** Change `testpass123` in the Streamlit Cloud secrets before sharing
- **API Key:** Your OpenAI API key should be added to Streamlit Cloud secrets
- **Limits:** Free tier has 1GB RAM, suitable for 5-20 concurrent users
- **Privacy:** Your repository can be private, app URL is public but password-protected

### Troubleshooting

If deployment fails:
1. Check that all packages in `requirements.txt` are available on PyPI
2. Review Streamlit Cloud logs for error messages
3. Ensure secrets are properly set in Streamlit Cloud dashboard

### Next Steps After Testing

Based on user feedback:
- **Phase 2:** Add proper multi-user authentication
- **Phase 3:** Move to paid hosting with database for user management
