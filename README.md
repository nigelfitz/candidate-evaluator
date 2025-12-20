# Candidate Evaluator

AI-powered candidate screening and analysis tool for recruiters, HR professionals, and hiring managers.

## 📁 Project Structure

```
candidate-evaluator/
├── streamlit_app/          # Original Streamlit prototype
│   ├── app.py             # Main Streamlit application
│   ├── requirements.txt   # Streamlit dependencies
│   ├── run-app.bat        # Launch script
│   └── backups/           # Previous versions
│
├── flask_app/             # Production Flask application (in development)
│   └── (coming soon)
│
├── shared/                # Resources used by both apps
│   ├── assets/           # Images, logos
│   ├── test_data/        # Sample CVs for testing
│   └── outputs/          # Generated reports
│
├── docs/                  # Documentation and guides
│   ├── LAUNCH_PLAN.md    # Product launch roadmap
│   └── ...
│
├── .venv/                 # Python virtual environment
├── .env                   # Environment variables (not in git)
└── .gitignore            # Git ignore rules
```

## 🚀 Quick Start

### Streamlit App (Current)
```bash
cd streamlit_app
python -m streamlit run app.py
```

### Flask App (Coming Soon)
```bash
cd flask_app
python app.py
```

## 🔧 Setup

1. Create virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

3. Create `.env` file with required keys:
   ```
   OPENAI_API_KEY=your_key_here
   PASSWORD=your_password_here
   ```

## 📊 Features

- 📄 PDF/DOCX resume parsing
- 🤖 AI-powered job description analysis
- 🎯 Candidate scoring and ranking
- 📊 Comprehensive reports (PDF, Excel, Word)
- 💡 AI-generated insights for top candidates
- 🔒 Secure file handling (no permanent storage)

## 🛠️ Tech Stack

**Current (Streamlit):**
- Python 3.9+
- Streamlit
- OpenAI GPT-4
- PyMuPDF, pdfplumber
- sentence-transformers

**Future (Flask):**
- Flask/FastAPI
- PostgreSQL
- Stripe payments
- JWT authentication

## 📝 License

Proprietary - All rights reserved

## 👤 Author

Built by experienced recruiters and AI engineers.

---

**Status:** Active development  
**Last Updated:** December 18, 2025
