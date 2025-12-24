"""Clear old draft data that may be incompatible"""
from database import db, Draft
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///candidate_evaluator.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    count = Draft.query.delete()
    db.session.commit()
    print(f"✅ Cleared {count} old draft(s) from database")
    print("You can now upload a fresh JD to analyze")
