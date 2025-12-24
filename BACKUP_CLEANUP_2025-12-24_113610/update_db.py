"""
Update database schema to add category_map column
"""
from database import db
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///candidate_evaluator.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    # Drop all tables and recreate
    db.drop_all()
    db.create_all()
    print("✅ Database recreated successfully with category_map column!")
    print("⚠️  Warning: All existing data has been deleted.")
