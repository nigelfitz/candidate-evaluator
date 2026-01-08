"""
Flask extensions - centralized to avoid circular imports
"""
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect()
