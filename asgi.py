# asgi.py
from app import app
from asgi_flask import ASGIFlask

asgi_app = ASGIFlask(app)
