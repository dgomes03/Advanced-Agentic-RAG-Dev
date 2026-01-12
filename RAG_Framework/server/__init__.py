"""
RAG Framework WebSocket Server Module

This module provides a Flask-SocketIO based web server for the RAG framework,
enabling real-time streaming of responses, tool call visualization, and
advanced reasoning display through WebSocket connections.
"""

from .app import create_app, run_server

__all__ = ['create_app', 'run_server']
