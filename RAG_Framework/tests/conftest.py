import sys
import os

# Make `RAG_Framework` importable as a package from anywhere pytest is invoked.
# tests/ lives inside RAG_Framework/, so we go up two levels to reach the
# directory that *contains* RAG_Framework/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
