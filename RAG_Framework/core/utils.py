import os
import pickle


def save_pickle(obj, filename):
    """Save object to pickle file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    """Load object from pickle file"""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None
