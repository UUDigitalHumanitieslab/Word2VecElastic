import os

def checkPath(path):
    """Check if path exists and if it does not, create it."""
    if not os.path.isdir(path):
        os.mkdir(path)
