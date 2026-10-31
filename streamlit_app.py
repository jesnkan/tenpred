"""Streamlit Community Cloud Entrypoint for tenpred.

Delegates execution to app.py while ensuring proper path resolution
and seamless continuous deployment.
"""

from pathlib import Path
import runpy
import sys

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Execute main application script
app_path = root_dir / "app.py"
runpy.run_path(str(app_path), run_name="__main__")

