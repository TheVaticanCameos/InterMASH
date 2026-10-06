import os
import sys

PROJ_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MA_SH_DIR = os.path.join(PROJ_HOME, "ma-sh")
if os.path.isdir(MA_SH_DIR) and MA_SH_DIR not in sys.path:
    sys.path.insert(0, MA_SH_DIR)
