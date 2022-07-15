import os
import shutil
import pathlib

# remove created directories
shutil.rmtree(pathlib.Path(os.path.dirname(__file__), "expAna_plots"))
