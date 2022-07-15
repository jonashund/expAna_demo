import os
import shutil
import pathlib

# remove created directories
shutil.rmtree(pathlib.Path(os.path.dirname(__file__), "expAna_data"))
shutil.rmtree(pathlib.Path(os.path.dirname(__file__), "expAna_docu"))
shutil.rmtree(pathlib.Path(os.path.dirname(__file__), "expAna_plots"))
shutil.rmtree(pathlib.Path(os.path.dirname(__file__), "data_export2tif"))
shutil.rmtree(pathlib.Path(os.path.dirname(__file__), "data_muDIC"))
