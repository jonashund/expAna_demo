import os
import shutil
import pathlib

# remove created directories
shutil.rmtree(pathlib.Path(os.path.dirname(__file__), "expAna_plots"))

os.remove(
    pathlib.Path(
        os.path.dirname(__file__),
        "expAna_data",
        "analysis_pcAbs_4555_uniax_tension_stress_crosshead_speed_0.1_specimen_orientation.pickle",
    )
)


os.remove(
    pathlib.Path(
        os.path.dirname(__file__),
        "expAna_data",
        "analysis_pcAbs_4555_uniax_tension_vol_strain_specimen_orientation_parallel_to_flow_crosshead_speed.pickle",
    )
)
