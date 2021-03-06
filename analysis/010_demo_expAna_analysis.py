import expAna

import demo_expAna_analysis_funcs as funcs

# Previously,projects have been created, stresses have been computated, and the results have been reviewed.
# Now, analyses of the individual experiments' results can be carried out to gain further insight into the impact of parameters of interest

# Example 1
# Create an analysis object of type "stress":
my_analysis = expAna.analysis.Analysis("stress")
# Setup the analysis by selecting all experiments on PC/ABS 45/55 carried out using a crosshead speed of 0.1 mm/s and compare them regarding the impact of different specimen orientations
my_analysis.setup(
    compare="specimen_orientation",
    select=["crosshead_speed", 0.1],
    ignore_list=["Test16CORN1", "Test13CORN1"],
)
# Perform the actual analysis
my_analysis.compute_data_stress()
# Visualise the results in a basic plot
my_analysis.plot_data()
# Export the results to a .pickle file
expAna.data_trans.export_analysis(
    my_analysis, out_filename=f"analysis_{my_analysis.export_prefix}.pickle",
)

# Example 2
# Create an analysis object of type "volume strain":
my_analysis = expAna.analysis.Analysis("vol_strain")
# Setup the analysis by selecting all experiments on PC/ABS 45/55 carried out with specimens in orientation parallel to flow and compare them regarding the impact crosshead speed
my_analysis.setup(
    compare="crosshead_speed",
    select=["specimen_orientation", "parallel to flow"],
    ignore_list=["Test16CORN1", "Test13CORN1"],
)
# Perform the actual analysis
my_analysis.compute_data_vol_strain()
# Visualise the results in a basic plot
my_analysis.plot_data()
# Export the results to a .pickle file
expAna.data_trans.export_analysis(
    my_analysis, out_filename=f"analysis_{my_analysis.export_prefix}.pickle",
)

# Example 1 performed and subsequently plotted in the style of Jonas Hund's PhD thesis
funcs.make_analysis_figure(
    select=["crosshead_speed", 0.1],
    compare="specimen_orientation",
    material="4555",
    experiment_type="uniax_tension",
    analysis_type="stress",
    ignore_list=["Test16CORN1", "Test13CORN1"],
    annotate_dict=None,
)
