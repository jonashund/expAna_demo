import os
import expAna

project_name = "test_expAna"

# # Run expDoc for documentation purposes from the ./tex directory
os.chdir("./tex")
expAna.docu.main(project_name)
os.chdir("..")

# # Analyse the example test with expAna

# Stress computation
# Option 1: Evaluate the exported Istra4D data to compute the stress in conjunction with the data from the documentation
expAna.eval2stress.main()

# # Option 2: Use the Istra4D acquisition data, perform digital image correlation with µDIC, and compute the stresses with expAna
expAna.acquis2tif.main()
expAna.dic_with_muDIC.main()
expAna.muDIC2stress.main()

# Basic visualisation of the results for checking and spotting errors
# Stress-strain curves for each experiment and DIC system used
expAna.review.stress(dic_system="istra")
expAna.review.stress(dic_system="muDIC")
# Force-displacement curves for each experiment and DIC system used
expAna.review.force()

# Plot of the DIC deformation field as an overlay on the raw image data
expAna.plot.dic_strains(experiment_name="Test8", displacement=5, strain_component="x")
expAna.plot.dic_strains(
    experiment_name="Test8", displacement=10, strain_component="x", max_triang_len=50,
)


# Examples for experiment filtering during the analysis
# experiment_list argument (only perform the operations for the experiments in the list)
expAna.acquis2tif.main(experiment_list=["Test8"])
# ignore_list (do not perform the operations for the experiments in the list)
expAna.dic_with_muDIC.main(
    ignore_list=["Test8"]
)  # should result in an empty list since the project only contains 'Test8'
# select (filtering experiments based on their documented properties, i.e. specimen orientation)
expAna.muDIC2stress.main(select=["specimen_orientation", "parallel to flow"])

# Example for clipping of curves to exclude crude DIC data post failure
expAna.review.stress(dic_system="istra", set_failure=True)
expAna.review.stress(dic_system="muDIC", set_failure=True)
