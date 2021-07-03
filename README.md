# AI Project - Shane Weisz (WSZSHA001) and Jonathan Tooke (TKXJON001)

## TED Talk Topic Classification

This directory consists of the following components:

1. `FinalReport.pdf` - a PDF report with a comprehensive description of the work
   done for the project.

2. `notebook/TED_TopicClassification_FullWorkflow.ipynb` - a Jupyter Notebook
   containing code and explanations for the full code workflow conducted for the
   project. Note the MLP hyperparameter search has been commented out, since
   this took place using GPUs and takes a long time to compute. The results of
   this hyperparameter search are stored in
   `notebook/MLP_HyperparameterSearch_Results.csv`.

3. `notebooks/test.csv` - a CSV file containing the encoded test data for the
   project, along with the raw transcripts, true labels and predicted labels
   using the final MLP model.

4. `scripts/reproducing_results.py` - a Python script that can be run to
   reproduce our test results. It makes use of various other helper scripts
   (`scripts/exploratory_data_analysis.py`,
   `scripts/preprocessing.py`,`scripts/analysis_of_performance.py`) which are
   modules breaking up the tasks performed in the Jupyter Notebook full
   workflow, along with the final chosen MLP model, which is stored as a .pb
   file in the folder `final_mlp_model`.

5. `data` - a folder containing the tedtalks CSV files, `main.csv` and
   `transcripts.csv`, used as the data for the project.
