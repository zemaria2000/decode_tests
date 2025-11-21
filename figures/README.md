# Figures

This folder contains several figures concerning different types of graphs. Here is a short description:

* `<dataset_name>_<model_name>_decode_opt.pdf`: DECODE tree representation for a specific dataset and model name (using the [optimisation](https://github.com/zemaria2000/decode_tests/tree/main/optimisations) hyperparameters).
* `<dataset_name>_<model_name>_sklearn_opt.pdf`: Decision Tree (from skelearn) representation for a specific dataset and model name (using when possible the [optimisation](https://github.com/zemaria2000/decode_tests/tree/main/optimisations) hyperparameters for the DECODE model).
* `<dataset_name>_<model_name>_corr_matrix.pdf`: Correlation matrix between the ranks that were obtained by SHAP, LIME, and PFI, compared to those of the DECODE and Decision Tree models. **Note that these results are not contemplated in the paper!** 