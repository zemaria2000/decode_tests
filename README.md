# Result dissemination for paper "DECODE: DEcision tree Capturing Opaque DEcisions" 

This repository contains all experimental results, figures, tables, and supplementary material associated with the paper:

> **DECODE: DEcision tree Capturing Opaque DEcisions**   
> <span style="font-size: 85%;">**Authors**:   
> * José Cação, TEMA - Centro de Tecnologia Mecânica e Automação, Universidade de Aveiro
> * Mário Antunes, IT - Instituto de Telecomunicações, Universidade de Aveiro  
> * José Paulo Santos, TEMA - Centro de Tecnologia Mecânica e Automação, Universidade de Aveiro </span>  

> [!NOTE]  
> * DECODE's source code is **not yet publicly available**, as the framework is still under development and testing  
> * This repository is focused mostly on providing **results and supplementary material**, no the original implementation  
> * However, it does share **code insights regarding** the conducted tests


## Included experiments

This repository includes all results that were showcased in the original paper, all applied to all datasets (3 classification, 2 regression) and with a series of baseline ML models.

1. **Performance tests**: performance tests were conducted to the following models:  
    * Logistic Regression (clf), Linear Regression (reg), K-Nearest Neighbours (clf and reg), Support Vector Machines (clf), Support Vector Regressor (reg), and Multilayer Perceptron (clf and reg);
    * DECODE representations based on each of those models;
    * Decision Tree classifiers and regressors with **equal hyperparameters** (when possible) to the DECODE representations.
2. **Fitting times comparison**: fitting times for the (1) classifiers and regressors, (2) a Decision Tree representation, and (3) DECODE building times were analysed. 
3. **Explanation times comparison**: compared to three state-of-the-art XAI techniques (SHAP, LIME, PFI) for feature rankings, the DECODE model was compared both in terms of global explanation times and local explanation times
4. **Feature ranking comparison** (not included in paper): an additional set of results, which still lacked consistency and detail to be published, were conducted to compare the feature importance rankings provided by the state-of-the-art XAI techniques, and the Decision Tree and DECODE tree-based structures

Note that the results are spanned across the [results](https://github.com/zemaria2000/decode_tests/tree/main/results) and [figures](https://github.com/zemaria2000/decode_tests/tree/main/figures) folders. Additionally, some datasets used are available in the [datasets](https://github.com/zemaria2000/decode_tests/tree/main/datasets) folder, with the [optimisations](https://github.com/zemaria2000/decode_tests/tree/main/optimisations) folder having results regarding the optimisations of the models


## Repository structure

```
.
├── datasets/                 # Datasets used for testing
├── figures/                  # Figures generated for analysis
├── optimisations/            # Outputs of optimisation processes
├── results/                  # Consolidated results (metrics, tables, etc.)
│   
|   # Classes/auxiliary function files
├── optimise.py               # Main optimisation class
├── pre_process_clf.py        # Preprocessing pipeline for classification datasets
├── pre_process_reg.py        # Preprocessing pipeline for regression datasets
├── tests.py                  # Combined tests wrapper
│
|   # All testing files
├── tests_optimisation.py     # Running optimisations
├── tests_performance.py      # Performance evaluation
├── tests_times_xai.py        # XAI global/local explanation times
├── tests_times.py            # Fitting times experiments
├── tests_xai_ranks.py        # Feature ranking correlation analysis
├── run_all_tests.sh          # Shell script to reproduce all experiments
|
├── LICENSE
└── README.md
```


## Complete paper results

As for the complete paper results, all **relevant tables can be consulted and analysed in the [results/README.md](https://github.com/zemaria2000/decode_tests/blob/main/datasets/README.md) file.**  
Additional **figure results** are available [here](https://github.com/zemaria2000/decode_tests/tree/main/figures), as well as **all optimisation runs** [here](https://github.com/zemaria2000/decode_tests/tree/main/optimisations).  
Once again, it is important to note that DECODE is still under development, and for now, results are only for visualisation and analysis. Hopefully, soon the repository will be available for user testing and experiments, based on the DECODE framework.


## Manuscript status

The associated manuscript has been submitted to the **ICAART 2026: 18th International Conference on Agents and Artificial Intelligence**. It is currently under the process of peer review, waiting to be accepted for the conference.  
Once it is accepted, and presented, a reference and DOI will be added!


## Contact

For any further questions, I leave here my contact (corrsponding author of the paper):  
José Cação  
Email: josemaria@ua.pt  
TEMA - Centro de Tecnologia Mecânica e Automação, Universidade de Aveiro [📍](https://www.google.com/maps/place/Centro+de+Tecnologia+Mec%C3%A2nica+e+Automa%C3%A7%C3%A3o+(TEMA)/@40.6296999,-8.6606291,17z/data=!3m1!4b1!4m6!3m5!1s0xd23a2ab6e3d01cf:0x659e4ff0e75dd370!8m2!3d40.6296999!4d-8.6580488!16s%2Fg%2F11g6xwqw4c?entry=ttu&g_ep=EgoyMDI1MTExNy4wIKXMDSoASAFQAw%3D%3D)