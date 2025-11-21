# Numerical results

This folder contains several .csv files that were saved for the numerical analysis of the results. These results were then compiled and several summary tables were developed and are presented in this README file.  
Regarding these results files, here is a short description of how they are organised:

* `"<dataset_name>_performance.csv"`: for each dataset, this csv contemplates the performance of all baseline models used (classifiers or regressors), their DECODE representation and a Decision Tree with the same hyperparameters as DECODE.
* `"<dataset_name>_time_tests.csv"`: for each dataset, each file summarises fitting times regarding all baseline models (classifiers or regressors), their derived DECODE representation and a Decision Tree with the same hyperparameters as DECODE. Additionally, DECODE's times are split into inference times by its baseline model and DECODE's remaining operations.
* `"<dataset_name>_<model_name>_xai_times.csv"`: for each dataset-baseline model combination, these files demonstrate the local and global explanation times by each feature ranking approach (SHAP, LIME, PFI) compared to the DECODE times to generate the same types of explanations.
* `"<dataset_name>_<model_name>_feature_ranks.csv"`: for each dataset-baseline model combination, these files include the feature ranking scores that resulted from the application of global rankings with the three state-of-the-art approaches (SHAP, LIME, PFI), DECODE and Decision Trees. These ranks are then plotted through a correlation matrix in the [figures](https://github.com/zemaria2000/decode_tests/tree/main/figures) folder.
* `"<dataset_name>_<model_name>_<feature_importances>.csv"`: summary of the feature importances given by the three state-of-the-art XAI methods assessed (SHAP, LIME, PFI), a Decision Tree and DECODE based on each model (model_name) and for each dataset (dataset_name). **Note**: these results are not showcased in the paper.


## Complete set of numerical results

Below, a series of overall tables are presented concerning the **complete set of results**


### Performance analysis

Tables that comprise all possible results for the classification datasets (MCC scores) and for the regression datasets (MAE scores).

| Dataset | Model | Model MCC| DECODE MCC | DT MCC |
|---------|-------|----------|------------|--------|
| Hawks     | LogReg | 0.956 | 0.945 | 0.945 |
|           | KNN    | 0.956 | 0.945 | 0.956 |
|           | MLP    | 0.967 | 0.945 | 0.956 |
|           | SVM    | 0.967 | 0.945 | 0.956 |
| Penguins  | LogReg | 1.000 | 0.933 | 0.874 |
|           | KNN    | 1.000 | 0.913 | 0.874 |
|           | MLP    | 1.000 | 0.931 | 0.931 |
|           | SVM    | 1.000 | 0.933 | 0.874 |
| Mushrooms | LogReg | 0.916 | 0.921 | 0.962 |
|           | KNN    | 0.996 | 0.963 | 0.963 |
|           | MLP    | 1.000 | 0.963 | 0.963 |
|           | SVM    | 0.984 | 0.958 | 0.953 |


| Dataset | Model | Model MAE | DECODE MAE | DT MAE |
|---------|-------|-----------|------------|--------|
| Housing     | LinReg | 0.368 | 0.381 | 0.332 |
|             | KNN    | 0.271 | 0.284 | 0.327 |
|             | MLP    | 0.368 | 0.335 | 0.332 |
|             | SVR    | 0.234 | 0.295 | 0.317 |
| Dimaonds    | LinReg | 0.214 | 0.227 | 0.148 |
|             | KNN    | 0.098 | 0.157 | 0.152 |
|             | MLP    | 0.116 | 0.171 | 0.148 |
|             | SVR    | 0.122 | 0.170 | 0.148 |


### Fitting Times Table

| Dataset | "Black-box" model | DT model | **DECODE**<br>Overall | **DECODE**<br>“Black-box” preds | **DECODE**<br>building |
|--------|-------------------|----------|------------------------|----------------------------------|-------------------------|
| **Hawks (clf)** | 0.0026 ± 0.0003<br>(LogReg) | 0.0010 ± 0.0001 | 0.0210 ± 0.0009 | 0.0010 ± 0.0001 (4.6%) | 0.0200 ± 0.0009 (95.4%) |
| | 0.0005 ± 0.0000<br>(KNN) | 0.0009 ± 0.0000 | 0.0568 ± 0.0007 | 0.0345 ± 0.0005 (60.7%) | 0.0223 ± 0.0003 (39.3%) |
| | 0.0818 ± 0.0006<br>(MLP) | 0.0009 ± 0.0000 | 0.0152 ± 0.0005 | 0.0010 ± 0.0001 (6.5%)  | 0.0142 ± 0.0004 (93.5%) |
| | 0.0047 ± 0.0000<br>(SVM) | 0.0010 ± 0.0001 | 0.0182 ± 0.0003 | 0.0029 ± 0.0001 (15.7%) | 0.0153 ± 0.0004 (84.3%) |
| **Mushrooms (clf)** | 1.3099 ± 0.1381<br>(LogReg) | 0.0038 ± 0.0001 | 0.0883 ± 0.0096 | 0.0092 ± 0.0034 (10.4%) | 0.0792 ± 0.0092 (89.6%) |
| | 0.0003 ± 0.0000<br>(KNN) | 0.0039 ± 0.0001 | 2.6621 ± 0.0109 | 2.5988 ± 0.1004 (97.6%) | 0.0633 ± 0.0040 (2.4%) |
| | 0.6168 ± 0.0079<br>(MLP) | 0.0046 ± 0.0021 | 0.0857 ± 0.0261 | 0.0090 ± 0.6918 (10.5%) | 0.0767 ± 0.0158 (89.5%) |
| | 0.6064 ± 0.0108<br>(SVM) | 0.0034 ± 0.0001 | 0.7554 ± 0.0082 | 0.6918 ± 0.0071 (91.6%) | 0.0637 ± 0.0014 (8.4%) |
| **Penguins (clf)** | 0.0017 ± 0.0004<br>(LogReg) | 0.0006 ± 0.0001 | 0.0242 ± 0.0003 | 0.0014 ± 0.0001 (5.7%) | 0.0228 ± 0.0003 (94.3%) |
| | 0.0006 ± 0.0000<br>(KNN) | 0.0006 ± 0.0002 | 0.0440 ± 0.0008 | 0.0218 ± 0.0004 (49.6%) | 0.0222 ± 0.0007 (50.4%) |
| | 0.0394 ± 0.0004<br>(MLP) | 0.0006 ± 0.0001 | 0.0228 ± 0.0004 | 0.0018 ± 0.0001 (8.0%)  | 0.0210 ± 0.0004 (92.0%) |
| | 0.0019 ± 0.0000<br>(SVM) | 0.0005 ± 0.0000 | 0.0211 ± 0.0003 | 0.0025 ± 0.0000 (11.7%) | 0.0186 ± 0.0003 (88.3%) |
| **Housing (reg)** | 0.0004 ± 0.0001<br>(LinReg) | 0.0011 ± 0.0001 | 0.1553 ± 0.0023 | 0.0034 ± 0.0002 (2.2%) | 0.1519 ± 0.0022 (97.8%) |
| | 0.0003 ± 0.0000<br>(KNN) | 0.0010 ± 0.0000 | 0.1590 ± 0.0011 | 0.0161 ± 0.0003 (10.1%) | 0.1429 ± 0.0009 (89.9%) |
| | 0.0470 ± 0.0002<br>(MLP) | 0.0011 ± 0.0001 | 0.1654 ± 0.0009 | 0.0038 ± 0.0001 (2.3%) | 0.1616 ± 0.0009 (97.7%) |
| | 0.0033 ± 0.0001<br>(SVR) | 0.0011 ± 0.0000 | 0.1580 ± 0.0011 | 0.0161 ± 0.0003 (10.2%) | 0.1418 ± 0.0011 (89.8%) |
| **Diamonds (reg)** | 0.0072 ± 0.0048<br>(LinReg) | 0.0404 ± 0.0004 | 1.1417 ± 0.0102 | 0.0016 ± 0.0027 (1.4%) | 1.1261 ± 0.0086 (98.6%) |
| | 0.0243 ± 0.0021<br>(KNN) | 0.0383 ± 0.0010 | 3.8154 ± 0.0462 | 2.9410 ± 0.0363 (77.1%) | 0.8745 ± 0.0109 (22.9%) |
| | 0.8918 ± 0.0095<br>(MLP) | 0.0410 ± 0.0004 | 0.9369 ± 0.0051 | 0.0235 ± 0.0022 (2.5%)  | 0.9134 ± 0.0054 (97.5%) |
| | 10.8605 ± 0.0092<br>(SVR) | 0.0384 ± 0.0008 | 58.6196 ± 0.3605 | 57.8790 ± 0.3566 (98.7%) | 0.7406 ± 0.0051 (1.3%) | 



### Global & Local Explanation Times in seconds (Classification Datasets)

| Dataset | Model | XAI method | Global explanations | Local explanations |
|---------|--------|-------------|----------------------|---------------------|
| **Hawks** | LogReg | SHAP   | 0.035974 ± 0.000366 | 0.000384 ± 0.000018 |
|         |        | LIME   | 0.912120 ± 0.024462 | 0.009027 ± 0.000206 |
|         |        | PFI    | 0.193361 ± 0.007271 | - |
|         |        | DECODE | **0.006281 ± 0.000178** | **0.000002 ± 0.000000** |
|         | KNN    | SHAP   | 3.514430 ± 0.019259 | 0.033867 ± 0.000287 |
|         |        | LIME   | 3.745872 ± 0.045484 | 0.037086 ± 0.000530 |
|         |        | PFI    | 0.925163 ± 0.007758 | - |
|         |        | DECODE | **0.012322 ± 0.001255** | **0.000002 ± 0.000000** |
|         | MLP    | SHAP   | 0.055156 ± 0.001424 | 0.000544 ± 0.000046 |
|         |        | LIME   | 0.929224 ± 0.009196 | 0.008976 ± 0.000295 |
|         |        | PFI    | 0.207119 ± 0.004141 | - |
|         |        | DECODE | **0.006567 ± 0.000518** | **0.000002 ± 0.000000** |
|         | SVM    | SHAP   | 0.332078 ± 0.005847 | 0.003010 ± 0.000090 |
|         |        | LIME   | 1.947523 ± 0.016588 | 0.019499 ± 0.000432 |
|         |        | PFI    | 0.256136 ± 0.001859 | - |
|         |        | DECODE | **0.006708 ± 0.000340** | **0.000002 ± 0.000000** |
| **Penguins** | LogReg | SHAP   | 0.028000 ± 0.000807 | 0.000292 ± 0.000037 |
|            |        | LIME   | 0.921572 ± 0.010435 | 0.008818 ± 0.000159 |
|            |        | PFI    | 0.170787 ± 0.003724 | - |
|            |        | DECODE | **0.012396 ± 0.000264** | **0.000003 ± 0.000000** |
|            | KNN    | SHAP   | 1.228633 ± 0.008496 | 0.013382 ± 0.000115 |
|            |        | LIME   | 2.089783 ± 0.025366 | 0.020605 ± 0.000611 |
|            |        | PFI    | 0.751779 ± 0.007963 | - |
|            |        | DECODE | **0.021651 ± 0.001067** | **0.000002 ± 0.000000** |
|            | MLP    | SHAP   | 0.036668 ± 0.001627 | 0.000369 ± 0.000029 |
|            |        | LIME   | 0.931082 ± 0.015616 | 0.008950 ± 0.000142 |
|            |        | PFI    | 0.182820 ± 0.006693 | - |
|            |        | DECODE | **0.020616 ± 0.000778** | **0.000002 ± 0.000000** |
|            | SVM    | SHAP   | 0.129608 ± 0.002136 | 0.001097 ± 0.000060 |
|            |        | LIME   | 1.830867 ± 0.003554 | 0.017935 ± 0.000208 |
|            |        | PFI    | 0.218315 ± 0.003275 | - |
|            |        | DECODE | **0.012966 ± 0.000132** | **0.000003 ± 0.000000** |
| **Mushrooms** | LogReg | SHAP   | 0.142817 ± 0.004305 | 0.001507 ± 0.000037 |
|              |        | LIME   | 2.928304 ± 0.056955 | 0.029215 ± 0.000414 |
|              |        | PFI    | 0.558778 ± 0.003287 | - |
|              |        | DECODE | **0.013185 ± 0.000437** | **0.000003 ± 0.000001** |
|              | KNN    | SHAP   | 26.173918 ± 0.132247 | 0.220737 ± 0.004986 |
|              |        | LIME   | 46.314447 ± 4.661895 | 0.470364 ± 0.024347 |
|              |        | PFI    | 10.747375 ± 0.099554 | - |
|              |        | DECODE | **0.058268 ± 0.004927** | **0.000003 ± 0.000000** |
|              | MLP    | SHAP   | 0.165529 ± 0.005583 | 0.001653 ± 0.000054 |
|              |        | LIME   | 2.482556 ± 0.116127 | 0.024196 ± 0.000258 |
|              |        | PFI    | 0.569686 ± 0.011880 | - |
|              |        | DECODE | **0.013334 ± 0.000388** | **0.000003 ± 0.000000** |
|              | SVM    | SHAP   | 7.385647 ± 0.227214 | 0.064615 ± 0.002130 |
|              |        | LIME   | 16.108573 ± 0.084673 | 0.143080 ± 0.002431 |
|              |        | PFI    | 2.965060 ± 0.026423 | - |
|              |        | DECODE | **0.021708 ± 0.001611** | **0.000003 ± 0.000000** |


## Global & Local Explanation Times in seconds (Regression Datasets)

| Dataset | Model | XAI method | Global explanations | Local explanations |
|---------|--------|-------------|----------------------|---------------------|
| **Housing** | LinReg | SHAP   | 0.225759 ± 0.008480 | 0.002243 ± 0.000053 |
|            |        | LIME   | 1.387532 ± 0.327665 | 0.012707 ± 0.000086 |
|            |        | PFI    | 0.098266 ± 0.001003 | - |
|            |        | DECODE | **0.084001 ± 0.000560** | **0.000003 ± 0.000001** |
|            | KNN    | SHAP   | 1.612116 ± 0.138615 | 0.014459 ± 0.000072 |
|            |        | LIME   | 3.721402 ± 0.232309 | 0.038878 ± 0.004638 |
|            |        | PFI    | 0.366892 ± 0.021993 | - |
|            |        | DECODE | **0.100015 ± 0.000928** | **0.000005 ± 0.000000** |
|            | MLP    | SHAP   | 0.215656 ± 0.002141 | 0.002220 ± 0.000067 |
|            |        | LIME   | 1.248624 ± 0.115854 | 0.013205 ± 0.000167 |
|            |        | PFI    | 0.115140 ± 0.005955 | - |
|            |        | DECODE | **0.091649 ± 0.001937** | **0.000005 ± 0.000000** |
|            | SVR    | SHAP   | 2.509374 ± 0.218522 | 0.023023 ± 0.000054 |
|            |        | LIME   | 5.097950 ± 0.148234 | 0.047992 ± 0.000348 |
|            |        | PFI    | 0.468705 ± 0.044721 | - |
|            |        | DECODE | **0.094374 ± 0.001481** | **0.000005 ± 0.000000** |
| **Diamonds** | LinReg | SHAP   | 0.097564 ± 0.011219 | 0.000950 ± 0.000029 |
|              |        | LIME   | 1.220049 ± 0.008525 | 0.012057 ± 0.000260 |
|              |        | PFI    | 0.071607 ± 0.001803 | - |
|              |        | DECODE | **0.071026 ± 0.001457** | **0.000005 ± 0.000000** |
|              | KNN    | SHAP   | 18.325174 ± 0.019529 | 0.111897 ± 0.001743 |
|              |        | LIME   | 30.036772 ± 0.109127 | 0.302087 ± 0.009096 |
|              |        | PFI    | 1.159112 ± 0.056691 | - |
|              |        | DECODE | **0.080496 ± 0.002376** | **0.000005 ± 0.000000** |
|              | MLP    | SHAP   | 0.089816 ± 0.010036 | 0.000838 ± 0.000016 |
|              |        | LIME   | 1.163369 ± 0.063189 | 0.010399 ± 0.004741 |
|              |        | PFI    | 0.079680 ± 0.000773 | - |
|              |        | DECODE | **0.070574 ± 0.000728** | **0.000005 ± 0.000000** |
|              | SVR    | SHAP   | 111.676717 ± 0.408665 | 1.106345 ± 0.016355 |
|              |        | LIME   | 112.989464 ± 0.483904 | 1.127636 ± 0.015679 |
|              |        | PFI    | 9.956586 ± 0.133395 | - |
|              |        | DECODE | **0.187267 ± 0.002736** | **0.000004 ± 0.000000** |

