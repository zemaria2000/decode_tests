import pandas as pd
import yaml
from tests import *
import exectimeit
from decode import RegressionTree, ClassificationTree

# Global vars
SEED = 17
USE_OPT = False
with open('default_hyperparams.yaml', 'r') as f:
    DEFAULT_PARAMS = yaml.safe_load(f)
N_TESTS = 5
rng = np.random.default_rng(SEED)


if __name__ == '__main__':

    #############################
    # 1. CLASSIFICATION DATASETS
    #############################

    for dataset_name in ['hawks', 'penguins', 'mushrooms']:

        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types = pre_process(dataset_name, SEED)
        if dataset_name == 'mushrooms':
            scaler = None
        # Defining the trees' and classifier's train and test data
        if dataset_name != 'mushrooms':   # Mushrooms is only categoricals, it doesn't need any normalisation
            clf_trainX, clf_trainY, clf_testX, clf_testY = trainX_scaled, trainY, testX_scaled, testY
        else:
            clf_trainX, clf_trainY, clf_testX, clf_testY = trainX, trainY, testX, testY
        tree_trainX, tree_trainY, tree_testX, tree_testY = trainX, trainY, testX, testY

        # Choose 100 samples at random for conducting explanations. Otherwise, it would take to much time...
        if len(trainX) > 100:
            idx_100 = rng.choice(len(trainX), 100, replace = False)
            X_clf, y_clf = clf_trainX[idx_100], clf_trainY[idx_100]
            X_tree, y_tree = tree_trainX[idx_100], tree_trainY[idx_100] 
        else:
            X_clf, y_clf = clf_trainX, clf_trainY
            X_tree, y_tree = tree_trainX, tree_trainY 
        
        # Choosing also another random index for background samples (10% of the original 100)
        idx_back = rng.choice(len(X_clf), int(len(X_clf)/10), replace = False)
        back_samples, back_labels = X_clf[idx_back], y_clf[idx_back]

        for model_name in ['logreg', 'knn', 'mlp', 'svm']:

            # Load decode hyperparameters
            if USE_OPT:
                decode_hyperparams = retrieve_optimised_hyperparams(dataset_name, model_name, model_type = 'classification')
            else:
                decode_hyperparams = DEFAULT_PARAMS['classification']             

            # Fit the classifier
            clf = choose_model(model_name, SEED)      
            clf.fit(clf_trainX, clf_trainY)

            # Instantiate the explainers + DECODE
            # SHAP
            masker = shap.maskers.Independent(back_samples)
            shap_explainer = shap.Explainer(clf.predict, masker)  
            # LIME 
            lime_explainer = LimeTabularExplainer(training_data = back_samples, mode = 'classification', training_labels = back_labels, verbose = False)
            # PFI 
            pfi_explainer = alibi.explainers.PermutationImportance(predictor = lambda x: clf.predict(x), score_fns = mcc_scoring_function)
            # DECODE
            decode = ClassificationTree(model = clf, scalerX = scaler, dataset = X_tree, targets = y_tree, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)

            ### Conduct global analyses - exectimeit library
            shap_global_times, shap_global_std, _ = exectimeit.timeit.timeit(N_TESTS, global_shap_values, X_clf, shap_explainer)
            lime_global_times, lime_global_std, _ = exectimeit.timeit.timeit(N_TESTS, lime_global, samples = X_clf, model = clf, explainer = lime_explainer, model_type = 'clf')
            pfi_global_times,  pfi_global_std,  _ = exectimeit.timeit.timeit(N_TESTS, permutation_feature_importance, input_X = X_clf, input_Y = y_clf, explainer = pfi_explainer)
            tree_global_times, tree_global_std, _ = exectimeit.timeit.timeit(N_TESTS, decode.build_tree, **decode_hyperparams)            

            ### Conduct local analyses - exectimeit library
            # Random sample 
            rand_idx = rng.choice(len(X_clf), 1)
            rand_sample, rand_tree_sample = X_clf[rand_idx], X_tree[rand_idx]
            # Generate a tree for assessment
            del decode
            decode = ClassificationTree(model = clf, scalerX = scaler, dataset = X_tree, targets = y_tree, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)
            decode.build_tree(**decode_hyperparams)
            # Local analyses
            shap_local_times, shap_local_std, _ = exectimeit.timeit.timeit(N_TESTS, local_shap_values, rand_sample, shap_explainer)
            lime_local_times, lime_local_std, _ = exectimeit.timeit.timeit(N_TESTS, lime_values, rand_sample.squeeze(), clf, lime_explainer, 'clf')
            pfi_local_times,  pfi_local_std     = None, None
            tree_local_times, tree_local_std, _ = exectimeit.timeit.timeit(N_TESTS, decode.make_predictions, rand_tree_sample) 

            # Compiling results
            global_cols = ['Global Time (s)', 'Global Std (s)']
            global_results_dict = {
                'SHAP': [shap_global_times, shap_global_std],
                'LIME': [lime_global_times, lime_global_std],
                'PFI':  [pfi_global_times, pfi_global_std],
                'DECODE': [tree_global_times, tree_global_std],
            }

            local_cols = ['Local Time (s)', 'Local Std (s)']
            local_results_dict = {
                'SHAP': [shap_local_times, shap_local_std],
                'LIME': [lime_local_times, lime_local_std],
                'PFI':  [pfi_local_times, pfi_local_std],
                'DECODE': [tree_local_times, tree_local_std],
            }

            global_df = pd.DataFrame(global_results_dict, index = global_cols).T
            local_df = pd.DataFrame(local_results_dict, index = local_cols).T
            overall_df = pd.concat([global_df, local_df], axis = 1)

            overall_df.to_csv(f'./results/{dataset_name}_{model_name}_xai_times.csv')


    #############################
    # 2. REGRESSION DATASETS
    #############################

    # for dataset_name in ['housing', 'diamonds']:
    for dataset_name in ['diamonds']:

        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types = pre_process(dataset_name, seed = SEED)
        reg_trainX, reg_trainY, reg_testX, reg_testY = trainX_scaled, trainY_scaled, testX_scaled, testY_scaled
        tree_trainX, tree_trainY, tree_testX, tree_testY = trainX, trainY, testX, testY

        # Choose 100 samples at random for conducting explanations. Otherwise, it would take to much time...
        if len(trainX) > 100:
            idx_100 = rng.choice(len(trainX), 100, replace = False)
            X_reg, y_reg = reg_trainX[idx_100], reg_trainY[idx_100]
            X_tree, y_tree = tree_trainX[idx_100], tree_trainY[idx_100] 
        else:
            X_reg, y_reg = reg_trainX, reg_trainY
            X_tree, y_tree = tree_trainX, tree_trainY 
        
        # Choosing also another random index for background samples (10% of the original 100)
        idx_back = rng.choice(len(X_reg), int(len(X_reg)/10), replace = False)
        back_samples, back_targets = X_reg[idx_back], y_reg[idx_back]

        for model_name in ['linreg', 'knn_r', 'mlp_r', 'svr']:

            # Load decode hyperparameters
            if USE_OPT:
                decode_hyperparams = retrieve_optimised_hyperparams(dataset_name, model_name, model_type = 'regression')
            else:
                decode_hyperparams = DEFAULT_PARAMS['regression']            

            # Fit the regressor
            reg = choose_model(model_name, SEED)      
            reg.fit(reg_trainX, reg_trainY)

            # Instantiate the explainers + DECODE
            # SHAP
            masker = shap.maskers.Independent(back_samples)
            shap_explainer = shap.Explainer(reg.predict, masker)  
            # LIME 
            lime_explainer = LimeTabularExplainer(training_data = back_samples, mode = 'regression', training_labels = back_targets, verbose = False)
            # PFI 
            pfi_explainer = alibi.explainers.PermutationImportance(predictor = lambda x: reg.predict(x), score_fns = r2_scoring_function)
            # DECODE
            decode = RegressionTree(model = reg, scalerX = scalerX, scalerY = scalerY, dataset = X_tree, targets = y_tree, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)

            ### Conduct global analyses - exectimeit library
            shap_global_times, shap_global_std, _ = exectimeit.timeit.timeit(N_TESTS, global_shap_values, X_reg, shap_explainer)
            lime_global_times, lime_global_std, _ = exectimeit.timeit.timeit(N_TESTS, lime_global, samples = X_reg, model = reg, explainer = lime_explainer, model_type = 'reg')
            pfi_global_times,  pfi_global_std,  _ = exectimeit.timeit.timeit(N_TESTS, permutation_feature_importance, input_X = X_reg, input_Y = y_reg, explainer = pfi_explainer)
            tree_global_times, tree_global_std, _ = exectimeit.timeit.timeit(N_TESTS, decode.build_tree, **decode_hyperparams)            

            ### Conduct local analyses - exectimeit library
            # Random sample 
            rand_idx = rng.choice(len(X_reg), 1)
            rand_sample, rand_tree_sample = X_reg[rand_idx], X_tree[rand_idx]
            # Generate a tree for assessment
            del decode
            decode = RegressionTree(model = reg, scalerX = scalerX, scalerY = scalerY, dataset = X_tree, targets = y_tree, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)
            decode.build_tree(**decode_hyperparams)
            # Local analyses
            shap_local_times, shap_local_std, _ = exectimeit.timeit.timeit(N_TESTS, local_shap_values, rand_sample, shap_explainer)
            lime_local_times, lime_local_std, _ = exectimeit.timeit.timeit(N_TESTS, lime_values, rand_sample.squeeze(), reg, lime_explainer, 'reg')
            pfi_local_times,  pfi_local_std     = None, None
            tree_local_times, tree_local_std, _ = exectimeit.timeit.timeit(N_TESTS, decode.make_predictions, rand_tree_sample) 

            # Compiling results
            global_cols = ['Global Time (s)', 'Global Std (s)']
            global_results_dict = {
                'SHAP': [shap_global_times, shap_global_std],
                'LIME': [lime_global_times, lime_global_std],
                'PFI':  [pfi_global_times, pfi_global_std],
                'DECODE': [tree_global_times, tree_global_std],
            }

            local_cols = ['Local Time (s)', 'Local Std (s)']
            local_results_dict = {
                'SHAP': [shap_local_times, shap_local_std],
                'LIME': [lime_local_times, lime_local_std],
                'PFI':  [pfi_local_times, pfi_local_std],
                'DECODE': [tree_local_times, tree_local_std],
            }

            global_df = pd.DataFrame(global_results_dict, index = global_cols).T
            local_df = pd.DataFrame(local_results_dict, index = local_cols).T
            overall_df = pd.concat([global_df, local_df], axis = 1)

            overall_df.to_csv(f'./results/{dataset_name}_{model_name}_xai_times.csv')