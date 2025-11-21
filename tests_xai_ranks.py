import pandas as pd
import yaml
from tests import *
import exectimeit
from decode import RegressionTree, ClassificationTree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Global vars
SEED = 17
USE_OPT = True
with open('default_hyperparams.yaml', 'r') as f:
    DEFAULT_PARAMS = yaml.safe_load(f)
rng = np.random.default_rng(SEED)
palette = sns.color_palette("Blues", as_cmap=True)


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
        
        # Limiting dataset size to 10k samples.
        if len(trainX) > 10000:
            idx_10000 = rng.choice(len(trainX), 10000, replace = False)
            X_clf, y_clf = clf_trainX[idx_10000], clf_trainY[idx_10000]
            X_tree, y_tree = tree_trainX[idx_10000], tree_trainY[idx_10000] 
        else:
            X_clf, y_clf = clf_trainX, clf_trainY
            X_tree, y_tree = tree_trainX, tree_trainY         

        # Choosing also random index for background samples (10% of the original dataset size)
        idx_back = rng.choice(len(X_clf), int(len(X_clf)/10), replace = False)
        back_samples, back_labels = X_clf[idx_back], y_clf[idx_back]

        for model_name in ['logreg', 'mlp', 'svm', 'knn']:

            # Load decode hyperparameters
            if USE_OPT:
                decode_hyperparams = retrieve_optimised_hyperparams(dataset_name, model_name, model_type = 'classification')
            else:
                decode_hyperparams = DEFAULT_PARAMS['classification']             

            # Fit the classifier
            clf = choose_model(model_name, SEED)      
            clf.fit(clf_trainX, clf_trainY)

            # Instantiate the explainers + DECODE + DT model
            # SHAP
            masker = shap.maskers.Independent(back_samples)
            shap_explainer = shap.Explainer(clf.predict, masker)  
            # LIME 
            lime_explainer = LimeTabularExplainer(training_data = back_samples, mode = 'classification', training_labels = back_labels, verbose = False)
            # PFI 
            pfi_explainer = alibi.explainers.PermutationImportance(predictor = lambda x: clf.predict(x), score_fns = mcc_scoring_function)
            # DECODE
            decode = ClassificationTree(model = clf, scalerX = scaler, dataset = X_tree, targets = y_tree, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)
            decode.build_tree(**decode_hyperparams)
            # DT model
            dt = DecisionTreeClassifier(criterion = decode_hyperparams['criterium'], max_depth = decode_hyperparams['max_depth'], min_samples_split = decode_hyperparams['min_samples_split'], min_impurity_decrease = decode_hyperparams['min_impurity_decrease'], random_state = SEED)
            dt.fit(X_tree, y_tree)            

            # Feature importances
            # SHAP importances
            shap_importances, _ = global_shap_values(samples = X_clf, explainer = shap_explainer)
            norm_shap_importances = shap_importances/np.sum(np.abs(shap_importances))
            # LIME importances
            norm_lime_importances = lime_global(samples = X_clf, model = clf, explainer = lime_explainer, model_type = 'clf')
            # PFI importances
            pfi_importances = permutation_feature_importance(input_X = X_clf, input_Y = y_clf, explainer = pfi_explainer)
            norm_pfi_importances = pfi_importances/np.sum(pfi_importances)
            # tree importances
            norm_tree_importances = decode.compute_feature_importance()
            # DT importances
            norm_dt_importances = dt.feature_importances_

            # Feature rankings
            num_features_tree = np.sum(norm_tree_importances > 0)
            # Retrieving all feature rankings
            shap_ranks = get_ranks(norm_shap_importances)
            shap_ranks = np.where(shap_ranks <= num_features_tree, shap_ranks, num_features_tree + 1)
            lime_ranks = get_ranks(norm_lime_importances)
            lime_ranks = np.where(lime_ranks <= num_features_tree, lime_ranks, num_features_tree + 1)
            pfi_ranks = get_ranks(norm_pfi_importances)
            pfi_ranks = np.where(pfi_ranks <= num_features_tree, pfi_ranks, num_features_tree + 1)
            tree_ranks = get_ranks(norm_tree_importances)
            tree_ranks = np.where(tree_ranks <= num_features_tree, tree_ranks, num_features_tree + 1)
            dt_ranks = get_ranks(norm_dt_importances)
            dt_ranks = np.where(dt_ranks <= num_features_tree, dt_ranks, num_features_tree + 1)

            # Processing the reuslts
            rank_df = pd.DataFrame({
                'SHAP': shap_ranks,
                'LIME': lime_ranks,
                'PFI': pfi_ranks,
                'DECODE': tree_ranks,
                'DT': dt_ranks
            }, index = [feature_name for feature_name in feature_names])
            # Importances dataframe
            columns = [feature_name for feature_name in feature_names]
            idx = ['SHAP', 'LIME', 'PFI', 'DECODE', 'DT']
            imps = [norm_shap_importances, norm_lime_importances, norm_pfi_importances, norm_tree_importances, norm_dt_importances]
            imp_df = pd.DataFrame(data = imps, columns = columns, index = idx)
            rank_df.to_csv(f'./results/{dataset_name}_{model_name}_feature_ranks.csv')
            imp_df.to_csv(f'./results/{dataset_name}_{model_name}_feature_importances.csv')

            # Storing the correlation matrix
            plt.figure(figsize=(6, 5))
            corr = rank_df.corr(method = 'spearman')
            sns.heatmap(corr, annot = True, cmap = palette, vmin = 0, vmax = 1)
            # plt.title("Rank Correlation Between Feature Importance Methods")
            plt.savefig(f'./figures/{dataset_name}_{model_name}_corr_matrix.pdf', dpi = 300)


    #############################
    # 2. REGRESSION DATASETS
    #############################

    for dataset_name in ['housing', 'diamonds']:

        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types = pre_process(dataset_name, seed = SEED)
        reg_trainX, reg_trainY, reg_testX, reg_testY = trainX_scaled, trainY_scaled, testX_scaled, testY_scaled
        tree_trainX, tree_trainY, tree_testX, tree_testY = trainX, trainY, testX, testY
        
        # Limiting dataset size to 10k samples.
        if len(trainX) > 10000:
            idx_10000 = rng.choice(len(trainX), 10000, replace = False)
            X_reg, y_reg = reg_trainX[idx_10000], reg_trainY[idx_10000]
            X_tree, y_tree = tree_trainX[idx_10000], tree_trainY[idx_10000] 
        else:
            X_reg, y_reg = reg_trainX, reg_trainY
            X_tree, y_tree = tree_trainX, tree_trainY 

        # Choosing also random index for background samples (10% of the original dataset size)
        idx_back = rng.choice(len(X_reg), int(len(X_reg)/10), replace = False)
        back_samples, back_targets = X_reg[idx_back], y_reg[idx_back]

        for model_name in ['linreg', 'mlp_r', 'knn_r', 'svr']:

            # Load decode hyperparameters
            if USE_OPT:
                decode_hyperparams = retrieve_optimised_hyperparams(dataset_name, model_name, model_type = 'regression')
            else:
                decode_hyperparams = DEFAULT_PARAMS['regression']             

            # Fit the regressor
            reg = choose_model(model_name, SEED)      
            reg.fit(X_reg, y_reg)

            # Instantiate the explainers + DECODE + DT model
            # SHAP
            masker = shap.maskers.Independent(back_samples)
            shap_explainer = shap.Explainer(reg.predict, masker)  
            # LIME 
            lime_explainer = LimeTabularExplainer(training_data = back_samples, mode = 'regression', training_labels = back_targets, verbose = False)
            # PFI 
            pfi_explainer = alibi.explainers.PermutationImportance(predictor = lambda x: reg.predict(x), score_fns = r2_scoring_function)
            # DECODE
            decode = RegressionTree(model = reg, scalerX = scalerX, scalerY = scalerY, dataset = X_tree, targets = y_tree, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)
            decode.build_tree(**decode_hyperparams)
            # DT model
            if decode_hyperparams['criterium'] == 'mae':
                criterium = 'absolute_error'
            else:
                criterium = 'squared_error'
            dt = DecisionTreeRegressor(criterion = criterium, max_depth = decode_hyperparams['max_depth'], min_samples_split = decode_hyperparams['min_samples_split'], min_impurity_decrease = decode_hyperparams['min_impurity_decrease'], random_state = SEED)
            dt.fit(X_tree, y_tree)            

            # Feature importances
            # SHAP importances
            shap_importances, _ = global_shap_values(samples = X_reg, explainer = shap_explainer)
            norm_shap_importances = shap_importances/np.sum(np.abs(shap_importances))
            # LIME importances
            norm_lime_importances = lime_global(samples = X_reg, model = reg, explainer = lime_explainer, model_type = 'reg')
            # PFI importances
            pfi_importances = permutation_feature_importance(input_X = X_reg, input_Y = y_reg, explainer = pfi_explainer)
            norm_pfi_importances = pfi_importances/np.sum(pfi_importances)
            # tree importances
            norm_tree_importances = decode.compute_feature_importance()
            # DT importances
            norm_dt_importances = dt.feature_importances_

            # Feature rankings
            num_features_tree = np.sum(norm_tree_importances > 0)
            # Retrieving all feature rankings
            shap_ranks = get_ranks(norm_shap_importances)
            shap_ranks = np.where(shap_ranks <= num_features_tree, shap_ranks, num_features_tree + 1)
            lime_ranks = get_ranks(norm_lime_importances)
            lime_ranks = np.where(lime_ranks <= num_features_tree, lime_ranks, num_features_tree + 1)
            pfi_ranks = get_ranks(norm_pfi_importances)
            pfi_ranks = np.where(pfi_ranks <= num_features_tree, pfi_ranks, num_features_tree + 1)
            tree_ranks = get_ranks(norm_tree_importances)
            tree_ranks = np.where(tree_ranks <= num_features_tree, tree_ranks, num_features_tree + 1)
            dt_ranks = get_ranks(norm_dt_importances)
            dt_ranks = np.where(dt_ranks <= num_features_tree, dt_ranks, num_features_tree + 1)

            # Processing the reuslts
            rank_df = pd.DataFrame({
                'SHAP': shap_ranks,
                'LIME': lime_ranks,
                'PFI': pfi_ranks,
                'DECODE': tree_ranks,
                'DT': dt_ranks
            }, index = [feature_name for feature_name in feature_names])
            # Importances dataframe
            columns = [feature_name for feature_name in feature_names]
            idx = ['SHAP', 'LIME', 'PFI', 'DECODE', 'DT']
            imps = [norm_shap_importances, norm_lime_importances, norm_pfi_importances, norm_tree_importances, norm_dt_importances]
            imp_df = pd.DataFrame(data = imps, columns = columns, index = idx)
            rank_df.to_csv(f'./results/{dataset_name}_{model_name}_feature_ranks.csv')
            imp_df.to_csv(f'./results/{dataset_name}_{model_name}_feature_importances.csv')

            # Storing the correlation matrix
            plt.figure(figsize=(6, 5))
            corr = rank_df.corr(method = 'spearman')
            sns.heatmap(corr, annot = True, cmap = palette, vmin = 0, vmax=1)
            # plt.title("Rank Correlation Between Feature Importance Methods")
            plt.savefig(f'./figures/{dataset_name}_{model_name}_corr_matrix.pdf', dpi = 300)