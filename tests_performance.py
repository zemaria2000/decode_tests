import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os
import yaml
from tests import choose_model, pre_process, retrieve_optimised_hyperparams
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import matthews_corrcoef, r2_score
from decode import RegressionTree, ClassificationTree
from sklearn.tree import plot_tree

# Global vars
SEED = 17
USE_OPT = True
with open('default_hyperparams.yaml', 'r') as f:
    DEFAULT_PARAMS = yaml.safe_load(f)


if __name__ == '__main__':

    #############################
    # 1. CLASSIFICATION DATASETS
    #############################

    # Classification datasets
    for dataset_name in ['hawks', 'penguins', 'mushrooms']:

        # Retrieve pre-processed data
        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, scaler, feature_names, target_names, feature_types = pre_process(dataset_name, SEED)
        if dataset_name == 'mushrooms':
            scaler = None
        # Defining the trees' and classifier's train and test data
        if dataset_name != 'mushrooms':   # Mushrooms is only categoricals, it doesn't need any normalisation
            clf_trainX, clf_trainY, clf_testX, clf_testY = trainX_scaled, trainY, testX_scaled, testY
        else:
            clf_trainX, clf_trainY, clf_testX, clf_testY = trainX, trainY, testX, testY
        tree_trainX, tree_trainY, tree_testX, tree_testY = trainX, trainY, testX, testY

        clf_results_df = pd.DataFrame()
        
        # Model fitting + Results
        ##################################################### 
        for model_name in ['logreg', 'knn', 'mlp', 'svm']:

            # Fit the classifier
            clf = choose_model(model_name, SEED)      
            clf.fit(clf_trainX, clf_trainY)
        
            # Retrieve DECODE hyperparams + fit
            if USE_OPT:
                decode_hyperparams = retrieve_optimised_hyperparams(dataset_name, model_name, model_type = 'classification')
            else:
                decode_hyperparams = DEFAULT_PARAMS['classification']
            decode = ClassificationTree(model = clf, scalerX = scaler, dataset = tree_trainX, targets = tree_trainY, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)
            decode.build_tree(**decode_hyperparams)

            # Fit a DT classifier with similar hyperparams
            dt = DecisionTreeClassifier(criterion = decode_hyperparams['criterium'], max_depth = decode_hyperparams['max_depth'], min_samples_split = decode_hyperparams['min_samples_split'], min_impurity_decrease = decode_hyperparams['min_impurity_decrease'], random_state = SEED)
            dt.fit(tree_trainX, tree_trainY)

            # Predictions + performance
            clf_preds = clf.predict(clf_testX)
            decode_preds, _ = decode.make_predictions(tree_testX)
            dt_preds = dt.predict(tree_testX)
            clf_mcc = matthews_corrcoef(clf_testY, clf_preds)
            decode_mcc = matthews_corrcoef(tree_testY, decode_preds)
            dt_mcc = matthews_corrcoef(tree_testY, dt_preds)

            results = dict()
            results['clf'] = clf_mcc
            results['decode'] = decode_mcc
            results['dt'] = dt_mcc

            clf_results_df = pd.concat((clf_results_df, pd.DataFrame(results, index = [model_name])), axis = 0)
        
            # Plots 
            if USE_OPT:
                decode_file_name = f'{dataset_name}_{model_name}_decode_opt'
                sklearn_file_name = f'{dataset_name}_{model_name}_sklearn_opt'
            else:
                decode_file_name = f'{dataset_name}_{model_name}_decode'
                sklearn_file_name = f'{dataset_name}_{model_name}_sklearn'                
            decode.plot_tree(save_fig = True, file_name = decode_file_name)
            # sklearn tree
            fig = plt.figure(figsize = (20, 10))
            plot_tree(dt, feature_names = feature_names, class_names = target_names, filled = True, rounded = True)
            plt.savefig(os.path.join(decode.storage.folders.figs, f'{sklearn_file_name}.pdf'), dpi = 300)
            plt.close(fig)

        # Store the global results for the dataset
        clf_results_df.to_csv(os.path.join('./results', f'{dataset_name}_performance.csv'))


    #############################
    # 2. REGRESSION DATASETS
    #############################

    # Regression datasets
    for dataset_name in ['housing', 'diamonds']:

        # Pre-process data
        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types = pre_process(dataset_name, seed = SEED)
        reg_trainX, reg_trainY, reg_testX, reg_testY = trainX_scaled, trainY_scaled, testX_scaled, testY_scaled
        tree_trainX, tree_trainY, tree_testX, tree_testY = trainX, trainY, testX, testY

        reg_results_df = pd.DataFrame()
        
        # Model fitting + Results
        ##################################################### 
        for model_name in ['linreg', 'knn_r', 'mlp_r', 'svr']:
            # Fit the classifier
            reg = choose_model(model_name)      
            reg.fit(reg_trainX, reg_trainY)
        
            # Retrieve DECODE hyperparams + fit
            if USE_OPT:
                decode_hyperparams = retrieve_optimised_hyperparams(dataset_name, model_name, model_type = 'regression')
            else:
                decode_hyperparams = DEFAULT_PARAMS['regression']
            decode = RegressionTree(model = reg, scalerX = scalerX, scalerY = scalerY, dataset = tree_trainX, targets = tree_trainY, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)
            decode.build_tree(**decode_hyperparams)

            # Fit a DT regressor with similar hyperparams
            if decode_hyperparams['criterium'] == 'mae':
                criterium = 'absolute_error'
            else:
                criterium = 'squared_error'
            dt = DecisionTreeRegressor(criterion = criterium, max_depth = decode_hyperparams['max_depth'], min_samples_split = decode_hyperparams['min_samples_split'], min_impurity_decrease = decode_hyperparams['min_impurity_decrease'], random_state = SEED)
            dt.fit(tree_trainX, tree_trainY)

            # Predictions + performance
            reg_preds = reg.predict(reg_testX)
            decode_preds, _ = decode.make_predictions(tree_testX)
            dt_preds = dt.predict(tree_testX)
            reg_r2 = r2_score(reg_testY, reg_preds)
            decode_r2 = r2_score(tree_testY, decode_preds)
            dt_r2 = r2_score(tree_testY, dt_preds)

            results = dict()
            results['reg'] = reg_r2
            results['decode'] = decode_r2
            results['dt'] = dt_r2

            reg_results_df = pd.concat((reg_results_df, pd.DataFrame(results, index = [model_name])), axis = 0)
        
            # Plots 
            if USE_OPT:
                decode_file_name = f'{dataset_name}_{model_name}_decode_opt'
                sklearn_file_name = f'{dataset_name}_{model_name}_sklearn_opt'
            else:
                decode_file_name = f'{dataset_name}_{model_name}_decode'
                sklearn_file_name = f'{dataset_name}_{model_name}_sklearn'                
            decode.plot_tree(plot_boxplots = True, save_fig = True, file_name = decode_file_name)
            # sklearn tree
            fig = plt.figure(figsize = (20, 10))
            plot_tree(dt, feature_names = feature_names, class_names = target_names, filled = True, rounded = True)
            plt.savefig(os.path.join(decode.storage.folders.figs, f'{sklearn_file_name}.pdf'), dpi = 300)
            plt.close(fig)

        # Store the global results for the dataset
        reg_results_df.to_csv(os.path.join('./results', f'{dataset_name}_performance.csv'))

    