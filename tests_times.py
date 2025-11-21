import pandas as pd
import yaml

from tests import choose_model, pre_process, retrieve_optimised_hyperparams
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from decode import RegressionTree, ClassificationTree, Timer


# Global vars
SEED = 17
USE_OPT = False
with open('default_hyperparams.yaml', 'r') as f:
    DEFAULT_PARAMS = yaml.safe_load(f)
N_TESTS = 5

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

        # Auxiliary dataframe
        clf_time_results_df = pd.DataFrame()
        
        # Iterating for all classification models
        for model_name in ['logreg', 'knn', 'mlp', 'svm']:

            # Retrieve DECODE hyperparams + fit
            if USE_OPT:
                decode_hyperparams = retrieve_optimised_hyperparams(dataset_name, model_name, model_type = 'classification')
            else:
                decode_hyperparams = DEFAULT_PARAMS['classification']            

            all_attempts_df = pd.DataFrame()
            
            # Conducting 'N_TESTS' number of tests
            for i in range(N_TESTS):
            
                # Instantiate a timer instance 
                timer = Timer()

                # Default classifier
                clf = choose_model(model_name, SEED) 
                timer.start('clf fitting')
                clf.fit(clf_trainX, clf_trainY)
                timer.stop('clf fitting')

                # Fitting the decode tree (already has the timers embedded!)
                decode = ClassificationTree(model = clf, scalerX = scaler, dataset = tree_trainX, targets = tree_trainY, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)
                decode.build_tree(**decode_hyperparams)

                # Fitting a Decision Tree Classifier with similar hyperparamters
                dt = DecisionTreeClassifier(criterion = decode_hyperparams['criterium'], max_depth = decode_hyperparams['max_depth'], min_samples_split = decode_hyperparams['min_samples_split'], min_impurity_decrease = decode_hyperparams['min_impurity_decrease'], random_state = SEED)
                timer.start('dt fitting')
                dt.fit(tree_trainX, tree_trainY)
                timer.stop('dt fitting')

                # Concatenating the results
                aux_df = pd.concat([pd.DataFrame(timer.times, index = [i]), pd.DataFrame(decode.timers.times, index = [i])], axis = 1)
                all_attempts_df = pd.concat([all_attempts_df, aux_df], axis = 0)

                # Delete all model instances + timer
                del clf, dt, timer, decode
        
            # Calculating average values and standard deviation
            avg_vals = all_attempts_df.mean().values
            std_vals = all_attempts_df.std().values
            aux_results = dict()
            for avg_val, std_val, col_name in zip(avg_vals, std_vals, all_attempts_df.columns):
                aux_results[f'{col_name} mean (s)'] = avg_val
                aux_results[f'{col_name} std (s)'] = std_val
            clf_time_results_df = pd.concat([clf_time_results_df, pd.DataFrame(aux_results, index = [model_name])], axis = 0)
        
        # Store results
        clf_time_results_df.to_csv(f'./results/{dataset_name}_time_tests.csv')

    #############################
    # 2. REGRESSION DATASETS
    #############################

    # Regression datasets
    for dataset_name in ['housing', 'diamonds']:
        
        # Pre-process data
        trainX, testX, trainY, testY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled, scalerX, scalerY, feature_names, target_names, feature_types = pre_process(dataset_name, seed = SEED)
        reg_trainX, reg_trainY, reg_testX, reg_testY = trainX_scaled, trainY_scaled, testX_scaled, testY_scaled
        tree_trainX, tree_trainY, tree_testX, tree_testY = trainX, trainY, testX, testY

        # Auxiliary dataframe
        reg_time_results_df = pd.DataFrame()

        for model_name in ['linreg', 'knn_r', 'mlp_r', 'svr']:

            # Retrieve DECODE hyperparams + fit
            if USE_OPT:
                decode_hyperparams = retrieve_optimised_hyperparams(dataset_name, model_name, model_type = 'regression')
            else:
                decode_hyperparams = DEFAULT_PARAMS['regression']         

            all_attempts_df = pd.DataFrame()

            # Conducting 'N_TESTS' number of tests
            for i in range(N_TESTS):
                
                # Instantiate a timer instance 
                timer = Timer()            
                
                # Default regressor
                reg = choose_model(model_name, SEED) 
                timer.start('reg fitting')
                reg.fit(reg_trainX, reg_trainY)
                timer.stop('reg fitting')

                # Fitting the decode tree
                decode = RegressionTree(model = reg, scalerX = scalerX, scalerY = scalerY, dataset = tree_trainX, targets = tree_trainY, feature_names = feature_names, target_names = target_names, feature_types = feature_types, seed = SEED)
                decode.build_tree(**decode_hyperparams)

                # Fitting a Decision Tree Classifier with similar hyperparamters
                if decode_hyperparams['criterium'] == 'mae':
                    criterium = 'absolute_error'
                else:
                    criterium = 'squared_error'

                dt = DecisionTreeRegressor(criterion = criterium, max_depth = decode_hyperparams['max_depth'], min_samples_split = decode_hyperparams['min_samples_split'], min_impurity_decrease = decode_hyperparams['min_impurity_decrease'], random_state = SEED)
                timer.start('dt fitting')
                dt.fit(tree_trainX, tree_trainY)
                timer.stop('dt fitting')
                ##################################################### 

                aux_df = pd.concat([pd.DataFrame(timer.times, index = [i]), pd.DataFrame(decode.timers.times, index = [i])], axis = 1)
                all_attempts_df = pd.concat([all_attempts_df, aux_df], axis = 0)

                # Delete all model instances + timer
                del reg, decode, dt, timer  

            # Calculating average values and standard deviation
            avg_vals = all_attempts_df.mean().values
            std_vals = all_attempts_df.std().values
            aux_results = dict()
            for avg_val, std_val, col_name in zip(avg_vals, std_vals, all_attempts_df.columns):
                aux_results[f'{col_name} mean (s)'] = avg_val
                aux_results[f'{col_name} std (s)'] = std_val
            reg_time_results_df = pd.concat([reg_time_results_df, pd.DataFrame(aux_results, index = [model_name])], axis = 0)

        # Store results
        reg_time_results_df.to_csv(f'./results/{dataset_name}_time_tests.csv')