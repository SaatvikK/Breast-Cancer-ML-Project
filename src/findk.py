#!/usr/bin/env python
# findk.py -- new feature-selection script modeled after controller pipeline (fixed CFS init)

import pandas as pd
import matplotlib.pyplot as plt
from preprocess import Preprocessor    # cleaning, mapping, splitting, CFS, imputation
# Ensure all models potentially used by find_optimal_k's default are imported
from models import LRC, SVM, NBC, KNN, DTC, DNN  # classifier wrappers


def find_optimal_k(
    data_path: str = "../data/data1.csv",
    set_num: int = 1,
    metric: str = 'accuracy',
    k_max: int = None,
    tau_redundancy: float = 0.8, # Default explicitly set
    test_size: float = 0.2,
    model_classes: list = None,
    dnn_hidden_sizes: list = None
) -> dict:
    """
    For each k:
      1. Load raw data, simple cleaning + mapping
      2. Perform CFS on full mapped DataFrame to select k features
      3. Split selected-DF into train/test
      4. Impute if needed
      5. Train & test each model, record 'metric'
      6. Track best k per model
    Returns {k: {model: val}, ..., 'kOptimal': {model: best_k}}
    """
    if model_classes is None:
        # This default list is kept for broader usability of the function,
        # but will be overridden by the specific list in __main__
        model_classes = [LRC, SVM, NBC, KNN, DTC, DNN]
    if dnn_hidden_sizes is None:
        # Default DNN hidden sizes, will be overridden in __main__
        dnn_hidden_sizes = [13, 8, 5]

    # load and preprocess base data
    df_raw = pd.read_csv(data_path)
    pre_base = Preprocessor(df_raw, setNum=set_num)
    df_clean = pre_base.simpleDataCleaning()
    df_mapped = pre_base.classMapping()

    # determine effective_k_max based on available features and user's k_max
    max_feats = df_mapped.shape[1] - 1 # Number of features excluding target
    if max_feats <= 0:
        raise ValueError("No features found in the data after initial mapping (excluding target variable). Cannot proceed.")


    if k_max is None: # User did not specify k_max
        effective_k_max = max_feats
    elif k_max <= 0: # User specified invalid k_max
        print(f"Warning: k_max ({k_max}) must be positive. Using all available features: {max_feats}.")
        effective_k_max = max_feats
    elif k_max > max_feats: # User specified k_max larger than available
        print(f"Warning: Requested k_max ({k_max}) is greater than available features ({max_feats}). Using k_max = {max_feats}.")
        effective_k_max = max_feats
    else: # User specified valid k_max within available range
        effective_k_max = k_max

    results = {'kOptimal': {}} # Stores {'model_name': {'k': best_k, 'score': best_score}} internally

    for k in range(1, effective_k_max + 1):
        print(f"Processing feature selection for k={k}/{effective_k_max}")
        # CFS on full mapping
        pre_fs = Preprocessor(df_mapped.copy(), setNum=set_num)
        pre_fs.dfTrain = pre_fs.df.copy() # Ensure dfTrain is set for corrFeatureSelection
        fs = pre_fs.corrFeatureSelection(k=k, tauRedundancy=tau_redundancy)
        df_sel = fs['selectedDF']

        # Split selected data
        pre_sp = Preprocessor(df_sel.copy(), setNum=set_num, splittedSet=True)
        df_train, df_test = pre_sp.splitSet(testSize=test_size)
        # Impute if necessary
        df_train, df_test, _ = pre_sp.dataCleaning()

        # Prepare X/y
        X_train = df_train.drop('diagnosis', axis=1)
        y_train = df_train['diagnosis']
        X_test = df_test.drop('diagnosis', axis=1)
        y_test = df_test['diagnosis']

        results[k] = {}
        # Train & evaluate models
        for Model_cls in model_classes:
            model_instance_name = Model_cls.__name__ # Default name, might be overridden by model.modelName

            if X_train.shape[1] == 0:
                print(f"Warning: No features available for training {model_instance_name} at k={k}. Skipping.")
                # Use Model_cls.__name__ as a fallback key if model instantiation fails early
                results[k][model_instance_name] = float('-inf') 
                if model_instance_name not in results['kOptimal']:
                     results['kOptimal'][model_instance_name] = {'k': 1, 'score': float('-inf')}
                continue

            if Model_cls is DNN:
                model = Model_cls(
                    inputSize=X_train.shape[1],
                    hiddenSizes=dnn_hidden_sizes,
                    outputSize=y_train.nunique()
                )
            else:
                model = Model_cls()
            
            name = model.modelName # Actual name from the model instance

            ok_train, err_train, _ = model.basetrain(X_train, y_train)
            if not ok_train:
                print(f"ERROR: Training failed for {name} at k={k}: {err_train}")
                results[k][name] = float('-inf')
                if name not in results['kOptimal']:
                    results['kOptimal'][name] = {'k': 1, 'score': float('-inf')}
                continue

            ok_test, err_test, _ = model.test(X_test)
            if not ok_test:
                print(f"ERROR: Testing failed for {name} at k={k}: {err_test}")
                results[k][name] = float('-inf')
                if name not in results['kOptimal']:
                     results['kOptimal'][name] = {'k': 1, 'score': float('-inf')}
                continue

            perf = model.performanceEval(y_test)
            current_metric_val = perf.get(metric)
            
            if current_metric_val is None:
                print(f"Warning: Metric '{metric}' not found for {name} at k={k}. Using -inf.")
                current_metric_val = float('-inf')
            results[k][name] = current_metric_val

            # Track best k and its score
            if name not in results['kOptimal'] or \
               (isinstance(current_metric_val, (int, float)) and current_metric_val > results['kOptimal'][name]['score']):
                results['kOptimal'][name] = {'k': k, 'score': current_metric_val}
            # Handle case where previous best score was -inf (failure) and current is the first valid score
            elif name in results['kOptimal'] and results['kOptimal'][name]['score'] == float('-inf') and \
                 isinstance(current_metric_val, (int, float)) and current_metric_val > float('-inf'):
                results['kOptimal'][name] = {'k': k, 'score': current_metric_val}


    # Transform kOptimal to the final required format {model: best_k}
    final_kOptimal_dict = {}
    for model_name, data in results['kOptimal'].items():
        if data['score'] > float('-inf'): # Check if any valid score was recorded
            final_kOptimal_dict[model_name] = data['k']
        else: 
            final_kOptimal_dict[model_name] = 1 # Default to 1 if no valid score found
            print(f"Warning: No valid score found for model {model_name}. Optimal k set to 1 (default).")
    results['kOptimal'] = final_kOptimal_dict
            
    return results


def plot_k_selection(
    results: dict,
    metric: str = 'Accuracy', # Caller should ensure this is title-cased if needed
    output_file: str = 'output.png'
) -> None:
    plt.figure(figsize=(12, 7)) 
    
    ks = sorted([k_val for k_val in results if isinstance(k_val, int)])

    if not ks:
        print("Warning: No k values found in results to plot.")
        plt.title(f'{metric} vs. Number of Selected Features (No data for k values)')
        plt.xlabel('Number of Selected Features (k)')
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Plot saved to {output_file} (No data for k values).")
        plt.show()
        return

    # Dynamically get model names from the results for the first k
    if not results[ks[0]] or not isinstance(results[ks[0]], dict):
        print(f"Warning: No model results found for k={ks[0]}. Cannot determine models to plot.")
        # ... (similar empty plot for no model data)
        plt.title(f'{metric} vs. Number of Selected Features (No model data for k={ks[0]})')
        plt.xlabel('Number of Selected Features (k)')
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Plot saved to {output_file} (No model data).")
        plt.show()
        return
        
    model_names_from_results = list(results[ks[0]].keys())
    
    preferred_order = ['regModel','svmModel','nbcModel','knnModel','dtcModel','DNN'] 
    plot_model_names = [m for m in preferred_order if m in model_names_from_results]
    remaining_models = sorted([m for m in model_names_from_results if m not in plot_model_names])
    plot_model_names.extend(remaining_models)

    if not plot_model_names:
        print("Warning: No model results found to plot after filtering/ordering.")
        # ... (similar empty plot logic)
        return

    for m_name in plot_model_names:
        vals = []
        valid_ks_for_model = []
        for k_val in ks:
            if m_name in results[k_val] and \
               results[k_val][m_name] is not None and \
               isinstance(results[k_val][m_name], (int, float)) and \
               results[k_val][m_name] > float('-inf'):
                vals.append(results[k_val][m_name])
                valid_ks_for_model.append(k_val)
        
        if not vals:
            print(f"Warning: No valid data points to plot for model {m_name}.")
            continue

        line, = plt.plot(valid_ks_for_model, vals, marker='o', label=m_name)

        if 'kOptimal' in results and m_name in results['kOptimal']:
            kb = results['kOptimal'][m_name]
            if kb in results and m_name in results[kb] and \
               results[kb][m_name] is not None and \
               isinstance(results[kb][m_name], (int, float)) and \
               results[kb][m_name] > float('-inf'):
                best_k_score = results[kb][m_name]
                plt.plot(kb, best_k_score, marker='*', markersize=12, color=line.get_color(), linestyle='None',
                         label=f'_nolegend_best_marker_for_{m_name}') # No separate legend for star
                plt.axvline(x=kb, color=line.get_color(), linestyle='--', 
                            label=f'{m_name} best k={kb} ({best_k_score:.3f})')
            else:
                 print(f"Warning: Optimal k={kb} for model {m_name} has no valid score data. Skipping axvline.")
        else:
            print(f"Warning: Model {m_name} not found in results['kOptimal']. Skipping axvline for best k.")

    plt.xlabel('Number of Selected Features (k)')
    plt.ylabel(metric)
    plt.title(f'{metric} vs. Number of Selected Features')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter out internal labels like '_nolegend_'
    filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if not l.startswith('_nolegend_')]
    if filtered_handles_labels:
        filtered_handles, filtered_labels = zip(*filtered_handles_labels)
        plt.legend(filtered_handles, filtered_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        print("No lines with valid labels were plotted to create a legend.")


    plt.grid(True)
    # Adjust layout to make space for legend if it's outside
    plt.tight_layout(rect=[0, 0, 0.83, 1]) # rect=[left, bottom, right, top]
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()


if __name__ == '__main__':
    # Hardcoded parameters
    data_p = "../data/original/data1.csv"
    set_n = 1
    metric_name = 'accuracy' 
    max_k_val = 30
    tau_val = 0.8
    test_s = 0.2
    output_f = 'output.png'

    # Specific models and DNN configuration for this run
    user_model_classes = [KNN, DTC, DNN] # Uses imported KNN, DTC, DNN classes
    user_dnn_hidden_sizes = [13, 8, 5]

    print("Starting k-feature selection process with hardcoded parameters...")
    print(f"Parameters: data='{data_p}', set_num={set_n}, metric='{metric_name}', max_k={max_k_val}, tau_redundancy={tau_val}, test_size={test_s}")
    print(f"Models to run: {[m.__name__ for m in user_model_classes]}")
    if DNN in user_model_classes:
        print(f"DNN hidden sizes: {user_dnn_hidden_sizes}")

    res = find_optimal_k(
        data_path=data_p,
        set_num=set_n,
        metric=metric_name,
        k_max=max_k_val,
        tau_redundancy=tau_val,
        test_size=test_s,
        model_classes=user_model_classes,
        dnn_hidden_sizes=user_dnn_hidden_sizes
    )

    print("\nFeature selection process completed. Results overview:")
    # Summarize kOptimal nicely
    if 'kOptimal' in res:
        print(f"\nOptimal k per model (based on '{metric_name}'):")
        for model_n, best_k in res['kOptimal'].items():
            if isinstance(best_k, int) and best_k in res and \
               model_n in res[best_k] and res[best_k][model_n] > float('-inf'):
                 score_at_best_k = res[best_k][model_n]
                 print(f"  - {model_n}: k = {best_k} (score: {score_at_best_k:.4f})")
            else:
                 print(f"  - {model_n}: k = {best_k} (score N/A or not found for this k)")
    else:
        print("No 'kOptimal' information found in results.")

    # Detailed scores can be very verbose, optionally print a small part or skip
    # print("\nFull results dictionary (first few k values):")
    # for i, (k_val_res, model_scores) in enumerate(res.items()):
    #     if i < 3 or k_val_res == 'kOptimal': # Print first 2 k-value results and kOptimal
    #         if k_val_res != 'kOptimal':
    #             print(f"\nScores for k={k_val_res}:")
    #             for model_n, score in model_scores.items():
    #                 print(f"    {model_n}: {score:.4f}" if isinstance(score, (int,float)) and score > float('-inf') else f"    {model_n}: {score}")


    plot_k_selection(res, metric=metric_name.title(), output_file=output_f)