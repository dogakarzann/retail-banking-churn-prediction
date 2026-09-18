import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from src.exception import CustomException
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             brier_score_loss, auc, roc_auc_score, classification_report, 
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold

def classification_kfold_cv(models, X_train, y_train, n_folds=5):
    '''
    Evaluate multiple machine learning models using k-fold cross-validation.

    This function evaluates a dictionary of machine learning models by training each model on the provided training data
    and evaluating their performance using k-fold cross-validation. The evaluation metric used is ROC-AUC score.

    Args:
        models (dict): A dictionary where the keys are model names and the values are instantiated machine learning model objects.
        X_train (array-like): The training feature data.
        y_train (array-like): The corresponding target labels for the training data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each model, including their average validation scores
                  and training scores.

    Raises:
        CustomException: If an error occurs while evaluating the models.

    '''


    try:
        # Stratified KFold in order to maintain the target proportion on each validation fold - dealing with imbalanced target.
        stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Dictionaries with validation and training scores of each model for plotting further.
        models_val_scores = dict()
        models_train_scores = dict()

        for model in models:
            # Get the model object from the key with his name.
            model_instance = models[model]

            # Measure training time.
            start_time = time.time()
            
            # Fit the model to the training data.
            model_instance.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time

            # Make predictions on training data and evaluate them.
            if hasattr(model_instance, "predict_proba"):
                y_train_pred = model_instance.predict_proba(X_train.values)[:, 1]
            elif hasattr(model_instance, "decision_function"):
                y_train_pred = model_instance.decision_function(X_train.values)
            else:
                y_train_pred = model_instance.predict(X_train.values)
            
            train_score = roc_auc_score(y_train, y_train_pred)

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            val_scores = cross_val_score(model_instance, X_train.values, y_train, scoring='roc_auc', cv=stratified_kfold)
            avg_val_score = val_scores.mean()
            val_score_std = val_scores.std()

            # Add the model scores to the validation and training scores dictionaries.
            models_val_scores[model] = avg_val_score
            models_train_scores[model] = train_score

            # Print the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'Training score: {train_score}')
            print(f'Average validation score: {avg_val_score}')
            print(f'Standard deviation: {val_score_std}')
            print(f'Training time: {round(training_time, 5)} seconds')
            print()

        # Convert scores to a dataframe
        val_df = pd.DataFrame(list(models_val_scores.items()), columns=['model', 'avg_val_score'])
        train_df = pd.DataFrame(list(models_train_scores.items()), columns=['model', 'train_score'])
        eval_df = val_df.merge(train_df, on='model')

        # Sort the dataframe by the best ROC-AUC score.
        eval_df  = eval_df.sort_values(['avg_val_score'], ascending=False).reset_index(drop=True)
        
        return eval_df
    
    except Exception as e:
        raise(CustomException(e, sys))
    

def plot_classification_kfold_cv(eval_df, figsize=(20, 7), bar_width=0.35, title_size=15,
                             title_pad=30, label_size=11, labelpad=20, legend_x=0.08, legend_y=1.08):
    '''
    Plot classification performance using k-fold cross-validation.

    Parameters:
        eval_df (DataFrame): DataFrame containing evaluation metrics for different models.
        figsize (tuple, optional): Figure size (width, height). Defaults to (20, 7).
        bar_width (float, optional): Width of bars in the plot. Defaults to 0.35.
        title_size (int, optional): Font size of the plot title. Defaults to 15.
        title_pad (int, optional): Padding of the plot title. Defaults to 30.
        label_size (int, optional): Font size of axis labels. Defaults to 11.
        labelpad (int, optional): Padding of axis labels. Defaults to 20.
        legend_x (float, optional): x-coordinate of legend position. Defaults to 0.08.
        legend_y (float, optional): y-coordinate of legend position. Defaults to 1.08.

    Raises:
        CustomException: Raised if an unexpected error occurs.

    Returns:
        None
    '''
    try:
        # Plot each model and their train and validation (average) scores.
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(eval_df['model']))
        y = np.arange(len(eval_df['train_score']))

        val_bars = ax.bar(x - bar_width/2, eval_df['avg_val_score'], bar_width, label='Val score', color='#023047')
        train_bars = ax.bar(x + bar_width/2, eval_df['train_score'], bar_width, label='Train score', color='#0077b6')

        ax.set_xlabel('Model', labelpad=labelpad, fontsize=label_size, loc='left')
        ax.set_ylabel('ROC-AUC', labelpad=labelpad, fontsize=label_size, loc='top')
        ax.set_title("Models' performances", fontweight='bold', fontsize=title_size, pad=title_pad, loc='left')
        ax.set_xticks(x, eval_df['model'], rotation=0, fontsize=10.8)
        ax.tick_params(axis='x', which='both', bottom=False)
        ax.tick_params(axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')

        # Define handles and labels for the legend with adjusted sizes
        handles = [plt.Rectangle((0,0), 0.1, 0.1, fc='#023047', edgecolor = 'none'),
                plt.Rectangle((0,0), 0.1, 0.1, fc='#0077b6', edgecolor = 'none')]
        labels = ['Val score', 'Train score']
            
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(legend_x, legend_y), frameon=False, ncol=2, fontsize=10)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classifier(y_true, y_pred, probas, y_test=None):
    '''
    Evaluate the performance of a binary classifier and visualize the results.

    This function calculates and displays various evaluation metrics for a binary classifier,
    including the classification report, confusion matrix, ROC curve and AUC, PR curve and AUC,
    brier score, gini and ks.

    Args:
    - y_true (pd.series): True binary labels.
    - y_pred (pd.series): Predicted binary labels.
    - probas (pd.series): Predicted probabilities of positive class.
    - y_test (pd.series, optional): Explicit y_test for KS calculation if it differs from y_true. By default uses y_true.

    Returns:
    - model_metrics (pd.DataFrame): A dataframe containing the classification metrics for the passed set.

    Raises:
    - CustomException: If an error occurs during evaluation.
    '''

    if y_test is None:
        y_test = y_true

    try:
        # Print classification report and calculate its metrics to include in the final metrics df.
        print(classification_report(y_true, y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Calculate and print brier score, gini and ks.
        brier_score = brier_score_loss(y_true, probas)
        print(f'Brier Score: {round(brier_score, 2)}')
        
        fpr, tpr, thresholds = roc_curve(y_true, probas)
        roc_auc = roc_auc_score(y_true, probas)
        gini = 2 * roc_auc - 1
        print(f'Gini: {round(gini, 2)}')
        
        scores = pd.DataFrame()
        scores['actual'] = y_test.reset_index(drop=True)
        scores['churn_probability'] = probas
        sorted_scores = scores.sort_values(by=['churn_probability'], ascending=False)
        sorted_scores['cum_negative'] = (1 - sorted_scores['actual']).cumsum() / (1 - sorted_scores['actual']).sum()
        sorted_scores['cum_positive'] = sorted_scores['actual'].cumsum() / sorted_scores['actual'].sum()
        sorted_scores['ks'] = np.abs(sorted_scores['cum_positive'] - sorted_scores['cum_negative'])
        ks = sorted_scores['ks'].max()
        
        print(f'KS: {round(ks, 2)}')
        
        # Confusion matrix.
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot = True, fmt = 'd')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Values')
        plt.ylabel('Real Values')
        plt.show()
        
        # Plot ROC Curve and ROC-AUC.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}', color='#023047')
        ax.plot([0, 1], [0, 1], linestyle='--', color='#e85d04')  # Random guessing line.
        ax.set_xlabel('False Positive Rate', fontsize=10.8, labelpad=20, loc='left')
        ax.set_ylabel('True Positive Rate', fontsize=10.8, labelpad=20, loc='top')
        ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_title('Receiver operating characteristic (ROC) curve', fontweight='bold', fontsize=12, pad=20, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.show()
    
        # PR AUC Curve and score.

        # Calculate model precision-recall curve.
        p, r, _ = precision_recall_curve(y_true, probas)
        pr_auc = auc(r, p)
        
        # Plot the model precision-recall curve.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r, p, marker='.', label=f'PR AUC = {pr_auc:.2f}', color='#023047')
        ax.set_xlabel('Recall', fontsize=10.8, labelpad=20, loc='left')
        ax.set_ylabel('Precision', fontsize=10.8, labelpad=20, loc='top')
        ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_title('Precision-recall (PR) curve', fontweight='bold', fontsize=12, pad=20, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.show()

        # Construct a DataFrame with metrics for passed sets.
        model_metrics = pd.DataFrame({
                                    'Metric': ['Accuracy',
                                               'Precision',
                                               'Recall',
                                               'F1-Score',
                                               'ROC-AUC',
                                               'KS',
                                               'Gini',
                                               'PR-AUC',
                                               'Brier'],
                                    'Value': [accuracy, 
                                              precision, 
                                              recall,
                                              f1,
                                              roc_auc,
                                              ks,
                                              gini, 
                                              pr_auc,
                                              brier_score,
                                              ],
                                    })
        
        return model_metrics

    except Exception as e:
        raise CustomException(e, sys)

    

def plot_feature_importances(model, data):
    '''
    Plot feature importances of a given machine learning model.

    This function takes a trained machine learning model and the corresponding dataset used for training, and plots the
    feature importances of the model's attributes. Feature importances are sorted in descending order for visualization.

    Args:
        model (object): The trained machine learning model with a feature_importances_ attribute.
        data (DataFrame): The dataset containing the features used for training the model.

    Returns:
        None (displays feature importances).

    Raises:
        CustomException: If an error occurs while plotting feature importances.

    '''
    
    try:
        # Get feature importances
        importances = model.feature_importances_
        feature_names = data.columns 


        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(12, 3))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), sorted_importances, tick_label=sorted_feature_names, color='#023047')
        plt.xticks(rotation=90)
        plt.show()

    except Exception as e:
        raise CustomException(e, sys)
    

def precision_vs_recall_curve(y_true, predicted_probas):
    '''
    Generate a precision-recall curve based on different threshold values.

    This function calculates precision and recall values for different threshold values,
    based on the model's estimated probability of being positive.

    Args:
        y_true (array-like): True binary labels of the data.
        predicted_probas (array-like): Predicted probabilities for the positive class.

    Returns:
        tuple: A tuple containing three elements:
            - precision (array): Precision values for different thresholds.
            - recall (array): Recall values for different thresholds.
            - threshold (array): Threshold values corresponding to precision and recall.

    Raises:
        CustomException: If an error occurs during curve generation.

    '''

    try:
        # Obtain precision and recall values (arrays) for different threshold
        # values, when compared to the model's estimated probability of being positive.
        precision, recall, threshold = precision_recall_curve(y_true, predicted_probas[:, 1])

        # Plotting precision-recall curve.
        plt.title('Precision vs Recall Curve')
        plt.plot(threshold, precision[:-1], 'b--', label='Precision')
        plt.plot(threshold, recall[:-1], 'r--', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='lower left')
        plt.ylim([0,1])

        return precision, recall, threshold

    except Exception as e:
        raise CustomException(e, sys)


def get_threshold_metrics(precision, recall, threshold, target_metric, target_metric_value):
    '''
    Get precision and recall metrics at a specified target metric value.

    This function retrieves precision and recall metrics at a specific target
    metric value for a given threshold. It is useful for selecting thresholds
    in classification problems.

    Args:
        precision (list): A list of precision values corresponding to different thresholds.
        recall (list): A list of recall values corresponding to different thresholds.
        threshold (list): A list of threshold values.
        target_metric (str): The target metric to consider ('recall' or 'precision').
        target_metric_value (float): The desired value of the target metric.

    Returns:
        tuple: A tuple containing three elements:
            - threshold_precision (float): Precision at the specified target metric value.
            - threshold_recall (float): Recall at the specified target metric value.
            - threshold_selected (float): The threshold value corresponding to the metrics.

    Raises:
        CustomException: If an error occurs while calculating the metrics.
    '''
    
    try:
        if target_metric == 'recall':
            recall_array = np.asarray(recall)
            target_value_index = np.where(recall_array[:-1] == target_metric_value)[0][0]

        else:
            precision_array = np.asarray(precision)
            target_value_index = np.where(precision_array[:-1] == target_metric_value)[0][0]
        
        threshold_precision = precision[target_value_index]
        threshold_recall = recall[target_value_index]
        threshold_selected = threshold[target_value_index]

        print(f'For a threshold of {threshold_selected}:')
        print(f'Recall: {threshold_recall}')
        print(f'Precision: {threshold_precision}')

        return threshold_precision, threshold_recall, threshold_selected

    except Exception as e:
        raise CustomException(e, sys)
    

def plot_probability_distributions(y_true, probas):
    '''
    Plots the kernel density estimate (KDE) of predicted probabilities for churners and non-churners.

    Parameters:
    - y_true (array-like): The true class labels (1 for churner, 0 for non-churner).
    - probas (array-like): Predicted probabilities for the positive class (churners).

    Raises:
    - CustomException: Raised if an unexpected error occurs during plotting.

    Example:
    ```python
    plot_probability_distributions(y_true, probas)
    ```

    Dependencies:
    - pandas
    - seaborn
    - matplotlib

    Note:
    The function creates a KDE plot illustrating the distribution of predicted probabilities for churners and non-churners.
    It provides visual insights into the model's ability to distinguish between the two classes.

    '''
    try:
        probas_df = pd.DataFrame({'churn_probability': probas,
                                'churn': y_true})

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(data=probas_df, x='churn_probability', hue='churn', fill=True, ax=ax, palette=['#023047', '#e85d04'])
        ax.set_title('Predicted probabilities distribution - churners and non-churners', fontweight='bold', fontsize=12, pad=45, loc='left')
        ax.set_xlabel('Predicted probabilities', fontsize=10.8, labelpad=20, loc='left')
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                    )
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        
        handles = [plt.Rectangle((0,0), 0.1, 0.1, fc='#e85d04', edgecolor = 'none'),
                plt.Rectangle((0,0), 0.1, 0.1, fc='#023047', edgecolor = 'none')]
        labels = ['Churn', 'Not churn']
            
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.14, 1.15), frameon=False, ncol=2, fontsize=10)

        
    except Exception as e:
        raise CustomException(e, sys)


def probability_scores_ordering(y_true, probas):
    '''
    Order and visualize the probability scores in deciles based on predicted probabilities and true labels.

    Parameters:
    - y_true (pd.Series): Actual target values for the set. 1 is churner and 0 is non-churner.
    - probas (pd.Series): Predicted probabilities of churning for the passed set.

    Returns:
    - None: Plots the probability scores ordering.

    Raises:
    - CustomException: An exception is raised if an error occurs during the execution.
    
    Example:
    ```python
    probability_scores_ordering(y_test, probas)
    ```
    '''
    try:
        # Add some noise to the predicted probabilities and round them to avoid duplicate problems in bin limits.
        noise = np.random.uniform(0, 0.0001, size=probas.shape)
        probas += noise
        #probas = round(probas, 10)
        
        # Create a DataFrame with the predicted probabilities of churning and actual values.
        probas_actual_df = pd.DataFrame({'probabilities': probas, 'actual': y_true.reset_index(drop=True)})
        
        # Sort the probas_actual_df by probabilities.
        probas_actual_df = probas_actual_df.sort_values(by='probabilities', ascending=True)
        
        # Calculate the deciles.
        probas_actual_df['deciles'] = pd.qcut(probas_actual_df['probabilities'], q=10, labels=False, duplicates='drop')
        
        # Calculate the churn rate per decile.
        decile_df = probas_actual_df.groupby(['deciles'])['actual'].mean().reset_index().rename(columns={'actual': 'churn_rate'})
        
        # Plot probability scores ordering.
        # Plot bar graph of deciles vs event rate.
        fig, ax = plt.subplots(figsize=(12, 3))
        
        bars = ax.bar(decile_df['deciles'], decile_df['churn_rate'], color='#023047')
        
        ax.set_title('Probability scores ordering - Churn rate per decile', loc='left', fontweight='bold', fontsize=14)
        ax.set_xticks(range(10), ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_xlabel('Decil', labelpad=25, loc='center')
        ax.yaxis.set_visible(False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(False)
        
        # Annotate churn rate inside each bar with increased font size
        for bar, absent_rate in zip(bars, decile_df['churn_rate']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.08, f'{absent_rate*100:.1f}%', ha='center', va='top', color='white', fontsize=10.4)
            
    except Exception as e:
        raise CustomException(e, sys)
