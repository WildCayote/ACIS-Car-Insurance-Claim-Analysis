import mlflow, shap
import mlflow.sklearn

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

def initialize_mlflow(uri: str, experiment_name: str):
    """
    Initializes MLflow by setting the tracking URI and retrieving or creating an experiment.

    This function sets the MLflow tracking URI, either creates a new experiment with the provided name,
    or retrieves the existing experiment if it already exists. The function returns the experiment ID.

    Args:
        uri (str): The tracking URI for MLflow (e.g., the location to log runs and artifacts).
        experiment_name (str): The name of the experiment to create or retrieve.

    Returns:
        str: The ID of the experiment.
    """

    # Initialize MLflow
    mlflow.set_tracking_uri(uri)  # Set up local directory for logging (In this case a directory called test in the same folder as the script)
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        search_result = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")
        experiment_id = search_result[0].experiment_id

    return experiment_id

def train_and_log_model(model, model_name,experiment_id, X_train, y_train, X_test, y_test):
    """
    Trains a given machine learning model, logs its performance metrics and parameters to an MLflow experiment,
    and saves the trained model.

    This function starts an MLflow run within the specified experiment, trains the provided model on the training data,
    evaluates it on the test data by calculating the mean squared error (MSE), and logs both the model and the metrics.

    Args:
        model (object): The machine learning model to be trained (e.g., sklearn model).
        model_name (str): The name of the model, used as the run name in MLflow and for logging.
        experiment_id (str or int): The ID of the MLflow experiment where the run should be logged.

    Returns:
        tuple: A tuple containing:
            - mse (float): Mean Squared Error of the model on the test set.
            - r2 (float): R-squared score of the model on the test set.
            - model (sklearn.base.BaseEstimator): The trained model.
    """
    with mlflow.start_run(run_name=model_name, experiment_id=experiment_id):
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Log the metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log the model itself
        mlflow.sklearn.log_model(model, model_name)
        
        # Return metrics and the trained model
        return mse, r2, model

def explain_model_with_shap(model, X_train, X_test):
    """
    Use SHAP to explain the predictions of the provided model.

    Args:
        model (sklearn.base.BaseEstimator): The trained scikit-learn model.
        X_train (pd.DataFrame or np.ndarray): The training features.
        X_test (pd.DataFrame or np.ndarray): The test features.
    """
    # Initialize SHAP explainer
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor)):
        explainer = shap.Explainer(model, X_train)
    elif isinstance(model, LinearRegression):
        explainer = shap.Explainer(model, X_train)
    else:
        raise ValueError("SHAP explainer not available for this model type.")
    
    # Compute SHAP values
    shap_values = explainer(X_test)
    
    # Plot SHAP values
    shap.summary_plot(shap_values, X_test)
    
    # Return SHAP values
    return shap_values