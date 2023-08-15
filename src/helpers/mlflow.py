from typing import Tuple, Any, Dict

import mlflow
import pandas as pd
from mlflow.tracking.client import MlflowClient


class MlflowModel:
    """Class to manage an MLflow model.

    This class provides methods to interact with an MLflow model,
    including fetching its version, run ID, S3 path, parameters, and metrics.

    Args:
        model_name (str): The name of the model.
        stage (str, optional): The stage of the model (e.g., "Production").
            Defaults to "Production".
    """

    def __init__(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> None:
        self.client = MlflowClient()
        self.model_name = model_name
        self.stage = stage
        self.setup_variables()
        self.setup_objects()

    def setup_variables(self) -> None:
        """Sets up the production path and version variables for the model."""
        self.production_path = f"models:/{self.model_name}/{self.stage}"
        self.__version__, self.run_id = self.__get_model_version_and_run_id()

    def setup_objects(self) -> None:
        """Sets up the model object and its related information."""
        self.model = self.__get_model()
        self.infos = self.__get_model_info(self.run_id)

    def __get_model_version_and_run_id(self) -> Tuple[int, str]:
        """Fetches the model version and run ID based on the stage.

        Returns:
            Tuple[int, str]: The version and run ID of the model.

        Raises:
            ValueError: If no model is found in the specified stage.
        """
        for model in self.client.search_model_versions(f"name='{self.model_name}'"):
            if model.current_stage == self.stage:
                return model.version, model.run_id

        raise ValueError(f"Not find any model in {self.stage} stage")

    def __get_model(self) -> Any:
        """Loads the MLflow model from the production path.

        Returns:
            Any: The loaded model.
        """
        return mlflow.pyfunc.load_model(self.production_path)

    def predict(self, X: pd.DataFrame) -> Any:
        """Predicts the output for the given input data using the model.

        Args:
            X (pd.DataFrame): input data.

        Returns:
            Any: The prediction result.
        """
        return self.model.predict(X)[0]

    def __get_model_info(self, run_id) -> None:
        """Fetches the model information using the given run ID.

        Args:
            run_id (str): The run ID of the model.
        """
        self.model_info = mlflow.get_run(run_id=run_id).to_dictionary()

    @property
    def parameters(self) -> Dict[str, Any]:
        """Fetches the model parameters.

        Returns:
            dict: The parameters of the model.
        """
        return self.model_info["data"]["params"]

    @property
    def metrics(self) -> Dict[str, Any]:
        """Fetches the model metrics.

        Returns:
            dict: The metrics of the model.
        """
        return self.model_info["data"]["metrics"]

    @property
    def s3_path(self) -> str:
        """Fetches the S3 path where the model is stored.

        Returns:
            str: The S3 path of the model.
        """
        return self.model_info["info"]["artifact_uri"]

    @property
    def version(self) -> int:
        """Fetches the version of the model in the specified stage.

        Returns:
            int: The version of the model.
        """
        return self.__version__


if __name__ == "__main__":
    model_name = "BreastCancerModel"
    m = MlflowModel(model_name)
