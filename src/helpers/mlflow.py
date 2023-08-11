import pandas as pd

import mlflow
from mlflow.tracking.client import MlflowClient


model_name = "BreastCancerModel"
stage = "Production"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")


class MlflowModel:
    def __init__(
        self,
        model_name: str,
        stage: str = "Production",
    ):
        self.client = MlflowClient()
        self.model_name = model_name
        self.stage = stage
        self.setup_variables()
        self.setup_objects()

    def setup_variables(self):
        """Get model path in production and also model version."""
        self.production_path = f"models:/{self.model_name}/{self.stage}"
        self.__version__, self.run_id = self.__get_model_version_and_run_id()

    def setup_objects(self):
        """Setup model and input columns name"""
        self.model = self.__get_model()
        # self.input_columns = self.__get_input_columns()
        self.infos = self.__get_model_info(self.run_id)

    def __get_model_version_and_run_id(self) -> int:
        """Get model production version and return it

        Returns:
            int: model version
        """
        for model in self.client.search_model_versions(f"name='{self.model_name}'"):
            if model.current_stage == self.stage:
                return model.version, model.run_id

        raise ValueError(f"Not find any model in {self.stage} stage")

    def __get_model(self):
        """Get model in production stage on mlflow.

        Returns:
            model: The return's type depends on mlflow flavor was passed
        """

        return mlflow.pyfunc.load_model(self.production_path)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    # def __get_input_columns(self) -> list:
    #     """Get input columns saved on mlflow.

    #     Returns:
    #         list: columns names used to train model and also used on prediction
    #     """
    #     model_infos = mlflow.pyfunc.load_model(self.production_path)
    #     return model_infos.metadata.get_input_schema().column_names()

    def __get_model_info(self, run_id) -> None:
        """Get run informations based on run id.

        Args:
            run_id (str): run id hash
        """
        self.model_info = mlflow.get_run(run_id=run_id).to_dictionary()

    @property
    def parameters(self) -> dict:
        """Get model parameters

        Returns:
            dict: parameters
        """
        return self.model_info["data"]["params"]

    @property
    def metrics(self) -> dict:
        """Get model metrics

        Returns:
            dict: metrics
        """
        return self.model_info["data"]["metrics"]

    @property
    def tags(self) -> dict:
        """Get model tags

        Returns:
            dict: tags
        """
        return self.model_info["data"]["tags"]

    @property
    def experiment_id(self) -> str:
        """Get Experiment id

        Returns:
            str: Experiment id
        """
        return self.model_info["info"]["experiment_id"]

    @property
    def s3_path(self) -> str:
        """Get s3 model path

        Returns:
            str: S3 path
        """
        return self.model_info["info"]["artifact_uri"]

    @property
    def version(self) -> int:
        """Get version model staged

        Returns:
            int: version model on production stage
        """
        return self.__version__


m = MlflowModel(model_name)
