import pandas as pd

import mlflow
from mlflow.tracking.client import MlflowClient


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
        for model in self.client.search_model_versions(f"name='{self.model_name}'"):
            if model.current_stage == self.stage:
                return model.version, model.run_id

        raise ValueError(f"Not find any model in {self.stage} stage")

    def __get_model(self):
        return mlflow.pyfunc.load_model(self.production_path)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)[0]

    def __get_model_info(self, run_id) -> None:
        self.model_info = mlflow.get_run(run_id=run_id).to_dictionary()

    @property
    def parameters(self) -> dict:
        return self.model_info["data"]["params"]

    @property
    def metrics(self) -> dict:
        return self.model_info["data"]["metrics"]

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


if __name__ == "__main__":
    model_name = "BreastCancerModel"
    m = MlflowModel(model_name)
