# Tiltfile
load("ext://helm_remote", "helm_remote")
load("ext://secret", "secret_from_dict")

docker_compose("infra/docker-compose.yml")

# Add labels to Docker services
dc_resource('minio', labels=["minio"])
dc_resource('db', labels=["db"])
dc_resource('web', labels=["mlflow"])
dc_resource('mc', labels=["mc"])

# build images
docker_build('model_api', '.', target="api")
docker_build('cancer-train', '.', target="train")


k8s_yaml(
    helm(
        "charts/",
        name="mollie-challenge",
        values=["charts/values.yaml"],
        set= [
            "image.repository=mollie-challenge",
            "image.tag=latest",
            "serviceAccount.create=true",
            "local.aws_endpoint=http://minio:9000",
            "local.aws_region=us-east-1",
            "local.aws_access_key_id=minio",
            "local.aws_secret_access_key=minio123",
        ]
    )
)

k8s_resource(
    "api",
    port_forwards="8000:8000",
)
