#!/bin/sh
set -o errexit

export KIND_CLUSTER_NAME="k8s"

# desired cluster name; default is "kind"
KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-kind}"

kind_version=$(kind version)
kind_network='kind'
reg_name='kind-registry'
reg_port='5000'

# create registry container unless it already existsex
running="$(docker inspect -f '{{.State.Running}}' "${reg_name}" 2>/dev/null || true)"
if [ "${running}" == 'true' ]; then
  cid="$(docker inspect -f '{{.ID}}' "${reg_name}")"
  echo "> Stopping and deleting Kind Registry container..."
  docker stop $cid >/dev/null
  docker rm $cid >/dev/null
fi

echo "> Deleting Kind cluster..."
kind delete cluster --name=$KIND_CLUSTER_NAME
