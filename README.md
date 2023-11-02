# Simple DL application to deploy in K8s

## Prerequisites
 * A k8s cluster (1.28+) with:
   * Nginx Ingress controller 
   * GPU nodes (nvidia.com/gpu.shared or change that in triton manifest)
 * An NFS server where you'll store `model-registry`. Or you can configure any other supported storage for that purpose.

### Build and push docker containers
```bash
docker build -f ./api/Dockerfile -t koolvn/simple-fast-api:app ./api
docker push koolvn/simple-fast-api:app 
```

### Start PostgreSQL
```bash
docker compose -f ./database/docker-compose.yml up -d
```

### Start fastapi and triton in k8s

Configure manifests at you own taste and then run

```bash
kubectl apply -f ./manifests/
```

### Use service
Go to `http://<ingress-controller-ip>/clf-app/`