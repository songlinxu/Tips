
Steps below:
- Create a web app in kubernetes and get an external IP: Follow the steps in https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app#cloud-shell_1
- Go to Squarespace (your owned domain) DNS settings and click "add record" under custom records. For "Host", enter "@". For "Type", select "A". For "Data", enter your external ID address from kubernete. Save and done! You can know visit your domain name or check it using "host cogteach.com" in google cloud shell.

## Important notes to check before EACH running commands in google cloud shell: 
First, Check if you have **Enabled APIs** from the Compute Engine, Artifact Registry, and Google Kubernetes Engine.

Second, **MUST Check** steps below before updating any new codes or deployments.
```
export PROJECT_ID=PROJECT_ID
gcloud config set project $PROJECT_ID
gcloud config set compute/region REGION
gcloud container clusters get-credentials hello-cluster --region REGION
gcloud auth configure-docker REGION-docker.pkg.dev

gcloud artifacts repositories add-iam-policy-binding hello-repo \
    --location=REGION \
    --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
    --role="roles/artifactregistry.reader"

```
```
Be SURE to specify the region when you delete the repo
gcloud artifacts repositories delete hello-repo \
   --location=REGION

```
```
curl -k https://cogteach.com
```
```
kubectl describe managedcertificate helloweb-managed-cert
```

**Troubleshoot**
- If you use ingress to add SSL certificate for https but gets the error "response 404 (backend NotFound), service rules for the path non-existent", maybe the reason is that you did not record your global IP address in the custom record section in your domain provider (squarespace)(Host: @, Type: A, Data: your global IP number).
- After you add your custom record, if you find your website codes are not updated, maybe the reason is that...


```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: helloweb
  annotations:
    kubernetes.io/ingress.global-static-ip-name: helloweb-ip
    ingress.kubernetes.io/ssl-cert: "your tls certificate string"
    networking.gke.io/managed-certificates: helloweb-managed-cert
    ingress.gcp.kubernetes.io/redirect-http-to-https: "true"
  labels:
    app: hello
spec:
  rules:
  - host: cogteach.com  # Replace with your actual domain
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: helloweb-backend
            port:
              number: 8080
  # tls:
  # - hosts:
  #   - cogteach.com  # Replace with your actual domain
  #   secretName: tls-secret  # Placeholder, GKE will manage this for you
---
apiVersion: v1
kind: Service
metadata:
  name: helloweb-backend
  labels:
    app: hello
spec:
  type: NodePort
  selector:
    app: hello
    tier: web
  ports:
  - port: 8080
    targetPort: 8080
```
