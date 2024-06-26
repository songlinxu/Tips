
Steps below:
- Create a web app in kubernetes and get an external IP: Follow the steps in https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app#cloud-shell_1
- Go to Squarespace (your owned domain) DNS settings and click "add record" under custom records. For "Host", enter "@". For "Type", select "A". For "Data", enter your external ID address from kubernete. Save and done! You can know visit your domain name or check it using "host cogteach.com" in google cloud shell.
- Note that you can either use Kubernete service or ingress to expose the app to the Internet. But we recommend you to use ingress because the service does not support https.
- https://cloud.google.com/kubernetes-engine/docs/tutorials/configuring-domain-name-static-ip
- You can either use kubectl apply -f deploy.yaml or kubectl create deployment hello-app --image=REGION-docker.pkg.dev/${PROJECT_ID}/hello-repo/hello-app:v1. But we recommend you to use the deploy.yaml so that you can directly run it for each update. But be sure to change the version name 'v1' otherwise, your website may not update. The version name is also global even across different deployment name.
- But be sure to build and push the image before deployment.
- You need to use manage-cert.yaml (codes below) to create the SSL certificate before using https. Note that it takes several hours before your certificate is active. Before that, your https://domain.com may not work. Do not forget to add your global IP into custom records in your domain provider. Also, if curl https://IP does not work, you can try curl domain.com.

```
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: helloweb-managed-cert
spec:
  domains:
    - cogteach.com

```
  
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
When there are errors, try to delete the repo. But **Be SURE** to specify the region when you delete the repo
```
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
- If you find that your **deployment does not update** in the website, it is very likely that your version name is the same as the previous version. Kubernete has one limitation that, if your version name is the same, then it will not deploy your new image because it thinks that you have not changed your image. For example, if your old version image is "us-east1-docker.pkg.dev/helloapp-427317/hello-repo/hello-app:v1", then even if you have changed your deployment name, you still need to change your version name from 'v1' to 'v2'. Otherwise, the kubernete still keeps the original version of 'v1' even if it is an very early image version from a different deployment name, but the version name is the same.
- When you define both ingress and backend service, be sure to add "---" between them. Otherwise, the ingress will not update.

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
