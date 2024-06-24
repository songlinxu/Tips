
Steps below:
- Create a web app in kubernetes and get an external IP: Follow the steps in https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app#cloud-shell_1
- Go to Squarespace (your owned domain) DNS settings and click "add record" under custom records. For "Host", enter "@". For "Type", select "A". For "Data", enter your external ID address from kubernete. Save and done! You can know visit your domain name or check it using "host cogteach.com" in google cloud shell.

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
