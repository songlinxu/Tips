
Steps below:
- Create a web app in kubernetes and get an external IP: Follow the steps in https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app#cloud-shell_1
- Point your kubernete external IP to your owned domain address: Directly go to the "Configure your domain name records" section in https://cloud.google.com/kubernetes-engine/docs/tutorials/configuring-domain-name-static-ip. Note that you do not need to repeat the previous steps in this doc. In the first step, we have used the kubernete service to generate an external IP.
