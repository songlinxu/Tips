
Steps below:
- Create a web app in kubernetes and get an external IP: Follow the steps in https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app#cloud-shell_1
- Go to Squarespace (your owned domain) DNS settings and click "add record" under custom records. For "Host", enter "@". For "Type", select "A". For "Data", enter your external ID address from kubernete. Save and done! You can know visit your domain name or check it using "host cogteach.com" in google cloud shell. 
