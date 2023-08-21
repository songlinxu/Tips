# This is a tutorial to deploy a python-based web application via Flask on AWS Elastic Beanstalk

## Steps






## Important Notes
- Be sure to first visit all the references below before you get started.
- On MacOS, when you upload your code zipped folder, you have to first remove the ".DS_Store" file before uploading. You could use this command: zip -vr application.zip ./ -x "*.DS_Store"
- In application.py, be sure to use application = Flask(__name__) instead of app = Flask(__name__).
- Be sure to create a python.config file in the .ebextensions folder and write:
```
  option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application:application
```



## Important Reference

1. https://www.cnblogs.com/xiaofuge/p/17266094.html
2. https://www.youtube.com/watch?v=fGxY_Hji8_U
3. https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html
4. https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/GettingStarted.html

