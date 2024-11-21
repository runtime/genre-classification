
# Unsupervised Learning: Genre Categorization & Semantic Tagging

#### This project is an exploration of both Data Science and MLOps. 
The app reads the descriptions of a mixed dataset of books, movies, songs and TV shows and uses clustering to find commonalities. this could be used for prediction or genre labeling.

Using AWS CDK --python to create the AWS resources and houses the python Scripts that will run off the Bastion Host.

#### about the app
the app reads data from a source RDS of mixed media with and focuses on the description and genre to find similarities with other media.

#### the aws stack

```
vpc
security groups
rds x2
ec2 private
ssm - to access the bastion host
iam - groups/roles/users needed to deploy and connect
```



#### the application structure

at this time there are two scripts in the /scripts directory
```
Genre-classification
|--scripts/
	|-- semantic_clustering.py
	|-- data_analysis.py
|--lambdas
	| – data_ingestion/
		|-- app.py
|-- requirements.txt
|--app.py
|--cdk.json
|--requirements.txt
|--README.md



```

The `cdk.json` file tells the CDK Toolkit how to execute your app.

This project is set up like a standard Python project.  The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory.  To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually.

To manually create a virtualenv on MacOS and Linux:

```
$ python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```

To add additional dependencies, for example other CDK libraries, just add
them to your `setup.py` file and rerun the `pip install -r requirements.txt`
command.

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

Enjoy!


#### After cdk Deploy
setting up the ec2, rds, postgres, ssm and env vars is not covered in these instructions.
You will need a postgresql db on the first rds and a postgresql db with vector extension on the second.
suggest you create a folder on your bastion_host to run the scripts, install .venv and pip install requirements.

#### instructions
ssm into your ec2
pip install -r requirements.txt
use psql to access the media_db
use psychopg2 as an orm i/o to postgres

#### notes
The lambda function to create dummy data in the db1 isn't working due to psycocp2 installation issues. 
Openai api version 0.38 is depreciated.

#### requirements
openai secret key

