# Setup

To develop the Lambda functions, do the following:

```sh
cd lib/lambda
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If you want to install a new Python dependency, do the following:

```sh
cd lib/lambda
source .venv/bin/activate
python3 -m pip install "<name of the dependency>"
python3 -m pip freeze > requirements.txt
```

Before deploying the CDK stacks, do the following:

```sh
cd lib/lambda
source .venv/bin/activate
python3 -m pip install -r requirements.txt -t ./package
```

To deploy, do the following:

```sh
export AWS_PROFILE=NippuDevelopment
cdk diff
cdk deploy
```

To test the deployed pipeline, do the following:

```sh
mkdir .data
# add a test image of a receipt to the `.data` folder
curl -s -X POST 'https://wmddtweiy2.execute-api.eu-central-1.amazonaws.com/prod' \
  -H 'Content-Type: application/json' \
  -d '{"base64_image":"'$(base64 -i .data/example.jpeg)'"}'
```
