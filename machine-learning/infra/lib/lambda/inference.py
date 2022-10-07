import json

import boto3

client = boto3.client('sagemaker-runtime')


def handler(event, context):
    """
    The request is expected to be in the following format:

    {
        "image": { "base64_image": "<base64 encoded image data>" },
        "ocr": { "blocks": ["<a list of word blocks>"] }
    }
    """
    print('request: {}'.format(json.dumps(event)))

    # TODO: take the image and word blocks and properly pass them through SageMaker
    response = client.invoke_endpoint(
        EndpointName='lolbalmodel',
        Body=json.dumps(
            {
                'inputs': {
                    'base64_image': event['image']['base64_image'],
                    'blocks': event['ocr']['blocks'],
                }
            }
        ),
        ContentType='application/json',
    )
    response = response['Body'].read()
    response = json.loads(response)

    print('sagemaker response: {}'.format(json.dumps(response)))

    result = {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': response
    }

    print('result: {}'.format(json.dumps(result)))
    return result
