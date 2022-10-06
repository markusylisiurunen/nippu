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
    as_text = ' '.join(list(
        map(lambda block: block['Text'], event['ocr']['blocks'])
    ))

    questions = [
        "What is the vendor's name?",
        "What is the vendor's address?",
        "What is the date of purchase?",
        "How much was the total amount?",
    ]

    response = client.invoke_endpoint(
        EndpointName='huggingface-pytorch-inference-2022-10-06-15-01-07-015',
        Body=json.dumps(
            {
                'inputs': list(
                    map(lambda q: {'context': as_text,
                        'question': q}, questions)
                )
            }
        ),
        ContentType='application/json',
    )
    response = response['Body'].read()
    response = json.loads(response)

    print('sagemaker response: {}'.format(json.dumps(response)))

    vendor_name = response[0]['answer']
    vendor_address = response[1]['answer']
    date_of_purchase = response[2]['answer']
    total_amount = response[3]['answer']

    result = {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': {
            'vendor_name': [{'value': vendor_name}],
            'vendor_address': [{'value': vendor_address}],
            'date_of_purchase': [{'value': date_of_purchase}],
            'total_amount': [{'value': total_amount}],
        }
    }

    print('result: {}'.format(json.dumps(result)))
    return result
