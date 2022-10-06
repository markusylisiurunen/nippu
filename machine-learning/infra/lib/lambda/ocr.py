import base64
import json

import boto3

client = boto3.client('textract')


def handler(event, context):
    """
    The request is expected to be in the following format:

    {
        "base64_image": "<base64 encoded image data>"
    }
    """
    print('request: {}'.format(json.dumps(event)))

    # extract the image bytes from the event
    image_bytes = base64.b64decode(event['base64_image'].encode('ascii'))

    # run the image through AWS Textract
    response = client.detect_document_text(Document={'Bytes': image_bytes})

    # filter only the word blocks from the full list of blocks
    blocks = response['Blocks']
    word_blocks = []

    for block in blocks:
        if block['BlockType'] != 'WORD':
            continue

        word_block = {
            'Text': block['Text'],
            'Geometry': block['Geometry']
        }

        word_blocks.append(word_block)

    print('word blocks: {}'.format(json.dumps(word_blocks)))

    # return the list of word blocks as the response
    result = {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': {'blocks': word_blocks}
    }

    print('result: {}'.format(json.dumps(word_blocks)))
    return result
