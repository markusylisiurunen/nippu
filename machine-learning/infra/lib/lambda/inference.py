import json


def handler(event, context):
    """
    The request is expected to be in the following format:

    {
        "base64_image": "<base64 encoded image data>",
        "blocks": ["<a list of word blocks>"]
    }
    """
    print('request: {}'.format(json.dumps(event)))

    # TODO: take the image and word blocks and pass them through SageMaker

    result = {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': {
            'vendor_name': [{'value': 'Keijon Kukkakauppa Oy'}],
            'vendor_address': [{'value': 'Ruusukuja 7, 00100, Helsinki'}],
            'date_of_purchase': [{'value': '2022-02-24'}],
            'total_amount': [{'value': '24.99'}],
        }
    }

    print('result: {}'.format(json.dumps(result)))
    return result
