import json


def handler(event, context):
    """
    This is not needed right now, but will be in the future.
    """
    print('request: {}'.format(json.dumps(event)))

    # TODO: format the result from inference to the proper format

    result = {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': event
    }

    print('result: {}'.format(json.dumps(result)))
    return result
