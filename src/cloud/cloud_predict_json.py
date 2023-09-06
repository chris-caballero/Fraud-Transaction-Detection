import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import json
import logging 

logging.basicConfig(level=logging.INFO)

# define model, version and project info
version = "v1"
project = "magnetic-flare-397719"
model = "xgboost_fraud_detection"
region = "us-central1"

def predict_json(request):
    """Send json data to a deployed model for prediction.
    Args:
        request: A json with instances to predict (batch input)
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    request_json = request.get_json()
    try:
      instances = request_json['instances']

      logging.info('Instances received')
        
      prefix = "{}-ml".format(region) if region else "ml"
      api_endpoint = "https://{}.googleapis.com".format(prefix)
      client_options = ClientOptions(api_endpoint=api_endpoint)

      logging.info('API Endpoint: %s', api_endpoint)

      service = googleapiclient.discovery.build(
          'ml', 'v1', client_options=client_options)
      name = 'projects/{}/models/{}'.format(project, model)
      if version is not None:
          name += '/versions/{}'.format(version)

      logging.info('Sending prediction request to AI Platform...')
      response = service.projects().predict(
          name=name,
          body={'instances': instances}
      ).execute()

      logging.info('Prediction response completed!')

      print('response', response)
      if 'error' in response:
          raise RuntimeError(response['error'])

      return json.dumps(response['predictions'])

    except Exception as e:
        logging.error('An error occurred: %s', str(e))
        return 'An error occurred: ' + str(e)