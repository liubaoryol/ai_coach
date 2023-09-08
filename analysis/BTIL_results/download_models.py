import os
import zipfile
import requests


def download_file_from_google_drive(id, destination):
  URL = "https://docs.google.com/uc?export=download"

  session = requests.Session()

  response = session.get(URL, params={'id': id}, stream=True)
  token = get_confirm_token(response)

  if token:
    params = {'id': id, 'confirm': token}
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, destination)


def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value

  return None


def save_response_content(response, destination):
  CHUNK_SIZE = 32768

  with open(destination, "wb") as f:
    for chunk in response.iter_content(CHUNK_SIZE):
      if chunk:  # filter out keep-alive new chunks
        f.write(chunk)


if __name__ == "__main__":

  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  learned_model_dir = data_dir + "learned_models/"

  if not os.path.exists(learned_model_dir):
    os.makedirs(learned_model_dir)
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  destination = data_dir + "downloaded_models.zip"

  file_id = '1bu6iLL4dgV_NEGm7_DwlWwUabYB4-weI'
  print("Downloading models from googe drive ...")
  download_file_from_google_drive(file_id, destination)

  print("Extracting file to %s ..." % learned_model_dir)
  with zipfile.ZipFile(destination, 'r') as zip_file:
    zip_file.extractall(learned_model_dir)
  print("Done downloading models")
