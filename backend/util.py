import json

import requests


def upload_file(upload_url, file_path):
    files = {'file': open(file_path, 'rb')}
    response = requests.post(upload_url, files=files)
    ret = response.content.decode('utf-8')
    ret_json = json.loads(ret)
    print ret_json
    return ret_json['data']


def post_json(post_url, post_data):
    headers = {'content-type': 'application/json'}
    response = requests.post(post_url, data=json.dumps(post_data), headers=headers)
    return response.content.decode('utf-8')


def post_form(post_url, post_data):
    headers = {'content-type': 'x-www-form-urlencoded'}
    response = requests.post(post_url, params=post_data, headers=headers)
    return response.content.decode('utf-8')

