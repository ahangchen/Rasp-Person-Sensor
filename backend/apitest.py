import requests


def img_upload():
    url = 'http://localhost:8081/file/img'
    files = {'file': open('../data/rasp-wifi.png', 'rb')}
    response = requests.post(url, files=files)
    print(response.content.decode('utf-8'))




if __name__ == '__main__':
    img_upload()