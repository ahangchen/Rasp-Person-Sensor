import requests
from sensor import upload_wifi_info


def img_upload():
    url = 'http://localhost:8081/file/img'
    files = {'file': open('../data/rasp-wifi.png', 'rb')}
    response = requests.post(url, files=files)
    print(response.content.decode('utf-8'))


def wifi_upload():
    upload_wifi_info('aa:bb:cc:dd:ee:ff', 10, 1)

if __name__ == '__main__':
    wifi_upload()
