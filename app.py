import flask
from flask import Flask, request, render_template
from flask_json import FlaskJSON, json_response
import joblib
from bs4 import BeautifulSoup
from flask_cors import CORS
import requests

from ml.model import scaler

app = Flask(__name__)
FlaskJSON(app)
CORS(app) # 모든 주소에 대해 허용

wind_spd = None
atm_pres = None
humid = None
temp = None
water_temp = None
max_wave_h = None
sig_wave_h = None
avg_wave_h = None
wave_cycle = None

@app.route("/")
def index():
    return flask.render_template('index.html')

# 실시간 지도
@app.route('/map', methods=['GET'])
def mapGET():
    if request.method == 'GET':

        wind_spd = request.args.get('wind_spd', 'value')
        ats_pres = request.args.get('atm_pres', 'value')
        humid = request.args.get('humid', 'value')
        temp = request.args.get('temp', 'value')
        water_temp = request.args.get('water_temp', 'value')
        max_wave_h = request.args.get('max_wave_h', 'value')
        sig_wave_h = request.args.get('sig_wave_h', 'value')
        avg_wave_h = request.args.get('avg_wave_h', 'value')
        wave_cycle = request.args.get('wave_cycle', 'value')

        # 입력 받은 변수 값을 가지고 사고 위험 확률 예측
        prediction = (model.predict_proba(scaler.transform([[wind_spd, ats_pres, humid, temp, water_temp, max_wave_h, sig_wave_h, avg_wave_h, wave_cycle]])))[0]

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        prediction_ratio = '{:.2}%'.format(str(prediction[1]*100))

        # 결과 리턴
        return json_response(prediction_ratio=prediction_ratio)

# 실시간 위치_method GET
@app.route('/location', methods=['GET'])
def locationGET():
    if request.method == 'GET':
        headers= {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"}   
        url = "https://www.weather.go.kr/weather/observation/marine_buoy.jsp"
        
        data = requests.get(url, headers=headers)
        soup = BeautifulSoup(data.text, 'html.parser')

        idx = int(request.args.get('idx', 'value'))
        
        if idx ==1:
            root =1
        else:
            root =0

        humid = soup.select_one('#content_weather > table.table_develop > tbody > tr:nth-child({0}) > td:nth-child({1})'.format(idx, root+6))
        max_wave_h = soup.select_one('#content_weather > table.table_develop > tbody > tr:nth-child({0}) > td:nth-child({1})'.format(idx, root+9))
        sig_wave_h = soup.select_one('#content_weather > table.table_develop > tbody > tr:nth-child({0}) > td:nth-child({1})'.format(idx, root+10))
        avg_wave_h = soup.select_one('#content_weather > table.table_develop > tbody > tr:nth-child({0}) > td:nth-child({1})'.format(idx, root+11))
        wave_cycle = soup.select_one('#content_weather > table.table_develop > tbody > tr:nth-child({0}) > td:nth-child({1})'.format(idx, root+12))

        humid = float(humid.get_text())
        max_wave_h = float(max_wave_h.get_text())
        sig_wave_h = float(sig_wave_h.get_text())
        avg_wave_h = float(avg_wave_h.get_text())
        wave_cycle = float(wave_cycle.get_text())

        wind_spd = request.args.get('wind_spd', 0)
        ats_pres = request.args.get('ats_pres', 0)
        temp = request.args.get('temp', 0)
        water_temp = request.args.get('water_temp', 0)
        # 입력 받은 변수 값을 가지고 사고 위험 확률 예측
        prediction = (model.predict_proba(scaler.transform([[wind_spd, ats_pres, humid, temp, water_temp, max_wave_h, sig_wave_h, avg_wave_h, wave_cycle]])))[0]
        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        prediction_ratio = '{:.2}%'.format(str(prediction[1]*100))
        # 결과 리턴
        return json_response(prediction_ratio=prediction_ratio, humid=humid, max_wave_h=max_wave_h, sig_wave_h=sig_wave_h, avg_wave_h=avg_wave_h, wave_cycle=wave_cycle)

# 실시간 위치_method POST
@app.route('/location', methods=['POST'])
def locationPOST():
    if request.method == 'POST':

        wind_spd = request.args.get('wind_spd', 'value')
        ats_pres = request.args.get('atm_pres', 'value')
        humid = request.args.get('humid', 'value')
        temp = request.args.get('temp', 'value')
        water_temp = request.args.get('water_temp', 'value')
        max_wave_h = request.args.get('max_wave_h', 'value')
        sig_wave_h = request.args.get('sig_wave_h', 'value')
        avg_wave_h = request.args.get('avg_wave_h', 'value')
        wave_cycle = request.args.get('wave_cycle', 'value')

        # 입력 받은 변수 값을 가지고 사고 위험 확률 예측
        prediction = (model.predict_proba(scaler.transform([[wind_spd, ats_pres, humid, temp, water_temp, max_wave_h, sig_wave_h, avg_wave_h, wave_cycle]])))[0]

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        prediction_ratio = '{:.2}%'.format(str(prediction[1]*100))

        # 결과 리턴
        return json_response(prediction_ratio=prediction_ratio)

if __name__=="__main__":
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = joblib.load('model/model.pkl')
    # Flask 서비스 스타트
    app.run(host='0.0.0.0',port=8080)