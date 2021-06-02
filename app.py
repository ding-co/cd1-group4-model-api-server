import flask
from flask import Flask, request, render_template
from flask_json import FlaskJSON, json_response
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
FlaskJSON(app)

wind_spd = None
atm_pres = None
humid = None
temp = None
water_temp = None
max_wave_h = None
sig_wave_h = None
avg_wave_h = None
wave_cycle = None

# Load ROSE processing data

doc = pd.read_csv('ml/rose_data.csv', encoding='cp949')

# New dataframe copy exclude unnecessary column

data = doc[['풍속', '기압', '습도', '기온', '수온', '최대파고.m.', '유의파고.m.', '평균파고.m.', '파주기.sec.', 'check']].copy()

# Variable set

x = np.array(data[['풍속', '기압', '습도', '기온', '수온', '최대파고.m.', '유의파고.m.', '평균파고.m.', '파주기.sec.']])
y = np.array(data[['check']])

# Scaling - MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)

@app.route("/")
def index():
    return flask.render_template('index.html')

# 실시간 지도
@app.route('/map', methods=['GET'])
def mapGET():
    if request.method == 'GET':

        wind_spd = request.args.get('wind_spd', 'value')
        atm_pres = request.args.get('atm_pres', 'value')
        humid = request.args.get('humid', 'value')
        temp = request.args.get('temp', 'value')
        water_temp = request.args.get('water_temp', 'value')
        max_wave_h = request.args.get('max_wave_h', 'value')
        sig_wave_h = request.args.get('sig_wave_h', 'value')
        avg_wave_h = request.args.get('kavg_wave_hey', 'value')
        wave_cycle = request.args.get('wave_cycle', 'value')
    
        # 입력 받은 변수 값을 가지고 사고 위험 확률 예측
        prediction = (model.predict_proba(scaler.transform([[wind_spd, atm_pres, humid, temp, water_temp, max_wave_h, sig_wave_h, avg_wave_h, wave_cycle]])))[0]

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        prediction_ratio = '{:.2}%'.format(str(prediction[1]*100))

        # 결과 리턴
        return json_response(prediction_ratio=prediction_ratio)

# 실시간 위치_method GET
@app.route('/location', methods=['GET'])
def locationGET():
    if request.method == 'GET':

        wind_spd = request.args.get('wind_spd', 'value')
        atm_pres = request.args.get('atm_pres', 'value')
        humid = request.args.get('humid', 'value')
        temp = request.args.get('temp', 'value')
        water_temp = request.args.get('water_temp', 'value')
        max_wave_h = request.args.get('max_wave_h', 'value')
        sig_wave_h = request.args.get('sig_wave_h', 'value')
        avg_wave_h = request.args.get('avg_wave_h', 'value')
        wave_cycle = request.args.get('wave_cycle', 'value')

        # 입력 받은 변수 값을 가지고 사고 위험 확률 예측
        prediction = (model.predict_proba(scaler.transform([[wind_spd, atm_pres, humid, temp, water_temp, max_wave_h, sig_wave_h, avg_wave_h, wave_cycle]])))[0]

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        prediction_ratio = '{:.2}%'.format(str(prediction[1]*100))

        # 결과 리턴
        return json_response(prediction_ratio=prediction_ratio)

# 실시간 위치_method POST
@app.route('/location', methods=['POST'])
def locationPOST():
    if request.method == 'POST':

        wind_spd = request.args.get('wind_spd', 'value')
        atm_pres = request.args.get('atm_pres', 'value')
        humid = request.args.get('humid', 'value')
        temp = request.args.get('temp', 'value')
        water_temp = request.args.get('water_temp', 'value')
        max_wave_h = request.args.get('max_wave_h', 'value')
        sig_wave_h = request.args.get('sig_wave_h', 'value')
        avg_wave_h = request.args.get('avg_wave_h', 'value')
        wave_cycle = request.args.get('wave_cycle', 'value')

        # 입력 받은 변수 값을 가지고 사고 위험 확률 예측
        prediction = (model.predict_proba(scaler.transform([[wind_spd, atm_pres, humid, temp, water_temp, max_wave_h, sig_wave_h, avg_wave_h, wave_cycle]])))[0]

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