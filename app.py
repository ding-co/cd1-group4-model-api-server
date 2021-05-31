import flask
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['GET'])
def make_prediction():
    if request.method == 'GET':

        # 입력 받은 변수 값을 가지고 사고 위험 확률 예측
        prediction = (model.predict_proba([[0.09195619, 0.67504492, 0.75955954, 0.66476231, 0.56372769,
       0.17235857, 0.14059516, 0.05915105, 0.24763523]]))[0]

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        label = '{:.2}%'.format(str(prediction[1]*100))

        # 결과 리턴
        return render_template('index.html', label=label)


if __name__=="__main__":
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = joblib.load('model/model.pkl')
    # Flask 서비스 스타트
    app.run(host='127.0.0.1', port=8080, debug=True)