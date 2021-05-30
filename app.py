import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib

app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # 결과 리턴
        return render_template('index.html', label=label)


if __name__=="__main__":
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = joblib.load('./model/model.pkl')
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8080, debug=True)