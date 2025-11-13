from flask import Flask
from routes.api_mobile import api_bp_mobile
from routes.api_web import api_bp_web

app = Flask(__name__)

app.register_blueprint(api_bp_mobile)
app.register_blueprint(api_bp_web)

if __name__ == '__main__':
    app.run(debug=True)