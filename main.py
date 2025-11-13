# <<<<<<< HEAD
# print("p")
# print("second push")
# =======
from flask import blueprints, Flask

app = Flask(__name__)

@app.route("/")
def main():
    return "bro"

if __name__ == "__main__":
    app.run(debug=True)

# >>>>>>> eca
