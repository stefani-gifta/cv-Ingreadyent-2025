from flask import Flask

app = Flask(__name__)

# @app.route("/")
# def hello_world():
#   return "<h1>Hello, World!</h1>"

from flask import render_template

@app.route('/')
def index():
  return render_template('index.html')