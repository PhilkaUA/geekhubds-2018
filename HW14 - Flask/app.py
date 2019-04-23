import json
from flask import Flask, Response, make_response, request, jsonify

app = Flask(__name__)


# First task (simple HOST)
@app.route('/')
def index():
    return "<h1>OK!</h1>"


@app.route('/user/<name>')
def user(name):
    return '<h1>OK! for ->> {0}!</h1>'.format(name)


# Second task ( Response 200)

JSON_MIME_TYPE = 'application/json'

#
# def json_response(data='', status=200, headers=None):
#     headers = headers or {}
#     if 'Content-Type' not in headers:
#         headers['Content-Type'] = JSON_MIME_TYPE
#     return make_response(data, status, headers)

geekhubds = [{'id': 0, 'title': 'GeekHub', 'author_id': 1}]

@app.route('/geekhub')
def hubs_list():
    response = Response(
        json.dumps(geekhubds), status=200, mimetype='application/json')
    return response

@app.route('/hubs')
def hub_list():
    content = json.dumps(geekhubds)
    response = make_response(
        content, 200, {'Content-Type': 'application/json'})
    return response


# Third task (Method GET)
# testing by POSTMAN
hubs_methods = [{'name': 'Python'}, {'name': 'Math'}, {'name': 'Science'}, {'name': 'ML'}]


@app.route('/get_hubs', methods=['GET'])
def all_hub():
    return jsonify(hubs_methods)


@app.route('/get_hubs/<name>', methods=['GET'])
def one_hub(name):
    hubs = [hub for hub in hubs_methods if hub['name'] == name]
    return jsonify({'hubs': hubs[0]})


# Fouth task (Method POST)
# testing by POSTMAN
@app.route('/get_hubs', methods=['POST'])
def add_hub():
    hubs_methods.append({'name': request.get_json('name')})
    return jsonify(hubs_methods)


if __name__ == '__main__':
    app.run(debug=True)