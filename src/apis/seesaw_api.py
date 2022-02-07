import os

from flask import Flask, jsonify
from flask_restful import Api
from flask_cors import CORS
import sys
import subprocess

app = Flask(__name__)
api = Api(app)

from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

SEESAW_SRC_PATH = "../seesaw.py"
SEESAW_PROGRAM_PATH = "web_interface_programs/ex1"


def create_command(args):
    cmd = ["python3", SEESAW_SRC_PATH, "--file", SEESAW_PROGRAM_PATH]

    if args['abstraction'] == 'True' or args['abstraction'] == 'true':
        cmd.append('--enable-abstraction')
        cmd.append('--mindepth')
        cmd.append(args['abstraction_lower_bound'])
        cmd.append('--maxdepth')
        cmd.append(args['abstraction_upper_bound'])
    if args['parallel'] == 'True' or args['parallel'] == 'true':
        cmd.append('--parallel')
    if args['report_instability'] == 'True' or args['report_instability'] == 'true':
        cmd.append('--report-instability')
    if args['enable_constraints'] == 'True' or args['enable_constraints'] == 'true':
        cmd.append('--enable-constr')
    return cmd

@app.route("/seesaw", methods=["GET"])
def get():
    parser = reqparse.RequestParser()  # initialize
    parser.add_argument('program')
    parser.add_argument('abstraction')  # add args
    parser.add_argument('abstraction_lower_bound')
    parser.add_argument('abstraction_upper_bound')
    parser.add_argument('parallel')
    parser.add_argument('report_instability')
    parser.add_argument('enable_constraints')
    args = parser.parse_args()  # parse arguments to dictionary

    cmd = create_command(args)
    if not os.path.isdir("web_interface_programs"):
        os.system('mkdir web_interface_programs')
    # subprocess.call(["mkdir", "web_interface_programs"])
    subprocess.call(["echo", args['program']], stdout=open("web_interface_programs/ex1", "w"))
    # subprocess.check_output(["cat", "web_interface_programs/ex1"]).decode(sys.stdout.encoding)
    print(cmd)
    result = subprocess.check_output(cmd).decode(sys.stdout.encoding)

    # response = jsonify(message="Simple server is running")
    # response.headers.add("Access-Control-Allow-Origin", "*")

    subprocess.call(["rm", "-rf", "web_interface_programs"])
    subprocess.call(["rm", "-rf", "default.txt"])
    subprocess.call(["rm", "-rf", "outfile.txt"])
    print(result)

    # return response
    return {'data': result}, 200

@app.route("/seesaw", methods=["POST"])
def post():
    pass
    # parser = reqparse.RequestParser()  # initialize
    # parser.add_argument('userId', required=True)  # add args
    # parser.add_argument('name', required=True)
    # parser.add_argument('city', required=True)
    # args = parser.parse_args()  # parse arguments to dictionary
    #
    # # read our CSV
    # data = pd.read_csv('users.csv')
    #
    # if args['userId'] in list(data['userId']):
    #     return {
    #         'message': f"'{args['userId']}' already exists."
    #     }, 409
    # else:
    #     # create new dataframe containing new values
    #     new_data = pd.DataFrame({
    #         'userId': [args['userId']],
    #         'name': [args['name']],
    #         'city': [args['city']],
    #         'locations': [[]]
    #     })
    #     # add the newly provided values
    #     data = data.append(new_data, ignore_index=True)
    #     data.to_csv('users.csv', index=False)  # save back to CSV
    #     return {'data': data.to_dict()}, 200  # return data with 200 OK

cors = CORS(app)

if __name__ == '__main__':
    app.run()  # run our Flask app
