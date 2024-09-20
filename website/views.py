# Importing required modules and functions from Flask and other libraries
from flask import Blueprint, render_template, request, jsonify
from algo import show_interacted_products, show_suggestions_func
import json

# Initializing a Flask Blueprint which allows you to organize your views
views = Blueprint('views', __name__)



@views.route('/', methods=['GET'])
def home():
    if request.method == 'GET':
        return render_template('home.html')


@views.route('/show-products', methods=['POST'])
def show_products():
    if request.method == 'POST':
        user = json.loads(request.data)
        user_id = int(user["userIndex"])
        return jsonify({"status": "success", 'data': show_interacted_products(user_id)})
    
@views.route('/show-suggestions', methods=['POST'])
def show_suggestions():
    if request.method == 'POST':
        data = json.loads(request.data)
        user_index = int(data["userIndex"])
        no_products = int(data['noproducts'])
        selected_algorithm = data['selectedAlgorithm']
        return jsonify({"status": "success",'data': show_suggestions_func(user_index,no_products,selected_algorithm)})