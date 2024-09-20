# Import the 'create_app' function and the 'socketio' object from the 'website' package.
import random
from website import create_app

# Call the 'create_app' function to create a Flask application instance.
app = create_app()

# This is the main entry point of the program.
if __name__ == '__main__':
    #random_port = random.randint(5000, 9999)
    #app.run(debug=True, port=random_port)
    app.run(debug=True, host='0.0.0.0', port=5798, use_reloader=False)
