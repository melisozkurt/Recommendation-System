# Flask CSGO Market Application

This project is a Flask-based web application for managing CSGO items across various market platforms such as Shadowpay, Waxpeer, and CSGO Market.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The application requires the following to be installed:

- Python 3.8 or higher
- Pip (Python Package Index)

### Installing


1. Create a Python virtual environment and activate it.

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. Set the FLASK_APP environment variable.

```bash
export FLASK_APP=main.py
```

4. Run the application.

```bash
flask run
```

## Usage
The application provides interfaces to Shadowpay, Waxpeer, and CSGO Market platforms. Users can start or stop each bot separately, and view logs of bot activities.


## Built With

- [Flask](https://flask.palletsprojects.com/) - The web framework used

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- Python
- Flask
- and many more...