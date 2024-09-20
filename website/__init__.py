from flask import Flask, Blueprint


def create_app():
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    from .views import views
    # Register the blueprints
    app.register_blueprint(views, url_prefix='/')

    return app
