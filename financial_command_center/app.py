from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///command_center.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "replace-with-secure-key"

db = SQLAlchemy(app)
migrate = Migrate(app, db)

from financial_command_center import models
from financial_command_center.routes.dashboard import dashboard_bp
from financial_command_center.routes.strategy import strategy_bp
from financial_command_center.routes.accounting import accounting_bp
from financial_command_center.routes.news import news_bp

app.register_blueprint(dashboard_bp)
app.register_blueprint(strategy_bp)
app.register_blueprint(accounting_bp)
app.register_blueprint(news_bp)

if __name__ == "__main__":
    app.run(debug=True)
