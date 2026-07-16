from flask import Blueprint, render_template
from financial_command_center.models import Strategy

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/")
def dashboard():
    strategies = Strategy.query.order_by(Strategy.name).all()
    return render_template("dashboard.html", strategies=strategies)
