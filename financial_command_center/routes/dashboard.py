from flask import Blueprint, render_template
from project.core.strategy_registry import list_styles, list_strategies_for_style

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/")
def dashboard():
    styles = list_styles() or []

    strategies = []
    for style in styles:
        strategies.extend(list_strategies_for_style(style) or [])

    return render_template("dashboard.html", strategies=strategies)