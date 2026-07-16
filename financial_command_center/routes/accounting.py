from flask import Blueprint, render_template
from financial_command_center.models import TaxSummary

accounting_bp = Blueprint("accounting", __name__, url_prefix="/accounting")

@accounting_bp.route("")
def accounting():
    summaries = TaxSummary.query.order_by(TaxSummary.year.desc()).all()
    return render_template("accounting.html", summaries=summaries)
