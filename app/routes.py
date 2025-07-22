from flask import Blueprint, request, jsonify
from app.services.sql_generator import generate_sql

bp = Blueprint('routes', __name__)

@bp.route('/generate-sql', methods=['POST'])
def handle_generate_sql():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        sql_query = generate_sql(data['query'])

        # Return error if SQL generation failed
        if sql_query.startswith("ERROR:"):
            return jsonify({"error": sql_query.replace("ERROR: ", "")}), 400

        return jsonify({"sql": sql_query})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
