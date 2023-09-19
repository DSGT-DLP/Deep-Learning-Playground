"""
Deprecated endpoints, do not use
"""

# @app.route("/api/getExecutionsData", methods=["POST"])
# def executions_table():
#     try:
#         request_data = json.loads(request.data)
#         user_id = request_data["user"]["uid"]
#         record = getAllUserExecutionsData(user_id)
#         return send_success({"record": record})
#     except Exception:
#         print(traceback.format_exc())
#         return send_traceback_error()
