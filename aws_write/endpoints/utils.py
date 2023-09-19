import json
import traceback


def send_success(results: dict):
    """
    Utility function to send success response from API with result data

    Params:
     - results (dict): Any data corresponding to API output

    Returns:
     - dict
    """
    return (json.dumps({"success": True, **results}), 200)


def send_error(message: str):
    """
    Utility function to send failure/error response from API with a custom error message

    Params:
     - message (str): error message

    Returns:
     - dict
    """
    return (json.dumps({"success": False, "message": message}), 400)


def send_traceback_error():
    """
    Wrapper function to send error messages related to code that the user may have entered during training request
    creation
    """
    return send_error(traceback.format_exc(limit=1))
