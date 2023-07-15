import traceback

from common.utils import *


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


def send_train_results(train_loss_results: dict):
    """
    Wrapper function to send results of a user's training request (successful one)

    Params:
      - train_loss_results: dict containing info about training results + stats
    """
    return send_success(
        {
            "message": "Dataset trained and results outputted successfully",
            "dl_results": csv_to_json(),
            "auxiliary_outputs": train_loss_results,
        }
    )


def send_detection_results(object_detection_results: dict):
    """
    Wrapper function to send success message for object detection requests


    Args:
     - object_detection_results (dict): object detection results

    Returns:
     - dict
    """
    return send_success(
        {
            "message": "Detection worked successfully",
            "dl_results": object_detection_results["dl_results"],
            "auxiliary_outputs": object_detection_results["auxiliary_outputs"],
        }
    )


def send_traceback_error():
    """
    Wrapper function to send error messages related to code that the user may have entered during training request
    creation
    """
    return send_error(traceback.format_exc(limit=1))
