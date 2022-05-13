from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from constants import ONNX_MODEL, OPEN_FILE_BUTTON, NETRON_URL



def open_onnx_file(onnx_model):
    """
    Helper function that uses selenium webdriver to open
    an onnx file containing the trained model
    Args:
        onnx_model (file name/path): path to onnx model
    """
    print("opening chrome")
    driver = webdriver.Chrome()
    print("going to netron.app")
    driver.get(NETRON_URL)
    print("uploading onnx model")
    element = driver.find_element_by_id(OPEN_FILE_BUTTON)
    element.send_keys(ONNX_MODEL)
    