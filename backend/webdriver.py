import os
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
    driver = webdriver.Chrome(
        "C:/Users/karki/Downloads/chromedriver_win32/chromedriver.exe"
    )
    print("going to netron.app")
    driver.get(NETRON_URL)
    print("uploading onnx model")
    element = driver.find_element_by_xpath("//input[@type='file']")
    driver.implicitly_wait(200)
    # element = driver.find_element_by_id(OPEN_FILE_BUTTON).click()
    element.send_keys(os.path.abspath(ONNX_MODEL))
