import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from constants import ONNX_MODEL, OPEN_FILE_BUTTON, NETRON_URL

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(
    executable_path="C:/Users/karki/Downloads/chromedriver_win32/chromedriver.exe",
    options=options,
)
driver.get(NETRON_URL)


def open_onnx_file():
    """
    Helper function that uses selenium webdriver to open
    an onnx file containing the trained model
    Args:
        onnx_model (file name/path): path to onnx model
    """
    element = driver.find_element_by_xpath("//*[@id='open-file-button']")
    driver.implicitly_wait(5)
    element.click()
    # element = driver.find_element_by_id(OPEN_FILE_BUTTON).click()
    full_path = str(os.path.abspath(ONNX_MODEL))
    print(os.path.abspath(ONNX_MODEL))
    driver.implicitly_wait(5)
    element.send_keys(full_path)
    print("upload success")
    # element.click()
