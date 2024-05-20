import importlib.util
import pandas as pd
import json
import os
import subprocess
import time
import shlex

IGNORE_INDEX = -100
current_file_path = os.path.abspath(__file__)
repo_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
import subprocess

def save_json(data, file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_module_from_path(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def openai_api_key():
    credentials = load_json(f"{repo_path}/configs/credentials.json")
    return credentials["openai_api_key"]


def import_class(class_name, path_to_class_def):
    spec = importlib.util.spec_from_file_location("class_name", path_to_class_def)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def wandb_api_key():
    credentials = load_json(f"{repo_path}/configs/credentials.json")
    return credentials["wandb_api_key"]

def import_function(file_path, function_name):
        # Load the module specified by the file path
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Retrieve the function by name from the loaded module
    func = getattr(module, function_name)
    
    # Return the function object
    return func

def execute(cmd):
    cmd = shlex.split(cmd)
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def execute_command(cmd_str):
    """
    Execute a command and print its output in real-time using streams.

    Args:
    - cmd_str (str): The command as a string to execute.

    Returns:
    - None
    """

    # Example
    for path in execute(cmd_str):
        print(path, end="")