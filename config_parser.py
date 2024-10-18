# Parser/loader of the configuration file
import json
def load_config():
    """
    Load the configuration from the 'proj_config.json' file.

    Returns:
        dict: The configuration data loaded from the JSON file.
    """
    with open("proj_config.json", "r") as file:
        config = json.load(file)
    return config

def get_selected_template():
    """
    Retrieve the selected template name from the configuration.

    Returns:
        str: The name of the selected main template.
    """
    return load_config()["proj_settings"][0]["mainTemplate"]

def get_main_template_data():
    """
    Get the data for the main template from the configuration.

    Returns:
        dict: The data corresponding to the main template.
    """
    return load_config()[get_selected_template()][0]

def get_random_seed():
    """
    Retrieve the 'random_seed' value from the main template data.

    Returns:
        int: The 'random_seed' value.
    """
    return get_main_template_data()["random_seed"]

def get_print_data_info():
    """
    Retrieve the 'print_data_info' value from the main template data.

    Returns:
        Any: The 'print_data_info' value.
    """
    return get_main_template_data()["print_data_info"]