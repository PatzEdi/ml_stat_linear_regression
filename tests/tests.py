import unittest
import sys
import os

# File for unit tests...

# We get the current directory of the script
current_script_path = os.path.dirname(os.path.realpath(__file__)) 
sys.path.append(os.path.join(current_script_path, "..")) # This is to find the proj_config.json file
 
import config_parser

class TestConfigParser(unittest.TestCase):
    """ Class to test the config parser, e.g. test the types it returns """
    def test_load_config(self):
        """Test that the config is loaded and is a dictionary."""
        config = config_parser.load_config()
        self.assertIsInstance(config, dict)

    def test_get_selected_template(self):
        """Test that the selected template is a string."""
        selected_template = config_parser.get_selected_template()
        self.assertIsInstance(selected_template, str)

    def test_get_main_template_data(self):
        """Test that the main template data is a dictionary."""
        main_template_data = config_parser.get_main_template_data()
        self.assertIsInstance(main_template_data, dict)

    def test_get_random_seed(self):
        """Test that the target coordinates amount is an integer."""
        target_coords_amount = config_parser.get_random_seed()
        self.assertIsInstance(target_coords_amount, int)

    def test_get_print_data_info(self):
        """Test that the print data info is a boolean."""
        print_data_info = config_parser.get_print_data_info()
        self.assertIsInstance(print_data_info, bool)
        
if __name__ == '__main__':
    unittest.main()