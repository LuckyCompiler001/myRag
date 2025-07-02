import sys
import os   
import logging

def func0()->None:
    # this is a test function
    print("This is a test function in main.py")
    # in the case you want to test the library. like loggging
    log_dir = './output'
    log_file_name = 'logfile.log'
    log_path = os.path.join(log_dir, log_file_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')  # Create empty file
        
    logging.basicConfig(filename ='./output/logfile.log', 
                        level = logging.INFO,
                        format ='%(message)s - %(filename)s - funcname: %(funcName)s - %(asctime)s - %(levelname)s')
    logging.info("This will be saved in the log file. And this is a test log message to track our work. hello from main.py. you will be able to see time, level, message and file and function name")


if __name__ == "__main__":
    func0()