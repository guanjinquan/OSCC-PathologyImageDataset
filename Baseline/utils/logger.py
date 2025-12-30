class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def write(self, message):
        with open(self.log_path, 'a') as f:
            print(message, file=f)

    def log_and_print(self, message):
        print(message)
        self.log(message)