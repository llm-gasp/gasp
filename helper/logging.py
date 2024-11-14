class Logging:
    def __init__(self, log_file):
        self.log_file = log_file

        # Clear the log file
        open(log_file, 'w').close()
    
    def log(self, args):
        with open(self.log_file, 'a') as log:
            log.write('----------------------\n')
            for arg in args:
                log.write(arg + '\n')
            log.write('----------------------\n')