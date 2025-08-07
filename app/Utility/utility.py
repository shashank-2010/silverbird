import logging

def log_execution(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Started: {func.__name__}")
        try:
            result = func(*args,**kwargs)
            logging.info(f"Finished: {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"Error occured in {func.__name__}")
            raise
    return wrapper


