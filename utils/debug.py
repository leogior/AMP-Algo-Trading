import logging
from backtesterClass.orderBookClass import OBData

def get_step():
    return OBData.step

class StepLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # Inject the dynamic "step" into the log record
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["step"] = get_step()
        return msg, kwargs

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set up formatter with step
formatter = logging.Formatter('%(asctime)s - %(levelname)s - Step: %(step)s - %(message)s')

# Create console handler and set its level and formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Wrap the logger with the adapter
logger = StepLoggerAdapter(logger, {})
