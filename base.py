# base.py
class BaseChatbot:
    def __init__(self, logger=None):
        self.logger = logger

    def run(self, input_text: str):
        raise NotImplementedError("This method should be implemented by the subclass")

    def post_process(self, response: str) -> str:
        return response

    def generate_response(self, input_text: str) -> str:
        out = self.run(input_text)
        response_text = ""
        for o in out:
            response_text += o
        return self.post_process(response_text)

    def _log_event(self, event: str, details: str, further: str):
        if self.logger:
            self.logger.info(event, extra={"details": details, "further": further})
