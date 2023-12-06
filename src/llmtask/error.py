
class FeedbackNotCalledError(Exception):
    def __init__(self, message="feedback() not called after getting prompt."):
        self.message = message
        super().__init__(self.message)
