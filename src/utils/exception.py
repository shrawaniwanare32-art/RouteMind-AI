"""Project-wide custom exceptions that always tell you *where* things broke."""
import sys
import traceback


def error_message_detail(error, error_detail=sys) -> tuple[str, str, int | None]:
    """Return (reason, file_name, line_number) for an exception."""
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        # walk to the innermost frame - that is where the real error happened
        while exc_tb.tb_next is not None:
            exc_tb = exc_tb.tb_next
        return str(error), exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno
    # raised outside an `except` block -> use the caller's frame
    frames = traceback.extract_stack()
    caller = frames[-3] if len(frames) >= 3 else frames[-1]
    return str(error), caller.filename, caller.lineno


class CustomException(Exception):
    """Base class: message + file + line number."""

    def __init__(self, error, error_detail=sys):
        if isinstance(error, CustomException):
            self.reason, self.file_name, self.line_no = error.reason, error.file_name, error.line_no
        else:
            self.reason, self.file_name, self.line_no = error_message_detail(error, error_detail)
        super().__init__(self.reason)

    def __str__(self) -> str:
        file_short = str(self.file_name).replace("\\", "/").split("/")[-1]
        return f"{self.__class__.__name__}: {self.reason} | file: {file_short} | line: {self.line_no}"

    def to_dict(self) -> dict:
        return {
            "error": self.__class__.__name__,
            "reason": self.reason,
            "file": str(self.file_name).replace("\\", "/").split("/")[-1],
            "line": self.line_no,
        }


class DataIngestionException(CustomException):
    pass


class DataValidationException(CustomException):
    pass


class DataTransformationException(CustomException):
    pass


class ModelTrainingException(CustomException):
    pass


class PredictionException(CustomException):
    pass


class MonitoringException(CustomException):
    pass


class APIException(CustomException):
    pass
