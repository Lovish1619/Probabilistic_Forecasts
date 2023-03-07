import sys


class CustomException(Exception):

    def __int__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error_message(error_message=error_message,
                                                                        error_detail=error_detail)

    @staticmethod  # This decorator is used to retrieve information without creating objects
    def get_detailed_error_message(error_message: Exception, error_detail: sys) -> str:
        """
        error_message: Exception Object
        error_detail: Object of sys module
        return: Error message with file name and line number as a string
        """

        # Capturing various error details from sys module
        _, _, exec_tb = error_detail.exc_info()
        line_number = exec_tb.tb_frame.f_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename

        # Customized error message
        error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}]\nError message: [{error_message}] "

        return error_message

    def __str__(self):  # This function returns when print statement is called
        return self.error_message

    def __repr__(self) -> str:  # This function returns when print statement is not called
        return CustomException.__name__.str()
