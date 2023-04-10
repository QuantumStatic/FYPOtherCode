class SaveFailedException(Exception):
    def __init__(self, msg:str = "Save Failed too many times."):
        super().__init__(msg)

class WorksheetNotFoundException(Exception):
    def __init__(self, name:str):
        super().__init__(f"Worksheet named '{name}' not found.")

class DataMissingException(Exception):
    def __init__(self, msg:str=None):
        super().__init__(f"Data Missing" if msg is not None else msg)

class CriticalDataMissingException(DataMissingException):
    def __init__(self, missing_data:str):
        super().__init__(missing_data)

class PermissionDenied(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class WorkbookCloseFailedException(Exception):
    def __init__(self, msg:str):
        super().__init__(f"Could not close file, there was the following error:\n{msg}")
