"""Contract errors.

Routes turn these into the provider-contract error body (see README).
"""


class ContractError(Exception):
    status: int = 500
    type: str = "retryable"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class InvalidRequest(ContractError):
    status = 400
    type = "invalid_request"


class PayloadTooLarge(ContractError):
    status = 413
    type = "invalid_request"


class ModelNotFound(ContractError):
    status = 404
    type = "not_found"


class Unavailable(ContractError):
    status = 503
    type = "retryable"
