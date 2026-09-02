class NambikkaiBaseError(Exception):
    pass


class ProviderError(NambikkaiBaseError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class ValidationError(NambikkaiBaseError):
    pass


class GatingError(NambikkaiBaseError):
    pass
