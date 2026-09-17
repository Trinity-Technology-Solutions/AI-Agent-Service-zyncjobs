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


class DataSourceError(NambikkaiBaseError):
    """Raised when the PostgreSQL data source cannot fulfil a request."""
    pass


class DataSourceConnectionError(DataSourceError):
    """Raised when the connection pool cannot reach the database."""
    pass
