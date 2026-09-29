"""Controlled errors shared by the local evaluation layers."""


class ServiceError(Exception):
    """Base class for errors that can be mapped to a safe API response later."""


class InvalidModelOutputError(ServiceError):
    """Raised when a provider result violates the model-output contract."""


class UnsupportedQuestionTypeError(ServiceError):
    """Raised when evaluation is requested for a question type not supported locally."""


class ModelProviderError(ServiceError):
    """Raised when the configured model provider cannot produce a usable result."""
