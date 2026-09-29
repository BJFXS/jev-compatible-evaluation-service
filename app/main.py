"""FastAPI entry layer for the local SystemOne service."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings
from app.errors import (
    InvalidModelOutputError,
    ModelProviderError,
    ServiceError,
    UnsupportedQuestionTypeError,
)
from app.model import Evaluator, OpenAIEvaluator
from app.schemas import SystemOneRequest, SystemOneResponse
from app.service import SystemOneService


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Return a stable public error envelope without exception details."""

    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def _provider_error_response(error: ModelProviderError) -> JSONResponse:
    """Classify safe provider categories without returning provider text."""

    message = str(error).lower()
    if "refused" in message:
        return _error_response(502, "model_refusal", "The model refused the evaluation.")
    if "authentication" in message or "configuration" in message or "api key" in message:
        return _error_response(502, "model_configuration_error", "The model provider is not configured.")
    if "response was incomplete" in message:
        return _error_response(502, "invalid_model_output", "The model returned an incomplete output.")
    if any(marker in message for marker in ("timed out", "transport failed", "rate limited", "server error")):
        return _error_response(503, "model_unavailable", "The model provider is temporarily unavailable.")
    return _error_response(502, "model_provider_error", "The model provider could not complete the evaluation.")



########################
###     ############
########################
def create_app(
    *,
    settings: Settings | None = None,
    evaluator: Evaluator | None = None,
    service: SystemOneService | None = None,
) -> FastAPI:
    """
    Build an injectable application without requiring provider credentials at import time.
        - 创建 FastAPI app，并支持 fake evaluator 或 service, 方便测试时完全不需要真实 API Key
        - 正常运行时构造 OpenAIEvaluator, 再交给 SystemOneService
    """

    if evaluator is not None and service is not None:
        raise ValueError("provide either evaluator or service, not both")

    if service is None:
        provider = evaluator if evaluator is not None else OpenAIEvaluator(settings=settings)
        service = SystemOneService(provider)

    ### 创建 FastAPI app
    application = FastAPI()
    ### 注入 service 到 app的state 中, 方便在router中使用
    application.state.systemone_service = service

    ### 处理可能的 exceptions
    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        return _error_response(422, "invalid_request", "The request does not match the local contract.")

    @application.exception_handler(InvalidModelOutputError)
    async def invalid_model_output_handler(_: Request, __: InvalidModelOutputError) -> JSONResponse:
        return _error_response(502, "invalid_model_output", "The model returned an invalid output.")

    @application.exception_handler(UnsupportedQuestionTypeError)
    async def unsupported_question_handler(_: Request, __: UnsupportedQuestionTypeError) -> JSONResponse:
        return _error_response(422, "invalid_request", "The request contains an unsupported question type.")

    @application.exception_handler(ModelProviderError)
    async def model_provider_error_handler(_: Request, error: ModelProviderError) -> JSONResponse:
        return _provider_error_response(error)

    @application.exception_handler(ServiceError)
    async def service_error_handler(_: Request, __: ServiceError) -> JSONResponse:
        return _error_response(500, "internal_error", "The service could not complete the request.")

    @application.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, __: Exception) -> JSONResponse:
        return _error_response(500, "internal_error", "The service could not complete the request.")

    ### 定义router， 并且在系统启动后 接收请求
        # FastAPI 会先把 JSON 请求转换为 schemas.py 设定的 SystemOneRequest 对象，然后传给 evaluate_request 函数
    @application.post("/v1/systemone", response_model=SystemOneResponse)
    async def systemone(request: SystemOneRequest) -> SystemOneResponse:
        return await application.state.systemone_service.evaluate_request(request)

    return application


app = create_app()
