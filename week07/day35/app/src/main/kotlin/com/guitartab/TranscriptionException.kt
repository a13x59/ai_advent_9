package com.guitartab

/** Исключение с HTTP-статусом для ошибок бизнес-логики транскрипции. */
class TranscriptionException(val statusCode: Int, message: String) : RuntimeException(message)
