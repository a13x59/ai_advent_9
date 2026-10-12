// config.js — конфигурация фронтенда агента.
//
// ЭТА версия — для локальной разработки (index.html открывается из IDEA/файла):
// агент запускается на localhost:8000 (DeepSeek) и localhost:8002 (Ollama).
//
// На VPS deploy.sh перезаписывает этот файл: там фронт раздаётся nginx'ом,
// поэтому base = '' (same-origin), а модель — qwen2.5:1.5b.
window.APP_CONFIG = {
    // Базовые URL агентского API.
    cloudBase: 'http://localhost:8000',
    localBase: 'http://localhost:8002',

    // Модели по умолчанию для каждого бэкенда.
    cloudModel: 'deepseek-chat',
    localModel: 'llama3.1',

    // Бэкенд, выбранный при загрузке: 'local' | 'cloud'.
    defaultBackend: 'cloud',

    // API-ключ приватного сервиса (заголовок X-Api-Key).
    // На VPS nginx проверяет ключ — введите его в поле интерфейса.
    apiKey: '',
};
