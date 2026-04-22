# pj18_sentiment

Влияние новостей на ностроение рынка.

Автоматизированная торговля фьючерсами MOEX.

## Пайплайн одним запуском

```bash
python run_all.py
```

`run_all.py` — единственная точка входа. Регистрируется в Windows Task Scheduler
на 21:00:05 ежедневно. В другое время торговля на этих счетах не ведётся.

## Установка

```bash
pip install -r requirements.txt
```

Дополнительно: установленный и запущенный **Ollama**
(`http://localhost:11434`) с моделями `gemma3:12b`, `qwen3:14b` (или другие модели из `sentiment_model`).
QUIK с доступом к `algotrade/input.tri` и включёнными lua-скриптами
`quik_export_minutes.lua` / `quik_export_positions.lua`.
