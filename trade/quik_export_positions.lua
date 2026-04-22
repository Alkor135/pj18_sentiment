--[[
    quik_export_positions.lua

    Периодически экспортирует открытые фьючерсные позиции из QUIK в JSON-файл,
    чтобы торговый Python-скрипт знал текущее состояние портфеля.

    Что делает:
      - Каждые PERIOD_MS (по умолчанию 1 час) читает таблицу futures_client_holding
        в QUIK для указанных торговых счетов (ACCOUNTS).
      - Пишет JSON-файл со списком позиций: тикер, торговый счёт, чистая позиция
        (totalnet), а также компоненты (startnet, openbuys, opensells) и varmargin
        для контроля.
      - Атомарная запись через .tmp + rename, чтобы Python не прочитал
        наполовину записанный файл.

    Формат выходного JSON:
      {
        "exported_at": "2026-04-17 21:15:00",
        "positions": [
          {
            "trdaccid": "SPBFUT192yc",
            "sec_code": "RIM6",
            "totalnet": 2,
            "startnet": 0,
            "openbuys": 2,
            "opensells": 0,
            "varmargin": 1234.5
          }
        ]
      }

    totalnet — чистая позиция (знаковая):
      >0 — лонг, <0 — шорт, 0 — вне рынка.
    Если totalnet отсутствует в таблице QUIK, вычисляется как
      startnet + openbuys - opensells.

    Запуск в QUIK:
      Сервисы → Lua скрипты → Добавить → выбрать этот файл → Запустить.
      Скрипт работает в фоне, пока QUIK запущен. Нагрузки почти нет (один
      проход по таблице раз в час). Для остановки: кнопка "Остановить" в
      диалоге Lua-скриптов.

    Настройка:
      OUT      — путь к выходному JSON-файлу.
      ACCOUNTS — список торговых счетов для фильтрации (остальные игнорируются).
      PERIOD_MS — интервал экспорта в миллисекундах (3 600 000 = 1 час).
]]

local OUT       = "C:\\Users\\Alkor\\VSCode\\pj17_combo_sentiment_embedding_trade\\trade\\quik_export\\positions.json"
local OUT_TMP   = OUT .. ".tmp"
local ACCOUNTS  = {"SPBFUT192yc", "SPBFUT16qg3"}
local PERIOD_MS = 3600000   -- 1 час

local is_run = true


function OnStop()
    is_run = false
    return 5
end


local function esc(s)
    return '"' .. tostring(s):gsub('\\', '\\\\'):gsub('"', '\\"') .. '"'
end


local function num(v)
    if v == nil then return "0" end
    return tostring(v)
end


local function dump()
    local acc_set = {}
    for _, a in ipairs(ACCOUNTS) do acc_set[a] = true end

    local n = getNumberOf("futures_client_holding")
    local items = {}

    for i = 0, n - 1 do
        local row = getItem("futures_client_holding", i)
        if row and acc_set[row.trdaccid] then
            local sn  = row.startnet  or 0
            local ob  = row.openbuys  or 0
            local os_ = row.opensells or 0
            local tn  = row.totalnet
            if tn == nil then tn = sn + ob - os_ end

            table.insert(items, string.format(
                '    {"trdaccid": %s, "sec_code": %s, "totalnet": %s, "startnet": %s, "openbuys": %s, "opensells": %s, "varmargin": %s}',
                esc(row.trdaccid),
                esc(row.sec_code),
                num(tn),
                num(sn),
                num(ob),
                num(os_),
                num(row.varmargin)
            ))
        end
    end

    local f, err = io.open(OUT_TMP, "w")
    if not f then
        message("quik_export_positions: io.open failed: " .. tostring(err), 3)
        return
    end

    local now = os.date("%Y-%m-%d %H:%M:%S")
    f:write('{\n')
    f:write('  "exported_at": ' .. esc(now) .. ',\n')
    f:write('  "positions": [\n')
    f:write(table.concat(items, ',\n') .. '\n')
    f:write('  ]\n')
    f:write('}\n')
    f:close()

    os.remove(OUT)
    os.rename(OUT_TMP, OUT)
end


function main()
    -- Первый экспорт сразу при запуске
    pcall(dump)

    while is_run do
        sleep(PERIOD_MS)
        pcall(dump)
    end
end
