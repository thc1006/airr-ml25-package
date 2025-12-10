#!/bin/bash
# GPU 溫度監控腳本 - 自動在溫度過高時警告
# 不會停止訓練，只會發出警告

LOG_FILE="./logs/gpu_temp_monitor.log"
MAX_TEMP=88  # 最高安全溫度
CHECK_INTERVAL=30  # 檢查間隔（秒）

mkdir -p logs

echo "========================================================================" | tee -a "$LOG_FILE"
echo "GPU 溫度監控啟動 - $(date)" | tee -a "$LOG_FILE"
echo "最高安全溫度: ${MAX_TEMP}°C" | tee -a "$LOG_FILE"
echo "檢查間隔: ${CHECK_INTERVAL} 秒" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    # 獲取 GPU 資訊
    GPU_INFO=$(nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu --format=csv,noheader,nounits)

    if [ $? -eq 0 ]; then
        # 解析數據
        TEMP=$(echo "$GPU_INFO" | awk -F', ' '{print $1}')
        FAN=$(echo "$GPU_INFO" | awk -F', ' '{print $2}')
        POWER=$(echo "$GPU_INFO" | awk -F', ' '{print $3}')
        UTIL=$(echo "$GPU_INFO" | awk -F', ' '{print $4}')

        TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

        # 判斷溫度狀態
        if [ "$TEMP" -gt "$MAX_TEMP" ]; then
            STATUS="🔥 過熱警告"
            echo "[$TIMESTAMP] $STATUS | 溫度: ${TEMP}°C | 風扇: ${FAN}% | 功耗: ${POWER}W | 使用率: ${UTIL}%" | tee -a "$LOG_FILE"
            echo "⚠️⚠️⚠️ 溫度超過 ${MAX_TEMP}°C！建議檢查散熱或調整訓練參數！" | tee -a "$LOG_FILE"
        elif [ "$TEMP" -gt 83 ]; then
            STATUS="⚡ 溫度偏高"
            echo "[$TIMESTAMP] $STATUS | 溫度: ${TEMP}°C | 風扇: ${FAN}% | 功耗: ${POWER}W | 使用率: ${UTIL}%" | tee -a "$LOG_FILE"
        else
            STATUS="✅ 正常"
            echo "[$TIMESTAMP] $STATUS | 溫度: ${TEMP}°C | 風扇: ${FAN}% | 功耗: ${POWER}W | 使用率: ${UTIL}%" | tee -a "$LOG_FILE"
        fi
    else
        echo "[$TIMESTAMP] ❌ 無法讀取 GPU 資訊" | tee -a "$LOG_FILE"
    fi

    sleep "$CHECK_INTERVAL"
done
