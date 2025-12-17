#!/usr/bin/env python3
"""
自動化訓練監控和恢復系統
- 自動檢測訓練崩潰
- 自動重啟訓練
- 監控 GPU 和 RAM 狀態
- 記錄所有問題
"""

import subprocess
import time
import os
import sys
from datetime import datetime

# 配置
CHECK_INTERVAL = 60  # 每 60 秒檢查一次
MAX_RESTARTS = 5     # 最多重啟次數
LOG_FILE = "logs/watchdog.log"

def log(msg):
    """記錄訊息到日誌和控制台"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

def is_training_running():
    """檢查訓練進程是否在運行"""
    result = subprocess.run(
        ["pgrep", "-f", "championship_dl.py"],
        capture_output=True, text=True
    )
    return result.returncode == 0

def get_gpu_status():
    """獲取 GPU 狀態"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "gpu_util": int(parts[0]),
                "mem_used": int(parts[1]),
                "mem_total": int(parts[2]),
                "temp": int(parts[3])
            }
    except Exception as e:
        log(f"GPU 狀態獲取失敗: {e}")
    return None

def get_ram_status():
    """獲取 RAM 狀態"""
    try:
        result = subprocess.run(["free", "-m"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        mem_line = lines[1].split()
        return {
            "total": int(mem_line[1]),
            "used": int(mem_line[2]),
            "available": int(mem_line[6])
        }
    except Exception as e:
        log(f"RAM 狀態獲取失敗: {e}")
    return None

def start_training():
    """啟動訓練進程"""
    log("🚀 啟動訓練進程...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/auto_train_{timestamp}.log"

    # 使用 nohup 啟動訓練
    cmd = f"nohup python3 auto_train_championship.py > {log_file} 2>&1 &"
    subprocess.run(cmd, shell=True)

    # 等待進程啟動
    time.sleep(10)

    if is_training_running():
        log(f"✅ 訓練已啟動，日誌: {log_file}")
        return True
    else:
        log("❌ 訓練啟動失敗")
        return False

def check_training_progress():
    """檢查訓練進度（從最新日誌）"""
    try:
        # 找到最新的日誌文件
        result = subprocess.run(
            "ls -t logs/auto_train*.log | head -1",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            log_file = result.stdout.strip()
            # 讀取最後幾行
            result = subprocess.run(
                f"tail -5 {log_file}",
                shell=True, capture_output=True, text=True
            )
            return result.stdout.strip()
    except Exception as e:
        log(f"進度檢查失敗: {e}")
    return "Unknown"

def check_for_models():
    """檢查是否有模型文件生成"""
    try:
        result = subprocess.run(
            "ls models/championship_fold*.pt 2>/dev/null | wc -l",
            shell=True, capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return 0

def main():
    """主監控循環"""
    log("=" * 60)
    log("🔄 自動監控系統啟動")
    log("=" * 60)

    restart_count = 0
    last_restart_time = None

    # 確保日誌目錄存在
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    while True:
        try:
            # 獲取系統狀態
            gpu = get_gpu_status()
            ram = get_ram_status()

            # 檢查是否有模型完成
            model_count = check_for_models()
            if model_count >= 8:
                log("🎉 訓練完成！所有 8 個模型已生成")
                log("請運行提交腳本生成預測結果")
                break

            # 檢查訓練是否在運行
            if is_training_running():
                # 訓練正在運行，記錄狀態
                if gpu and ram:
                    log(f"✅ 訓練運行中 | GPU: {gpu['gpu_util']}%, {gpu['mem_used']}MB, {gpu['temp']}°C | "
                        f"RAM: {ram['used']}MB/{ram['total']}MB | 模型: {model_count}/8")

                # 重置重啟計數（如果訓練運行超過 10 分鐘）
                if last_restart_time and (time.time() - last_restart_time) > 600:
                    restart_count = 0
                    log("ℹ️ 重啟計數已重置（訓練穩定運行 10 分鐘）")
                    last_restart_time = None
            else:
                # 訓練停止了
                log("⚠️ 訓練已停止！")

                if ram:
                    log(f"RAM 狀態: {ram['used']}MB/{ram['total']}MB (可用: {ram['available']}MB)")

                # 檢查是否達到最大重啟次數
                if restart_count >= MAX_RESTARTS:
                    log(f"❌ 已達到最大重啟次數 ({MAX_RESTARTS})，停止監控")
                    log("請手動檢查問題")
                    break

                # 等待一會兒讓系統穩定
                log("⏳ 等待 30 秒後重啟...")
                time.sleep(30)

                # 嘗試重啟
                if start_training():
                    restart_count += 1
                    last_restart_time = time.time()
                    log(f"🔄 重啟成功 (第 {restart_count}/{MAX_RESTARTS} 次)")
                else:
                    log("❌ 重啟失敗，等待下一次檢查")

            # 等待下一次檢查
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log("\n⛔ 監控系統被手動停止")
            break
        except Exception as e:
            log(f"❌ 監控錯誤: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
