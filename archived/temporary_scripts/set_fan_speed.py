#!/usr/bin/env python3
"""
GPU 風扇控制腳本 - 設定 RTX 5080 風扇速度
不會中斷訓練進程
"""

import subprocess
import sys
import time

def set_fan_speed(speed_percent=60):
    """
    設定 GPU 風扇速度

    Args:
        speed_percent: 風扇速度百分比 (0-100)
    """
    try:
        # 方法 1: 使用 nvidia-settings (需要 X server)
        print(f"🔧 正在設定 GPU 風扇速度為 {speed_percent}%...")

        # 啟用手動風扇控制
        cmd1 = [
            'nvidia-settings',
            '-a', '[gpu:0]/GPUFanControlState=1'
        ]

        # 設定風扇速度
        cmd2 = [
            'nvidia-settings',
            '-a', f'[fan:0]/GPUTargetFanSpeed={speed_percent}'
        ]

        # 執行指令（忽略 DISPLAY 錯誤）
        env = {'DISPLAY': ':0'}

        result1 = subprocess.run(cmd1, capture_output=True, text=True, env=env)
        if result1.returncode != 0:
            print(f"⚠️ 啟用手動控制失敗: {result1.stderr}")
            print("嘗試替代方法...")
            return try_alternative_method(speed_percent)

        result2 = subprocess.run(cmd2, capture_output=True, text=True, env=env)
        if result2.returncode != 0:
            print(f"⚠️ 設定風扇速度失敗: {result2.stderr}")
            print("嘗試替代方法...")
            return try_alternative_method(speed_percent)

        print(f"✅ 成功設定風扇速度為 {speed_percent}%")

        # 驗證設定
        time.sleep(2)
        check_fan_status()
        return True

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return False

def try_alternative_method(speed_percent):
    """
    替代方法：使用 nvidia-smi 或直接寫入 sysfs
    """
    print("\n嘗試使用 nvidia-smi persistence mode...")

    # 啟用 persistence mode（有時可以喚醒風扇控制）
    try:
        subprocess.run(['sudo', 'nvidia-smi', '-pm', '1'], check=False)
        print("✅ Persistence mode 已啟用")

        # 再次嘗試 nvidia-settings
        time.sleep(1)
        result = subprocess.run([
            'nvidia-settings',
            '-a', f'[fan:0]/GPUTargetFanSpeed={speed_percent}'
        ], capture_output=True, text=True, env={'DISPLAY': ':0'})

        if result.returncode == 0:
            print(f"✅ 成功設定風扇速度為 {speed_percent}%（第二次嘗試）")
            return True

    except Exception as e:
        print(f"⚠️ 替代方法失敗: {e}")

    return False

def check_fan_status():
    """檢查當前 GPU 狀態"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=temperature.gpu,fan.speed,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        temp, fan, power = result.stdout.strip().split(', ')
        print(f"\n📊 GPU 狀態:")
        print(f"   溫度: {temp}°C")
        print(f"   風扇: {fan}%")
        print(f"   功耗: {power}W")

        temp_int = int(float(temp))
        if temp_int > 88:
            print(f"   ⚠️ 溫度過高！建議檢查散熱")
        elif temp_int > 80:
            print(f"   ⚡ 溫度正常（略高）")
        else:
            print(f"   ✅ 溫度正常")

    except Exception as e:
        print(f"⚠️ 無法檢查狀態: {e}")

def monitor_temperature(interval=30, max_temp=88):
    """
    持續監控溫度（可選）

    Args:
        interval: 檢查間隔（秒）
        max_temp: 最高安全溫度
    """
    print(f"\n🔍 開始溫度監控（每 {interval} 秒檢查一次）")
    print(f"   警報溫度: {max_temp}°C")
    print("   按 Ctrl+C 停止監控\n")

    try:
        while True:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=temperature.gpu,fan.speed,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)

            temp, fan, power = result.stdout.strip().split(', ')
            temp_int = int(float(temp))

            timestamp = time.strftime("%H:%M:%S")
            status = "⚠️ 過熱!" if temp_int > max_temp else "✅"

            print(f"[{timestamp}] {status} 溫度: {temp}°C | 風扇: {fan}% | 功耗: {power}W")

            if temp_int > max_temp:
                print(f"⚠️⚠️⚠️ 溫度超過 {max_temp}°C！建議檢查散熱！")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n監控已停止")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='GPU 風扇控制工具')
    parser.add_argument('--speed', type=int, default=60,
                       help='風扇速度 (0-100%%, 預設: 60)')
    parser.add_argument('--monitor', action='store_true',
                       help='設定後持續監控溫度')
    parser.add_argument('--interval', type=int, default=30,
                       help='監控間隔秒數 (預設: 30)')

    args = parser.parse_args()

    # 設定風扇速度
    success = set_fan_speed(args.speed)

    # 可選：持續監控
    if args.monitor and success:
        monitor_temperature(interval=args.interval)
    elif not success:
        print("\n⚠️ 風扇控制可能無法正常運作")
        print("這可能是因為:")
        print("  1. RTX 5080 使用硬體風扇控制（無法軟體覆蓋）")
        print("  2. 需要 X Server (GUI) 環境")
        print("  3. 驅動版本不支援手動控制")
        print("\n建議:")
        print("  - 檢查實體風扇是否真的在轉")
        print("  - 如果風扇在轉，nvidia-smi 可能只是誤報 0%")
        print("  - 改善機箱散熱（開側板、加風扇）")
        sys.exit(1)
