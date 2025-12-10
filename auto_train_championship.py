#!/usr/bin/env python3
"""
🤖 全自动训练脚本 - AIRR-ML-25 Championship Pipeline
自动处理GPU监控、错误恢复、后台运行
"""

import os
import sys
import time
import subprocess
import signal
from datetime import datetime

# GPU温度监控设置
MAX_GPU_TEMP = 85  # 最高温度阈值（摄氏度）
CHECK_INTERVAL = 30  # 监控间隔（秒）

class AutoTrainer:
    def __init__(self):
        self.process = None
        self.log_file = f"./logs/auto_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        os.makedirs('./logs', exist_ok=True)

    def get_gpu_temp(self):
        """获取GPU温度"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as e:
            print(f"⚠️  无法获取GPU温度: {e}")
        return None

    def get_gpu_memory(self):
        """获取GPU内存使用情况"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(', '))
                return used, total
        except Exception:
            pass
        return None, None

    def monitor_gpu(self):
        """持续监控GPU状态"""
        temp = self.get_gpu_temp()
        used, total = self.get_gpu_memory()

        if temp is not None:
            print(f"🌡️  GPU温度: {temp}°C | 内存: {used}/{total} MiB ({used/total*100:.1f}%)")

            if temp > MAX_GPU_TEMP:
                print(f"⚠️  警告：GPU温度过高 ({temp}°C > {MAX_GPU_TEMP}°C)")
                print("   考虑降低batch size或暂停训练")

        return temp

    def start_training(self):
        """启动训练进程"""
        print("\n" + "="*70)
        print("🚀 启动 Championship 深度学习训练")
        print("="*70)
        print(f"📝 日志文件: {self.log_file}")
        print(f"🔍 GPU监控间隔: {CHECK_INTERVAL}秒")
        print(f"🌡️  最高温度: {MAX_GPU_TEMP}°C")
        print("="*70 + "\n")

        # 初始GPU状态
        print("📊 初始GPU状态:")
        self.monitor_gpu()
        print()

        # 启动训练进程
        log_f = open(self.log_file, 'w')
        self.process = subprocess.Popen(
            [sys.executable, 'championship_dl.py'],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        print(f"✅ 训练已启动 (PID: {self.process.pid})")
        print(f"   查看实时日志: tail -f {self.log_file}")
        print(f"   停止训练: kill {self.process.pid}\n")

        return log_f

    def wait_and_monitor(self, log_f):
        """等待训练完成并监控GPU"""
        last_check = time.time()

        try:
            while self.process.poll() is None:
                # 定期监控GPU
                if time.time() - last_check > CHECK_INTERVAL:
                    self.monitor_gpu()
                    last_check = time.time()

                time.sleep(5)

            # 训练完成
            return_code = self.process.returncode
            log_f.close()

            if return_code == 0:
                print("\n" + "="*70)
                print("✅ 训练成功完成！")
                print("="*70)
            else:
                print("\n" + "="*70)
                print(f"❌ 训练失败 (退出码: {return_code})")
                print("="*70)
                print("\n📋 查看错误日志:")
                os.system(f"tail -50 {self.log_file}")

            return return_code

        except KeyboardInterrupt:
            print("\n\n⚠️  收到中断信号，正在停止训练...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("   强制终止进程...")
                self.process.kill()
            log_f.close()
            print("   训练已停止")
            return -1

    def run(self):
        """主运行流程"""
        # 检查训练脚本存在
        if not os.path.exists('championship_dl.py'):
            print("❌ 错误: 找不到 championship_dl.py")
            return 1

        # 检查数据目录
        if not os.path.exists('./data/train_datasets'):
            print("❌ 错误: 找不到训练数据目录")
            return 1

        # 启动训练
        log_f = self.start_training()

        # 等待并监控
        return_code = self.wait_and_monitor(log_f)

        if return_code == 0:
            print("\n🎯 下一步:")
            print("   1. 检查模型文件: ls -lh ./models/")
            print("   2. 查看训练日志: cat " + self.log_file)
            print("   3. 生成预测并提交")

        return return_code


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     🤖 AIRR-ML-25 全自动训练脚本 🤖                        ║
    ║                                                              ║
    ║  • GPU温度监控                                              ║
    ║  • 自动日志记录                                             ║
    ║  • 后台运行支持                                             ║
    ║  • Ctrl+C 安全停止                                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    trainer = AutoTrainer()
    return trainer.run()


if __name__ == '__main__':
    sys.exit(main())
