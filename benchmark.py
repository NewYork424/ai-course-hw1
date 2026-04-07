import subprocess
import sys
import time

import os

def run_benchmark(agent1, agent2, games=10, size=5):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    play_script = os.path.join(script_dir, "play.py")
    
    cmd = [
        sys.executable, play_script,
        "--agent1", agent1,
        "--agent2", agent2,
        "--size", str(size),
        "--games", str(games),
        "--quiet"
    ]
    print(f"正在运行测试: 黑方[{agent1}] vs 白方[{agent2}] ({games}局) ...请稍候")
    
    # 获取输出流字节，再进行解码（兼容 Windows gbk 终端环境）
    result = subprocess.run(cmd, capture_output=True)
    try:
        output = result.stdout.decode('utf-8')
    except UnicodeDecodeError:
        output = result.stdout.decode('gbk', errors='replace')
        
    lines = output.split('\n')
    stats_idx = -1
    for i, line in enumerate(lines):
        if "========== 统计 ==========" in line or "统计" in line:
            stats_idx = i
            break
    
    if stats_idx != -1:
        summary = '\n'.join(lines[stats_idx:]).strip()
    else:
        summary = output.strip()
        if not summary:
            summary = "运行失败或无输出:\n" + result.stderr
            
    return summary

if __name__ == "__main__":
    # 配置不同策略的黑白方组合，每种20回合
    matchups = [
        ("mcts", "random", 20),
        ("random", "mcts", 20),
        ("minimax", "random", 20),
        ("random", "minimax", 20),
        ("minimax", "mcts", 20),
        ("mcts", "minimax", 20),
    ]
    
    # 测试不同尺寸的棋盘 (注意：Minimax 在 6x6 棋盘下可能会运行较慢，耐心等待)
    board_sizes = [6]
    
    print("开始自动化基准测试...\n")
    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        for size in board_sizes:
            print(f"========== 正在测试 {size}x{size} 棋盘 ==========")
            f.write(f"========== 测试 {size}x{size} 棋盘 ==========\n")
            for a1, a2, games in matchups:
                summary = run_benchmark(a1, a2, games=games, size=size)
                
                header = f"\n=== 【棋盘 {size}x{size}】黑方: {a1}  vs  白方: {a2} ({games}局) ==="
                print(header)
                print(summary)
                print("-" * 50)
                
                f.write(header + "\n")
            f.write(summary + "\n\n")
            f.write("-" * 50 + "\n")
            
    print("\n✅ 所有测试完成！结果已保存到 benchmark_results.txt")
    print("请把 benchmark_results.txt 的内容或者终端的输出复制给我，我来为你填入报告！")