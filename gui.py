import tkinter as tk
from tkinter import messagebox
from dlgo.goboard import GameState, Move
from dlgo.gotypes import Player, Point
import threading
import time
import argparse
import sys
import importlib

# 常量设置
BOARD_SIZE = 5
CELL_SIZE = 60
PADDING = 40
DOT_RADIUS = 25

class GoGUI:
    def __init__(self, root, board_size=BOARD_SIZE, bot_agent_black=None, bot_agent_white=None):
        self.root = root
        self.board_size = board_size
        self.bot_agent_black = bot_agent_black
        self.bot_agent_white = bot_agent_white
        
        self.game = GameState.new_game(board_size)

        self.root.title(f"围棋 {board_size}x{board_size} - 人机对弈")
        
        self.canvas_size = CELL_SIZE * (board_size - 1) + 2 * PADDING
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="#f3d79e")
        self.canvas.pack(padx=20, pady=20)
        
        # 绑定点击事件
        self.canvas.bind("<Button-1>", self.on_click)

        # 状态标语
        self.status_var = tk.StringVar()
        self.update_status()
        self.status_label = tk.Label(root, textvariable=self.status_var, font=("Arial", 14))
        self.status_label.pack(pady=10)

        # 控制按钮
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="重新开始", command=self.reset_game, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="跳过 (Pass)", command=self.pass_turn, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)

        self.draw_board()

        # 如果黑方是AI，起手由AI走子
        if self.bot_agent_black:
            self.bot_move()

    def draw_board(self):
        self.canvas.delete("all")
        # 画线
        for i in range(self.board_size):
            x = PADDING + i * CELL_SIZE
            # 竖线
            self.canvas.create_line(x, PADDING, x, self.canvas_size - PADDING, width=1)
            # 横线
            self.canvas.create_line(PADDING, x, self.canvas_size - PADDING, x, width=1)

        # 画星位 (对于5x5，中间点画个小星位)
        if self.board_size == 5:
            center = PADDING + 2 * CELL_SIZE
            self.canvas.create_oval(center - 4, center - 4, center + 4, center + 4, fill="black")

        # 绘制棋子
        for r in range(1, self.board_size + 1):
            for c in range(1, self.board_size + 1):
                p = Point(row=r, col=c)
                stone = self.game.board.get(p)
                if stone is not None:
                    color = "black" if stone == Player.black else "white"
                    outline = "black" if stone == Player.white else "white"
                    
                    x = PADDING + (c - 1) * CELL_SIZE
                    y = PADDING + (r - 1) * CELL_SIZE
                    
                    self.canvas.create_oval(x - DOT_RADIUS, y - DOT_RADIUS, 
                                            x + DOT_RADIUS, y + DOT_RADIUS, 
                                            fill=color, outline=outline, width=1)
                    
                    # 标记最后一手
                    if self.game.last_move and self.game.last_move.is_play and self.game.last_move.point == p:
                        mark_color = "red" if color == "white" else "red"
                        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=mark_color)

    def on_click(self, event):
        if self.game.is_over():
            return
            
        # 如果当前是AI回合，不响应点击
        current_bot = self.bot_agent_black if self.game.next_player == Player.black else self.bot_agent_white
        if current_bot:
            return

        # 计算点击最近的交叉点
        col = round((event.x - PADDING) / CELL_SIZE) + 1
        row = round((event.y - PADDING) / CELL_SIZE) + 1
        
        if 1 <= row <= self.board_size and 1 <= col <= self.board_size:
            point = Point(row=row, col=col)
            move = Move.play(point)
            
            if self.game.is_valid_move(move):
                self.game = self.game.apply_move(move)
                self.draw_board()
                self.update_status()
                
                if self.game.is_over():
                    self.update_status()
                    # 避免多次弹窗的问题，交由 update_status 统一处理
                else:
                    self.update_status()
                    # 检查下一手是否依然是AI
                    next_bot = self.bot_agent_black if self.game.next_player == Player.black else self.bot_agent_white
                    if next_bot:
                        self.bot_move()
            else:
                self.status_var.set("非法落子！")

    def pass_turn(self):
        if self.game.is_over():
            return
            
        current_bot = self.bot_agent_black if self.game.next_player == Player.black else self.bot_agent_white
        if current_bot:
            return

        move = Move.pass_turn()
        self.game = self.game.apply_move(move)
        self.draw_board()
        self.update_status()
        
        if not self.game.is_over():
            next_bot = self.bot_agent_black if self.game.next_player == Player.black else self.bot_agent_white
            if next_bot:
                self.bot_move()
            
    def bot_move(self):
        self.status_var.set("AI 思考中...")
        self.root.update()
        
        # 确定当前应当走子的AI
        current_bot = self.bot_agent_black if self.game.next_player == Player.black else self.bot_agent_white
        if not current_bot:
            return
        
        # 使用线程避免阻塞GUI
        def compute_move():
            start_time = time.time()
            move = current_bot.select_move(self.game)
            end_time = time.time()
            
            def apply():
                self.game = self.game.apply_move(move)
                self.draw_board()
                
                if move.is_pass:
                    print(f"AI passed it's turn! (Took {end_time-start_time:.2f}s)")
                elif move.is_play:
                    print(f"AI played at ({move.point.row}, {move.point.col}). (Took {end_time-start_time:.2f}s)")
                else:
                    print(f"AI resigned! (Took {end_time-start_time:.2f}s)")
                
                if self.game.is_over():
                    self.update_status()
                else:
                    self.update_status()
                    # 继续检查下一个如果是AI，则继续触发（递归）
                    next_bot = self.bot_agent_black if self.game.next_player == Player.black else self.bot_agent_white
                    if next_bot:
                        # 稍微停顿一下效果更好，不至于瞬间走完
                        self.root.after(100, self.bot_move)

            self.root.after(0, apply)
            
        threading.Thread(target=compute_move, daemon=True).start()

    def update_status(self):
        if self.game.is_over():
            self.show_winner()
            return
            
        player_color = "黑棋" if self.game.next_player == Player.black else "白棋"
        self.status_var.set(f"当前回合: {player_color}")

    def show_winner(self):
        from dlgo.scoring import compute_game_result
        result = compute_game_result(self.game)
        winner_text = "黑方胜利" if result.winner == Player.black else ("白方胜利" if result.winner == Player.white else "平局")
        msg = f"游戏结束！\n{winner_text}\n黑方得分: {result.b}\n白方得分: {result.w}\n(贴目: {result.komi})"
        self.status_var.set("游戏结束")
        messagebox.showinfo("结束", msg)

    def reset_game(self):
        self.game = GameState.new_game(self.board_size)
        self.draw_board()
        self.update_status()
        if self.bot_agent_black:
            self.bot_move()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="围棋可视化程序")
    parser.add_argument("--size", type=int, default=5, help="棋盘大小")
    parser.add_argument("--black_agent", type=str, default="None", choices=["random", "mcts", "minimax", "None"], help="选择执黑的AI，如果是None代表人类")
    parser.add_argument("--white_agent", type=str, default="None", choices=["random", "mcts", "minimax", "None"], help="选择执白的AI，如果是None代表人类")
    
    args = parser.parse_args()
    
    def init_agent(agent_type):
        if agent_type == "random":
            from agents.random_agent import RandomAgent
            return RandomAgent()
        elif agent_type == "mcts":
            from agents.mcts_agent import MCTSAgent
            return MCTSAgent(num_rounds=2000)
        elif agent_type == "minimax":
            from agents.minimax_agent import MinimaxAgent
            return MinimaxAgent()
        return None
            
    bot_black = init_agent(args.black_agent)
    bot_white = init_agent(args.white_agent)

    root = tk.Tk()
    root.resizable(False, False)
    app = GoGUI(root, board_size=args.size, bot_agent_black=bot_black, bot_agent_white=bot_white)
    root.mainloop()