"""
第三小问（选做）：Minimax 智能体

实现 Minimax + Alpha-Beta 剪枝算法，与 MCTS 对比效果。
可选实现，用于对比不同搜索算法的差异。

参考：《深度学习与围棋》第 3 章
"""

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo import scoring

__all__ = ["MinimaxAgent"]



class MinimaxAgent:
    """
    Minimax 智能体（带 Alpha-Beta 剪枝）。

    属性：
        max_depth: 搜索最大深度
        evaluator: 局面评估函数
    """

    def __init__(self, max_depth=3, evaluator=None):
        self.max_depth = max_depth
        # 默认评估函数（TODO：学生可替换为神经网络）
        self.evaluator = evaluator or self._default_evaluator
        self.cache = GameResultCache()

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        # TODO: 实现 Minimax 搜索，调用 minimax 或 alphabeta
        best_value = -float('inf')
        best_move = None
        
        self.bot_player = game_state.next_player # 记录当前由于要用大根搜索的执子方
        
        # 优先选择可以下的位置
        moves = self._get_ordered_moves(game_state)
        if not moves:
            return Move.pass_turn()

        alpha = -float('inf')
        beta = float('inf')

        for move in moves:
            next_state = game_state.apply_move(move)
            # 对手的回合，我们在下一层追求最小化（由于是零和博弈，或者直接传maximizing_player=False）
            # 但是 alpha-beta/minimax 中，maximizing_player 指我方
            value = self.alphabeta(next_state, self.max_depth - 1, alpha, beta, False)
            if value > best_value:
                best_value = value
                best_move = move
            # 根节点同样需要更新 alpha 进行剪枝
            alpha = max(alpha, best_value)

        return best_move if best_move is not None else Move.pass_turn()

    def minimax(self, game_state, depth, maximizing_player):
        """
        基础 Minimax 算法。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            maximizing_player: 是否在当前层最大化（True=我方）

        Returns:
            该局面的评估值
        """
        # TODO: 实现 Minimax
        # 提示：
        # 1. 终局或 depth=0 时返回评估值
        # 2. 如果是最大化方：取所有子节点最大值
        # 3. 如果是最小化方：取所有子节点最小值
        if depth == 0 or game_state.is_over():
            return self.evaluator(game_state)
            
        moves = self._get_ordered_moves(game_state)
        if not moves:
            moves = [Move.pass_turn()]
            
        if maximizing_player:
            max_eval = -float('inf')
            for move in moves:
                next_state = game_state.apply_move(move)
                eval_val = self.minimax(next_state, depth - 1, False)
                max_eval = max(max_eval, eval_val)
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                next_state = game_state.apply_move(move)
                eval_val = self.minimax(next_state, depth - 1, True)
                min_eval = min(min_eval, eval_val)
            return min_eval

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Alpha-Beta 剪枝优化版 Minimax。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            alpha: 当前最大下界
            beta: 当前最小上界
            maximizing_player: 是否在当前层最大化

        Returns:
            该局面的评估值
        """
        # TODO: 实现 Alpha-Beta 剪枝
        # 提示：在 minimax 基础上添加剪枝逻辑
        # - 最大化方：如果 value >= beta 则剪枝
        # - 最小化方：如果 value <= alpha 则剪枝
        if depth == 0 or game_state.is_over():
            return self.evaluator(game_state)
            
        # Zobrist hash 只缓存了棋盘状态，需要加上当前轮到谁下棋，否则不同执子方面对同一盘面会使用错误的缓存
        zobrist_hash = (game_state.board.zobrist_hash(), game_state.next_player)
        cached_result = self.cache.get(zobrist_hash)
        if cached_result is not None and cached_result['depth'] >= depth:
            flag = cached_result['flag']
            val = cached_result['value']
            if flag == 'exact':
                return val
            elif flag == 'lower' and val >= beta:
                return val
            elif flag == 'upper' and val <= alpha:
                return val
            
        moves = self._get_ordered_moves(game_state)
        if not moves:
            moves = [Move.pass_turn()]
            
        orig_alpha = alpha
            
        if maximizing_player:
            max_eval = -float('inf')
            for move in moves:
                next_state = game_state.apply_move(move)
                eval_val = self.alphabeta(next_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            
            flag = 'exact'
            if max_eval <= orig_alpha:
                flag = 'upper'
            elif max_eval >= beta:
                flag = 'lower'
            self.cache.put(zobrist_hash, depth, max_eval, flag)
            
            return max_eval
        else:
            min_eval = float('inf')
            orig_beta = beta
            for move in moves:
                next_state = game_state.apply_move(move)
                eval_val = self.alphabeta(next_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
                    
            flag = 'exact'
            if min_eval <= alpha:
                flag = 'upper'
            elif min_eval >= orig_beta:
                flag = 'lower'
            self.cache.put(zobrist_hash, depth, min_eval, flag)
            
            return min_eval

    def _default_evaluator(self, game_state):
        """
        默认局面评估函数（简单版本）。

        学生作业：替换为更复杂的评估函数，如：
            - 气数统计
            - 眼位识别
            - 神经网络评估

        Args:
            game_state: 游戏状态

        Returns:
            评估值（正数对我方有利）
        """
        # TODO: 实现简单的启发式评估
        # 示例：子数差 + 气数差
        territory = scoring.evaluate_territory(game_state.board)
        b_stones = territory.num_black_stones
        w_stones = territory.num_white_stones
        
        my_stones = b_stones if self.bot_player == Player.black else w_stones
        opponent_stones = w_stones if self.bot_player == Player.black else b_stones
        
        # 统计双方的气数
        my_liberties = 0
        opponent_liberties = 0
        
        # 使用集合避免由于同一个字符串多个棋子导致气数被重复计算
        seen_points = set()
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                p = Point(row=r, col=c)
                if p in seen_points:
                    continue
                    
                go_string = game_state.board.get_go_string(p)
                if go_string is not None:
                    # 记录这个棋块所有的棋子，避免重复统计
                    for stone in go_string.stones:
                        seen_points.add(stone)
                        
                    if go_string.color == self.bot_player:
                        my_liberties += go_string.num_liberties
                    else:
                        opponent_liberties += go_string.num_liberties
        
        # 棋子数量作为主要指标（权重10），气数作为次要指标（权重1），驱动多吃子和少被吃的策略
        score = (my_stones - opponent_stones) * 10 + (my_liberties - opponent_liberties)
        
        return score

    def _get_ordered_moves(self, game_state):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """
        # TODO: 实现棋步排序
        # 提示：优先检查吃子、提子、连络等好棋
        moves = game_state.legal_moves()
        if not moves:
            return []

        def move_heuristic(move):
            if not move.is_play:
                return -100  # pass放最后
            
            score = 0
            point = move.point
            player = game_state.next_player
            opponent = player.other
            
            # 简单检查周围对方棋子气数为1（可以吃子）或者自己棋子气数为1（需要长气）
            for r, c in [(point.row+1, point.col), (point.row-1, point.col), 
                         (point.row, point.col+1), (point.row, point.col-1)]:
                neighbor = Point(r, c)
                if game_state.board.is_on_grid(neighbor):
                    neighbor_string = game_state.board.get_go_string(neighbor)
                    if neighbor_string is not None:
                        if neighbor_string.color == opponent and neighbor_string.num_liberties == 1:
                            score += 10  # 优先尝试能够吃子的落点
                        elif neighbor_string.color == player and neighbor_string.num_liberties == 1:
                            score += 5   # 其次尝试能给危险己方棋子长气/连络的落点
                            
            return score

        return sorted(moves, key=move_heuristic, reverse=True)



class GameResultCache:
    """
    局面缓存（Transposition Table）。

    用 Zobrist 哈希缓存已评估的局面，避免重复计算。
    """

    def __init__(self):
        self.cache = {}

    def get(self, zobrist_hash):
        """获取缓存的评估值。"""
        return self.cache.get(zobrist_hash)

    def put(self, zobrist_hash, depth, value, flag='exact'):
        """
        缓存评估结果。

        Args:
            zobrist_hash: 局面哈希
            depth: 搜索深度
            value: 评估值
            flag: 'exact'/'lower'/'upper'（精确值/下界/上界）
        """
        # TODO: 实现缓存逻辑（考虑深度优先替换策略）
        # 深度优先替换策略：仅在更深时覆盖
        existing = self.cache.get(zobrist_hash)
        if existing is None or existing['depth'] < depth:
            self.cache[zobrist_hash] = {
                'depth': depth,
                'value': value,
                'flag': flag
            }
