"""
MCTS (蒙特卡洛树搜索) 智能体模板。

学生作业：完成 MCTS 算法的核心实现。
参考：《深度学习与围棋》第 4 章
"""

from random import random

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move

__all__ = ["MCTSAgent"]



class MCTSNode:
    """
    MCTS 树节点。


    属性：
        game_state: 当前局面
        parent: 父节点（None 表示根节点）
        children: 子节点列表
        visit_count: 访问次数
        value_sum: 累积价值（胜场数）
        prior: 先验概率（来自策略网络，可选）
    """

    def __init__(self, game_state, parent=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    @property
    def value(self):
        """计算平均价值 = value_sum / visit_count，防止除零。"""
        # TODO: 实现价值计算
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def is_leaf(self):
        """是否为叶节点（未展开）。"""
        return len(self.children) == 0

    def is_terminal(self):
        """是否为终局节点。"""
        return self.game_state.is_over()

    def best_child(self, c=1.414):
        """
        选择最佳子节点（UCT 算法）。

        UCT = value + c * sqrt(ln(parent_visits) / visits)

        Args:
            c: 探索常数（默认 sqrt(2)）

        Returns:
            最佳子节点
        """
        # TODO: 实现 UCT 选择
        import math
        import random
        best_children = []
        best_uct = float('-inf')
        for child in self.children:
            if child.visit_count == 0:
                uct = float('inf')
            else:
                uct = child.value + c * math.sqrt(math.log(self.visit_count) / child.visit_count)
            
            if uct > best_uct:
                best_uct = uct
                best_children = [child]
            elif uct == best_uct:
                best_children.append(child)
                
        return random.choice(best_children) if best_children else None

    def expand(self):
        """
        展开节点：为所有合法棋步创建子节点。

        Returns:
            新创建的子节点（用于后续模拟）
        """
        # TODO: 实现节点展开
        moves = self.game_state.legal_moves()
        for move in moves:
            next_state = self.game_state.apply_move(move)
            child_node = MCTSNode(next_state, parent=self)
            self.children.append(child_node)

    def backup(self, value):
        """
        反向传播：更新从当前节点到根节点的统计。

        Args:
            value: 从当前局面模拟得到的结果（1=胜，0=负，0.5=和）
        """
        # TODO: 实现反向传播
        self.value_sum += value
        self.visit_count += 1
        if self.parent:
            self.parent.backup(1.0 - value)  # 反向传播给对手视角的结果


class MCTSAgent:
    """
    MCTS 智能体。

    属性：
        num_rounds: 每次决策的模拟轮数
        temperature: 温度参数（控制探索程度）
    """

    def __init__(self, num_rounds=1000, temperature=1.0):
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        流程：
            1. 创建根节点
            2. 进行 num_rounds 轮模拟：
               a. Selection: 用 UCT 选择路径到叶节点
               b. Expansion: 展开叶节点
               c. Simulation: 随机模拟至终局
               d. Backup: 反向传播结果
            3. 选择访问次数最多的棋步

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        # TODO: 实现 MCTS 主循环
        root_node = MCTSNode(game_state)
        for _ in range(self.num_rounds):
            node = root_node
            while not node.is_leaf() and not node.is_terminal():
                node = node.best_child()

            if not node.is_terminal():
                # 标准 MCTS: 只有被访问过的叶子节点才需要展开
                if node.visit_count > 0:
                    node.expand()
                    if node.children:
                        import random
                        node = random.choice(node.children)

            value = self._simulate(node.game_state)
            node.backup(value)

        return self._select_best_move(root_node)

    def _simulate(self, game_state):
        """
        快速模拟（Rollout）：随机走子至终局。

        【第二小问要求】
        标准 MCTS 使用完全随机走子，但需要实现至少两种优化方法：
        1. 启发式走子策略（如：优先选有气、不自杀、提子的走法）
        2. 限制模拟深度（如：最多走 20-30 步后停止评估）
        3. 其他：快速走子评估（RAVE）、池势启发等

        Args:
            game_state: 起始局面

        Returns:
            从当前玩家视角的结果（1=胜，0=负，0.5=和）
        """
        import random
        from dlgo.scoring import compute_game_result

        current_state = game_state
        # 修正：我们需要返回“刚刚落子那一方”（即导致这个状态的玩家）的胜负结果。
        # 原来等于 current_state.next_player，这会导致最大化对手的胜率，从而疯狂选择认输这种给对手机会的步主。
        player = current_state.next_player.other

        # 优化2：限制模拟深度（例如最多走30步），防止模拟时间过长
        MAX_DEPTH = 60
        depth = 0

        while not current_state.is_over() and depth < MAX_DEPTH:
            moves = list(current_state.legal_moves())
            if not moves:
                break
            
            # 优化1：启发式走子。这里只做一个简单的避免立即自杀并倾向占星/小目的走子。
            # 为了效率，我们随机挑选一部分走法进行启发式判定
            chosen_move = None
            random.shuffle(moves)
            for move in moves[:10]:
                if move.is_play:
                    next_state = current_state.apply_move(move)
                    go_string = next_state.board.get_go_string(move.point)
                    if go_string is not None and go_string.num_liberties > 1:
                        chosen_move = move
                        break
            
            if chosen_move is None:
                chosen_move = moves[0]
            
            current_state = current_state.apply_move(chosen_move)
            depth += 1

        # 计算得分：调用 score 或直接看谁赢
        # 由于有可能深度限制没打完，所以这里我们通过此时的盘面得分粗略判断
        game_result = compute_game_result(current_state)
        if game_result.winner == player:
            return 1.0
        elif game_result.winner is None:
            return 0.5
        else:
            return 0.0

    def _select_best_move(self, root):
        """
        根据访问次数选择最佳棋步。

        Args:
            root: MCTS 树根节点

        Returns:
            最佳棋步
        """
        # TODO: 根据访问次数或价值选择
        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.game_state.last_move
