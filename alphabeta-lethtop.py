import math

TEMPLATE_FIELD = '|e|e|e|\n|e|e|e|\n|e|e|e|\n'
HUGE_NUMBER = 1000000


class AlphaBetaNode(object):
    def __init__(self):
        pass

    def generate_children(self):
        pass

    def is_max_node(self):
        pass

    def is_end_state(self):
        pass

    def value(self):
        pass


class TicTacToe(AlphaBetaNode):
    """Class that contains current state of the game and implements AlphaBetaNode methods
    :attr state: Current state of the board (str)
    :attr state: Indicates whose turn it is (Boolean)
    """

    def __init__(self, state, crosses_turn):
        super().__init__()
        self.state = state
        self.crosses_turn = crosses_turn

    def is_end_state(self):
        return ('?' not in self.state) or self.won('x') or self.won('o')

    def won(self, c):
        triples = [self.state[0:3], self.state[3:6], self.state[6:9], self.state[::3], self.state[1::3],
                   self.state[2::3], self.state[0] + self.state[4] + self.state[8],
                   self.state[2] + self.state[4] + self.state[6]]
        combo = 3 * c
        return combo in triples

    def __str__(self):
        field = TEMPLATE_FIELD
        for c in self.state:
            field = field.replace('e', c, 1)

        return field

    def is_max_node(self):
        return self.crosses_turn

    def generate_children(self):
        """
        Generates list of all possible states after this turn
        :return: list of TicTacToe objects
        """
        children, i = [], 0
        player = 'x' if self.is_max_node() else 'o'
        
        for i in range(9):
            if self.state[i] == '?':
                new_state = self.state[:i] + player + self.state[i+1:]
                new_node = TicTacToe(new_state, not self.is_max_node())
                children.append(new_node)

        return children

    def value(self):
        """
        Current score of the game (0, 1, -1)
        :return: int
        """
        if self.won('x'): return 1
        if self.won('o'): return -1
        if self.is_end_state(): return 0



def alpha_beta_value(node):
    """Implements the MinMax algorithm with alpha-beta pruning
    :param node: State of the game (TicTacToe)
    :return: int
    """
    if node.is_max_node():
        v = max_value(node, -math.inf, math.inf)
    else: v = min_value(node, -math.inf, math.inf)
    return v


def max_value(node, alpha, beta):
    if node.is_end_state(): 
        print(node, '\n', 'MAX ENDSTATE', str(node.value()), alpha, beta, '\n')
        return node.value()
    v = -math.inf
    for child in node.generate_children():
        v = max(v, min_value(child, alpha, beta))
        alpha = max(alpha, v)
        if alpha >= beta: return v
    return v


def min_value(node, alpha, beta):
    if node.is_end_state(): 
        print(node, '\n', 'MIN ENDSTATE', str(node.value()), alpha, beta, '\n')
        return node.value()
    v = math.inf
    for child in node.generate_children():
        v = min(v, max_value(child, alpha, beta))
        beta = min(alpha, v)
        if alpha >= beta: return v
    return v




# JUNKYARD
# children = self.__str__().find("?", i)
# [char if i != child_idx else player for i, char in enumerate(self.state)]

        # child_idx = self.__str__().find("?", i)



            # i = child_idx
            # child_idx = self.__str__().find("?", i)
            # print(new_state)

            #         while child_idx != -1:
            # new_state = [char if i != child_idx else player for i, char in enumerate(self.state)]#'{}{}{}'.format(self.state[:child_idx], player, self.state[child_idx+1:])
            # new_node = TicTacToe(new_state, not self.is_max_node())
            # children.append(new_node)