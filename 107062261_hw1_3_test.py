import numpy as np
import pickle as pickle

class Game:
    def __init__(self, p1, p2, Q_learn=None, Q = {}, alpha = 0.3, gamma=0.9):
        self.p1 = p1
        self.p2 = p2
        self.current = p1
        self.other = p2
        self.board = Board()

        self.Q_learn = Q_learn
        if self.Q_learn:
            self.Q = Q
            self.alpha = alpha
            self.gamma = gamma
            self.share_Q()

    @property
    def Q_learn(self):
        if self._Q_learn is not None:
            return self._Q_learn
        if isinstance(self.p1, Computer) or isinstance(self.p2, Computer):
            return True

    @Q_learn.setter
    def Q_learn(self, _Q_learn):
        self._Q_learn = _Q_learn
    
    def share_Q(self):
        if isinstance(self.p1, Computer):
            self.p1.Q = self.Q
        if isinstance(self.p2, Computer):
            self.p2.Q = self.Q

        
    def reset(self):
        self.board = Board(board=np.zeros((3,3)))
        self.current = self.p1
        self.other = self.p2

    def switch(self):
        if self.current == self.p1:
            self.current = self.p2
            self.other = self.p1
        else:
            self.current = self.p1
            self.other = self.p2

    def learn(self, move):
        state = Computer.addState(self.board, self.current.player, self.Q)
        nextBoard = self.board.update(move, self.current.player)
        reward = nextBoard.reward()
        nextState = Computer.addState(nextBoard, self.other.player, self.Q)

        if nextBoard.over:
            result = reward
        else:
            nextQ = self.Q[nextState]
            if self.current.player == 1:
                result = reward + self.gamma*min(nextQ.values())
            else:
                result = reward + self.gamma*max(nextQ.values())
        delta = self.alpha*(result - self.Q[state][move])
        self.Q[state][move] += delta
    
    #computer
    def play(self):
        while not self.board.over:
            #p1
            positions = self.board.available()
            p1Move = self.p1.action(positions, self.board)
            if self.Q_learn:
                self.learn(p1Move)

            
            if self.board.winner() is not None:
                pass
            else:
                self.switch()
                #change player
                positions = self.board.available()
                p2Move = self.p2.action(positions, self.board)
                if self.Q_learn:
                    self.learn(p2Move)
                if self.board.winner() is None:
                    self.switch()

    def playHuman(self):
        while not self.board.over:
            #p1
            positions = self.board.available()
            p1Move = self.p1.action(positions, self.board)
            self.board.update(p1Move, self.p1.player)
            if isinstance(self.p1, Computer) or isinstance(self.p1, Random):
                print(p1Move)
                self.show()
            if self.board.winner() is not None:
                if self.board.winner() == 1:
                    print("p1 Win")
                else:
                    print("Draw")
            else:
                self.switch()
                #change player
                positions = self.board.available()
                p2Move = self.p2.action(positions, self.board)
                self.board.update(p2Move, self.p2.player)
                if isinstance(self.p2, Computer) or isinstance(self.p2, Random):
                    print(p2Move)
                    self.show()
                
                if self.board.winner() is not None:
                    if self.board.winner() == -1:
                        print("p2 Win")
                    else:
                        print("Draw")
                else:
                    self.switch()

    def show(self):
        for i in range(3):
            print('-------------')
            out = '| '
            for j in range(3):
                if self.board.board[i, j] == 1:
                    token = 'x'
                if self.board.board[i, j] == -1:
                    token = 'o'
                if self.board.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')

class Board:
    def __init__(self, board=np.zeros((3,3))):
        self.board = board
        self.over = False
    def Hash(self, player):
        # self.boardHash = str(self.board.reshape(3*3))
        self.boardHash = self.board.reshape(3*3)
        st = ''
        for i in self.boardHash:
            st += str(int(i))
        return st + str(player)
    def winner(self):
        for i in range(3):
            if sum(self.board[i,:]) == 3 or sum(self.board[:,i]) == 3:
                # print("row, column p1")
                self.over = True
                return 1 #X
            if sum(self.board[i,:]) == -3 or sum(self.board[:,i]) == -3:
                # print("row, column p2")
                self.over = True
                return -1 #o
        
        diagonal = sum([self.board[i,i] for i in range(3)])
        cross = sum([self.board[2-i,i] for i in range(3)])

        if diagonal == 3 or cross == 3:
            # print("dia cross p1")
            self.over = True
            return 1
        if diagonal == -3 or cross == -3:
            # print("dia cross p2")
            self.over = True
            return -1

        if len(self.available()) == 0: #no move left
            # print("no more")
            self.over = True
            return 0 #Draw
        
        #not end
        self.over = False
        return None
        
    def available(self):
        return [(i,j) for i in range(3) for j in range(3) if self.board[i,j] == 0]

    def reward(self):
        res = self.winner()
        if self.over:
            if res == 1:
                return 1.0
            elif res == -1:
                return -1.0
            else:
                return 0.5 #draw
        else: #game not over
            return 0.0

    def update(self, move, player):
        self.board[move] = player
        return self

class Player(object):
    def __init__(self, player):
        self.player = player
    
class Computer(Player):
    def __init__(self, player, Q = {}, epsilon = 0.2):
        super().__init__(player = player)
        self.Q = Q
        self.epsilon = epsilon
    
    def action(self, positions, board):
        if np.random.uniform() < self.epsilon:
            moves = board.available()
            print(moves)
            if moves:
                return moves[np.random.choice(len(moves))]
        else:
            state = self.addState(board, self.player, self.Q)
            Q1 = self.Q[state]
            if self.player == 1:
                return self.argminmax(Q1, max)
            else:
                return self.argminmax(Q1, min)

    @staticmethod        
    def addState(board, player, Q):
        default = 1.0
        state = board.Hash(player)
        # print(state)
        if Q.get(state) is None:
            moves = board.available()
            Q[state] = {move: default for move in moves}
        return state

    @staticmethod  
    def argminmax(Q, minmax): 
        minmaxQ = minmax(list(Q.values()))
        if list(Q.values()).count(minmaxQ) > 1:
            #if there are multiple choice, randomly choose one
            bestMove = [move for move in list(Q.keys()) if Q[move] == minmaxQ]
            move = bestMove[np.random.choice(len(bestMove))]
        else:
            move = minmax(Q, key = Q.get)
        return move

class Me(Player):
    def __init__(self, player, Q = {}):
        super().__init__(player = player)
        self.Q = Q
    
    def action(self, player, state, board):
        state = self.addState(board, self.player, state, self.Q)
        # print(state)
        Q1 = self.Q[state]
        if player == 1:
            return self.argminmax(Q1, max)
        else:
            return self.argminmax(Q1, min)

    @staticmethod        
    def addState(board, player, state, Q):
        default = 1.0
        if Q.get(state) is None:
            # print('None')
            moves = [(i,j) for i in range(3) for j in range(3) if board[i,j] == 0]
            Q[state] = {move: default for move in moves}
        return state

    @staticmethod  
    def argminmax(Q, minmax): 
        minmaxQ = minmax(list(Q.values()))
        if list(Q.values()).count(minmaxQ) > 1:
            #if there are multiple choice, randomly choose one
            bestMove = [move for move in list(Q.keys()) if Q[move] == minmaxQ]
            move = bestMove[np.random.choice(len(bestMove))]
        else:
            move = minmax(Q, key = Q.get)
        return move

class Human(Player):
    def action(self, move, board):
        while True:
            string = []
            x = input()
            for y in x:
                if y.isdigit():
                    string.append(y)
            action = tuple(map(int, string))
            if action in move:
                return action

class Random(Computer):
    def action(self, move, board):
        print(move)
        return move[np.random.choice(len(move))]

Q = pickle.load(open("107062261_hw1_3_data.p", "rb"))
while True:
    try: 
        x = input()
    except EOFError:
        break
    board = np.zeros((3,3))
    inp = x.split()
    symbol = inp[0]
    boardState = inp[1:]
    step = 0
    for i in range(3):
        for j in range(3):
            board[i,j] = boardState[step]
            step += 1
    State = ''
    for i in boardState:
        State += str(i)
    State += symbol
    user = Me(int(symbol), Q=Q)
    move = user.action(int(symbol), State, board)
    i, j = move
    print(str(j) + ' ' + str(i))
