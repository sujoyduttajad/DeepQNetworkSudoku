import numpy as np
import time
import pandas as pd
import os


def split_word(word):
    return [char for char in word]


CSV_COLUMNS = ['puzzles', 'solutions']
CSV_FILE_PATH = r"/content/sudoku.csv"

# Read CSV file and sample 500 rows of data
df_puzzles = pd.read_csv(CSV_FILE_PATH, skipinitialspace=True, names=CSV_COLUMNS, index_col=False)

# Display the dataset info
df_puzzles.info()

# DataFrame type
puzzle_data = df_puzzles['puzzles'].astype(str)
solution_data = df_puzzles['solutions'].astype(str)


class SudokuBoard(object):
    def __init__(self):
        super(SudokuBoard, self).__init__()
        # Action space and features
        self.action_space = ['u', 'd', 'l', 'r', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.num_actions = len(self.action_space)
        self.num_features = 2
        self.puzzle_index = 1
        self._initialize_puzzle()

    def _initialize_puzzle(self):
        # Initialize puzzle and solution
        self.current_puzzle = puzzle_data.iloc[self.puzzle_index]
        self.puzzle_reshaped = np.asarray(self.current_puzzle)
        self.puzzle_array = split_word(str(self.puzzle_reshaped))
        self.binary_puzzle = ['0' if i == '0' else '1' for i in self.puzzle_array]
        self.puzzle_array = np.array(self.puzzle_array).reshape(9, 9)
        self.binary_puzzle_array = np.array(self.binary_puzzle).reshape(9, 9)
        self.agent_position = np.array([0, 0])

        self.current_solution = solution_data.iloc[self.puzzle_index]
        self.solution_reshaped = np.asarray(self.current_solution)
        self.solution_array = split_word(str(self.solution_reshaped))
        self.solution_array = np.array(self.solution_array).reshape(9, 9)
        self.puzzle_index += 1

    def reset(self):
        time.sleep(0.0001)
        # Reset agent position to (0, 0)
        return np.array([0, 0])

    def step(self, action):
        s = self.agent_position
        stemp = s
        if action == 0:
            if stemp[0] > 0:
                stemp[0] = stemp[0] - 1
        elif action == 1:
            if stemp[0] < 8:
                stemp[0] = stemp[0] + 1
        elif action == 2:
            if stemp[1] < 8:
                stemp[1] = stemp[1] + 1
        elif action == 3:
            if stemp[1] > 0:
                stemp[1] = stemp[1] - 1
        elif action >= 4 and action <= 12:
            if self.binary_puzzle_array[s[0], s[1]] == '0':
                self.puzzle_array[s[0], s[1]] = str(action - 3)

        if (self.puzzle_array == self.solution_array).all():
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        if action < 4:
            s_ = stemp
        else:
            s_ = s

        clear_screen = lambda: os.system('cls')
        clear_screen()
        print(self.puzzle_array)

        time.sleep(0.0001)
        return s_, reward, done
