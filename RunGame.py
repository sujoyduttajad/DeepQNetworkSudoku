import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['quizzes', 'solutions']

df_train = pd.read_csv(CSV_FILE_PATH,
                       skipinitialspace=True,
                       names=COLUMNS,
                       index_col=False)

num_quizzes = df_train['quizzes'].count()

# Define lists to store accuracy and episode number
accuracy_history = []
episode_history = []

def run_game():
    global correct_solutions, episode_history, accuracy_history  # Declare global variables
    correct_solutions = 0
    step = 0
    episodes = 10
    for episode in range(episodes):
        current_state = env.reset()
        while True:
            action = deep_q_network.choose_action(current_state)
            future_state, reward, done = env.step(action)
            deep_q_network.store_transition(current_state, action, reward, future_state)
            print(episode)
            if (step > 200) and (step % 5 == 0):
                deep_q_network.learn()
            current_state = future_state
            if done:
                break
            step += 1
        
        # Check if the puzzle is solved correctly
        if (env.puzzle_array == env.solution_array).all():
            correct_solutions += 1
        
        # Store accuracy and episode number
        accuracy_history.append(correct_solutions / num_quizzes)
        episode_history.append(episode + 1)

print('Game over')

if __name__ == "__main__":
    env = SudokuBoard()
    deep_q_network = CustomDeepQNetwork(env.num_actions, env.num_features,
                                        learning_rate=0.01,
                                        reward_decay=0.9,
                                        e_greedy=0.9,
                                        replace_target_iter=200,
                                        memory_size=2000)
    for i in range(1, num_quizzes):
        run_game()
        env.build_board()

# Plot the accuracy graph
plt.plot(episode_history, accuracy_history, marker='o')
plt.title('Accuracy Over Training Episodes')
plt.xlabel('Episode Number')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
