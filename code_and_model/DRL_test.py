import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO, TD3
from APF_Settings import APFEnv


def get_ab(model, pos, obstacles, goal):
    env = APFEnv(pos)
    get_state = env.apf_rev_rotate(goal, obstacles)
    state = np.concatenate((
        get_state[0],
        get_state[1],
        get_state[2],
        np.array([np.linalg.norm(env.heuristic(goal))])
    ))
    np.set_printoptions(precision=2, suppress=True)
    action, _states = model.predict(state, deterministic=True)
    a = action[0]
    b = [action[1], action[2]]
    if np.linalg.norm(b) > 1:
        b = b / np.linalg.norm(b)
    b = env.apf_inverse_rotate(goal, obstacles, b)

    return a, b


# 모델 로드
model = PPO.load("ppo_robot.zip")
# model = TD3.load("td3_robot.zip")

# 로봇들의 출발점과 목표점 설정
start_positions = [
    np.array([0, 0], dtype=float),
    np.array([20, 0], dtype=float),
]

goal_positions = [
    np.array([20, 20.1], dtype=float),
    np.array([0, 20], dtype=float),
]

# 고정 장애물 배열
obstacles = np.array([], dtype=float)

# 경로 저장
paths = [[pos.copy()] for pos in start_positions]

# 경로 계획
positions = start_positions.copy()
max_steps = 10000
steps = 0

fig, ax = plt.subplots(figsize=(10, 10))
reached_goals = [False] * len(start_positions)


def init():
    ax.set_xlim(-50, 200)
    ax.set_ylim(-50, 200)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.tick_params(axis='both', which='major', labelsize=14)  # 틱 레이블 크기 조정
    ax.set_aspect('equal', adjustable='box')
    return []


def update(frame):
    global steps
    ax.clear()
    ax.set_xlim(-5, 25)
    ax.set_ylim(-5, 25)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_aspect('equal', adjustable='box')

    colors = ['red', 'blue']
    labels = ['Drone 1 Path', 'Drone 2 Path']
    new_positions = []

    for i in range(len(positions)):
        if not reached_goals[i]:
            pos = positions[i]
            goal = goal_positions[i]

            env = APFEnv(pos)
            other_robots = [np.append(other_pos, env.limit) for j, other_pos in enumerate(positions) if i != j]
            if obstacles.size > 0:
                all_obstacles = np.vstack([obstacles, other_robots])
            else:
                all_obstacles = np.array(other_robots)

            a, b = get_ab(model, pos, all_obstacles, goal)
            env = APFEnv(pos)
            new_pos = pos + np.array(env.apf_drl(goal, all_obstacles, a, b))
            new_positions.append(new_pos)
        else:
            new_positions.append(positions[i])

    for i in range(len(positions)):
        if not reached_goals[i]:
            positions[i] = new_positions[i]
            paths[i].append(new_positions[i].copy())

            if np.linalg.norm(new_positions[i] - goal_positions[i]) <= 1:
                reached_goals[i] = True

        path = np.array(paths[i])
        ax.plot(path[:, 0], path[:, 1], label=labels[i], color=colors[i], linewidth=5)
        ax.scatter(start_positions[i][0], start_positions[i][1], color=colors[i], marker='o', s=100,
                   label=f'Start {i + 1}')
        ax.scatter(goal_positions[i][0], goal_positions[i][1], color=colors[i], marker='*', s=200,  # 크기를 200으로 조정
                   label=f'Goal {i + 1}')

    if obstacles.size > 0:
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='black', fill=True, alpha=0.3)
            ax.add_patch(circle)

    steps += 1
    if steps >= max_steps or all(reached_goals):
        ani.event_source.stop()

    ax.legend(loc='upper left', fontsize=17)  # 범례를 왼쪽 위로 배치하고 글자 크기를 20으로 조정

    return []


ani = FuncAnimation(fig, update, frames=range(0, max_steps, 30), init_func=init, blit=False, repeat=False)

# 애니메이션 저장
ani.save('path_planning.gif', writer=PillowWriter(fps=30))

update(max_steps - 1)
plt.savefig('final.png', bbox_inches='tight')

plt.show()
