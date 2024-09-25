import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from DRL_env import RobotEnv


def plot_learning_curve(log_dir, file_name='learning_curve_td3.png', averaging_window=1):
    results = np.load(log_dir + '/evaluations.npz')
    timesteps = results['timesteps']
    results_mean = results['results'].mean(axis=1)

    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Results mean shape: {results_mean.shape}")

    if len(timesteps) < averaging_window:
        print(
            f"Averaging window ({averaging_window}) is larger than the number of timesteps ({len(timesteps)}). Reducing averaging window size.")
        averaging_window = len(timesteps)

    # Compute the moving average of rewards
    averaged_rewards = np.convolve(results_mean, np.ones(averaging_window) / averaging_window, mode='valid')

    # Adjust timesteps to match the length of averaged_rewards
    adjusted_timesteps = timesteps[averaging_window - 1:]

    # 특정 구간을 제외한 데이터 필터링
    mask = adjusted_timesteps < 2 * 10 ** 6
    filtered_timesteps = adjusted_timesteps[mask]
    filtered_rewards = averaged_rewards[mask]

    plt.figure(figsize=(8, 6))
    plt.plot(filtered_timesteps, filtered_rewards, color='royalblue', linestyle='-', linewidth=2)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Mean Reward', fontsize=14)
    plt.title('TD3 Learning Curve', fontsize=16)
    plt.ylim(-300, 0)
    plt.xlim(0 * 10**6, 2.25 * 10 ** 6)  # x축을 0에서 5,000,000까지 설정하여 오른쪽에 공백 추가

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 추가 정보 텍스트 박스
    textstr = '\n'.join((
        r'Moving Average: %d' % (averaging_window,),
        r'Total Timesteps: 2e6'))

    # 텍스트 박스 스타일
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='gray', linewidth=2)

    # 그래프에 텍스트 박스 추가
    plt.gca().text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold',
                   linespacing=2.0, ha='right')

    plt.savefig(file_name, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    log_dir = "./logs_td3/"

    env = RobotEnv()
    check_env(env)

    eval_env = RobotEnv()
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=1000,
                                 deterministic=True, render=False)

    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000000, callback=eval_callback)
    model.save("td3_robot")

    plot_learning_curve(log_dir, file_name='learning_curve_td3.png', averaging_window=250)
