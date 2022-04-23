# https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym
import gym
import random
random.seed(0)  # make results reproducible

num_episodes = 10


def main():
    env = gym.make('Pong-v4', render_mode='human')  # create the game
    env.seed(0)  # make results reproducible
    rewards = []

    for _ in range(num_episodes):
        env.reset()
        episode_reward = 0
        while True:
            env.render()
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)  # random action
            episode_reward += reward
            if done:
                print('Reward: %d' % episode_reward)
                rewards.append(episode_reward)
                break
    print('Average reward: %.2f' % (sum(rewards) / len(rewards)))


if __name__ == '__main__':
    main()