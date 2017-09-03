import gym
import universe
env = gym.make('flashgames.DriftRunners2-v0')
env.configure(remotes=1)
observation_n = env.reset()
while True:
 action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
 observation_n, reward_n, done_n, info = env.step(action_n)
 print ("observation: ", observation_n)
 print ("reward: ", reward_n)
env.render()
