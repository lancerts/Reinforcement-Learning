import gym
from gym import wrappers
import numpy as np

CREATE_REPORT = 1

def modify_reward(reward, done):
    if done and reward == 0:
        return -100.0
    elif done:
        return 50.0
    else:
        return 1.0

#make the environment
env = gym.make('FrozenLake-v0')

#save the log (for submission)
logfile = 'tmp/FronzenLake-experiment-1'
if CREATE_REPORT:
    env = wrappers.Monitor(env, logfile, force=True)

# for reproducibility
env.seed(0)
np.random.seed(0)

print(env.observation_space.n, env.action_space.n)

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

#set parameters
lr = .05 #.85
y=.99
num_episodes = 500

#create list to save the total rewards per episode
rList = []

#play episode
first = 0;
for i in range(num_episodes):
    #reset environment and get a new observation (the first state)
    s = env.reset()
    rAll = 0 #to save total reward
    done = False
    j=0 #current timestep
    
    #The Q-Table learning algorithm
    while not done: #j<99: # will try in 100 timesteps
        #env.render()
        j+=1 #increse the timestep
        
        #choose an action by greedily picking from Q table
        # add noise to simulate uncertainty, getting smaller as episode increased
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1))) 
        
        #get new state and reward from environment by doing the action
        s_new, r, done, _ = env.step(a)
        
        r_mod = modify_reward(r,done)
        #update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r_mod+y*np.max(Q[s_new,:])-Q[s,a])

        #logging
        rAll+=r
                
        #quit if success or fall to the hole
        if done == True:
            if first == 0 and r != 0.0:
                first = i
                print(i)

            break
        
        #set new state for the next iteration
        s = s_new
        
    #outside while loop
    rList.append(rAll)

print ("Score over time: " +  str(sum(rList)/num_episodes))

print ("Final Q-Table Values")
print (np.argmax(Q,1))

env.close()

def moving_average(x, n=100):
    x = x.cumsum()
    return (x[n:] - x[:-n]) / n

ma = moving_average(np.asarray(rList))
print ("Best 100-episode average reward was %f." % ma.max())

solved = len(np.where(ma >= .78)[0])>0 # criteria set by openai
if solved:
    print ("Solved after %d episodes." % np.where(ma >= .78)[0][0])
else:
    print ("unsolved!")

#upload the result
if CREATE_REPORT:
    key = 'sk_Hp2c6vsJQCbswbDg5DGYA'
    gym.upload(logfile, api_key=key)