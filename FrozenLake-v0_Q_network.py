import gym
from gym import wrappers
import numpy as np
import tensorflow as tf

CREATE_REPORT = 1



def modify_reward(reward, done):
    if done and reward == 0:
        return -50.0
    elif done:
        return 10.0
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
tf.set_random_seed(0)

print(env.observation_space.n, env.action_space.n)

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

#set parameters
lr = .1 #.85
y=.99
num_episodes = 1000
epsilon = 0.1
#create list to save the total rewards per episode
rList = []



tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs,W)
opt_action = tf.argmax(Qout,1)[0]

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.RMSPropOptimizer(learning_rate= lr,decay=0.99, momentum=0.9, epsilon=1e-8)
updateModel = trainer.minimize(loss)



init = tf.global_variables_initializer()

#play episode
first = 0;
with tf.Session() as sess:
    sess.run(init)
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
            a,allQ = sess.run([opt_action,Qout],feed_dict={inputs:np.identity(16)[s:s+1]})
            if np.random.rand(1) < epsilon:
                a = env.action_space.sample()
        #get new state and reward from environment by doing the action
            s_new, r, done, _ = env.step(a)
        
            r_mod = modify_reward(r,done)
        #update Q-Table with new knowledge
            Q_new = sess.run(Qout,feed_dict={inputs:np.identity(16)[s_new:s_new+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ_new = np.max(Q_new)
            targetQ = allQ
            targetQ[0,a] = r + y*maxQ_new
            #Train our network using target and predicted Q values
            sess.run([updateModel,W],feed_dict={inputs:np.identity(16)[s:s+1],nextQ:targetQ})

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
            if done == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)

        
    #outside while loop
        rList.append(rAll)

        print ("Score over time: " +  str(sum(rList)/num_episodes) + " Episodes " + str(i))


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