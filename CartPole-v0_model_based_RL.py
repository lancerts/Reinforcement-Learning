import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
import gym

CREATE_REPORT = 1


env = gym.make('CartPole-v0')
#save the log (for submission)
logfile = 'tmp/CartPole-v0-experiment-1'
if CREATE_REPORT:
    env = wrappers.Monitor(env, logfile, force=True)
    
##################################################    
#Hyperparameters
##################################################
H = 30 # number of hidden layer neurons
learning_rate_P = 6e-2
learning_rate_M = 3e-3
gamma = 0.999 # discount factor for reward

model_bs = 5 # Batch size when learning from model
real_bs = 1 # Batch size when learning from real environment

total_episodes = 300
N_episodes = 90 #Number of espisodes that we start to switch from real observations to learned model in training

drawFromModel = False # When set to True, will use model for observations
trainTheModel = True # Whether to train the model
trainThePolicy = True # Whether to train the policy

# model initialization
D = 4 # input dimensionality
    
# For reproducibility
env.seed(0)
np.random.seed(0)
tf.reset_default_graph()
tf.set_random_seed(0)

##################################################    
##Policy Network
##################################################
observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
W1_P = tf.get_variable("W1_P", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1_P))
W2_P = tf.get_variable("W2_P", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2_P)
probability = tf.nn.sigmoid(score)

variables = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")
adam = tf.train.AdamOptimizer(learning_rate=learning_rate_P, epsilon=1e-7)
W1_P_Grad = tf.placeholder(tf.float32,name="batch_P_grad1")
W2_P_Grad = tf.placeholder(tf.float32,name="batch_P_grad2")
batch_P_Grad = [W1_P_Grad,W2_P_Grad]
loglik = input_y*tf.log(probability) + (1 - input_y)*tf.log(1-probability)
loss = -tf.reduce_mean(loglik * advantages) 
newGrads = tf.gradients(loss,variables)
updateGrads = adam.apply_gradients(zip(batch_P_Grad,variables))
    
    
##################################################    
##Model Network
##################################################    
H_M = 256 # model layer size

input_data = tf.placeholder(tf.float32, [None, 5])
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [H_M, 50])
    softmax_b = tf.get_variable("softmax_b", [50])

previous_state = tf.placeholder(tf.float32, [None,5] , name="previous_state")
W1_M = tf.get_variable("W1_M", shape=[5, H_M],
           initializer=tf.contrib.layers.xavier_initializer()) #Weights
B1_M  = tf.Variable(tf.zeros([H_M]),name="B1_M")  #Bias
layer1_M = tf.nn.relu(tf.matmul(previous_state,W1_M) + B1_M )

W2_M = tf.get_variable("W2_M", shape=[H_M, H_M],
           initializer=tf.contrib.layers.xavier_initializer())
B2_M = tf.Variable(tf.zeros([H_M]),name="B2_M")
layer2_M = tf.nn.relu(tf.matmul(layer1_M,W2_M) + B2_M)

wO = tf.get_variable("wO", shape=[H_M, D],
           initializer=tf.contrib.layers.xavier_initializer()) #weights for observation
wR = tf.get_variable("wR", shape=[H_M, 1],
           initializer=tf.contrib.layers.xavier_initializer()) #weights for reward
wD = tf.get_variable("wD", shape=[H_M, 1],
           initializer=tf.contrib.layers.xavier_initializer()) #weights for done

bO = tf.Variable(tf.zeros([D]),name="bO")   #bias for observation
bR = tf.Variable(tf.zeros([1]),name="bR")   #bias for reward
bD = tf.Variable(tf.ones([1]),name="bD")    #bias for done


predicted_observation = tf.matmul(layer2_M,wO,name="predicted_observation") + bO
predicted_reward = tf.matmul(layer2_M,wR,name="predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2_M,wD,name="predicted_done") + bD)

true_observation = tf.placeholder(tf.float32,[None,D],name="true_observation")
true_reward = tf.placeholder(tf.float32,[None,1],name="true_reward")
true_done = tf.placeholder(tf.float32,[None,1],name="true_done")


predicted_state = tf.concat([predicted_observation,predicted_reward,predicted_done],1)

observation_loss = tf.square(true_observation - predicted_observation)

reward_loss = tf.square(true_reward - predicted_reward)

done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1-predicted_done, 1-true_done)
done_loss = -tf.log(done_loss)

model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

modelAdam = tf.train.AdamOptimizer(learning_rate=learning_rate_M,epsilon=1e-7)
updateModel = modelAdam.minimize(model_loss)    
    
##################################################    
##Helper Functions
##################################################    

def resetGradBuffer(gradBuffer):
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer
        
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# This function uses our model to produce a new state when given a previous state and action
def stepModel(sess, xs, action):
    toFeed = np.reshape(np.hstack([xs[-1][0],np.array(action)]),[1,5])
    myPredict = sess.run([predicted_state],feed_dict={previous_state: toFeed})
#    reward = myPredict[0][:,4]
    reward = np.clip(myPredict[0][:,4],0,10)
    observation = myPredict[0][:,0:4]
#    observation[:,0] = np.clip(observation[:,0],-2.4,2.4)
#    observation[:,2] = np.clip(observation[:,2],-0.4,0.4)
#    doneP = np.clip(myPredict[0][:,5],0,1)
    doneP = myPredict[0][:,5]
    if doneP > 0.1 or len(xs)>= 300:
        done = True
    else:
        done = False
    return observation, reward, done    
    

##################################################    
##Training
##################################################        
    
    
    
xs,drs,ys,ds = [],[],[],[]
running_reward = None
runnung_sum = 0
reward_sum = 0
episode_number = 1
real_episodes = 1
init = tf.global_variables_initializer()
batch_size = real_bs




# Launch the graph
with tf.Session() as sess:  
    sess.run(init)
    observation = env.reset()
    x = observation
    gradBuffer = sess.run(variables)
    gradBuffer = resetGradBuffer(gradBuffer)
    
    while episode_number <= total_episodes:
            
        x = np.reshape(observation,[1,D])

        tfprob = sess.run(probability,feed_dict={observations: x})

        action = 1 if np.random.uniform() < tfprob else 0

        # record various intermediates (needed later for backprop)
        xs.append(x) 
        y = 1 if action == 1 else 0 
        ys.append(y)
        
        # step the  model or real environment and get new measurements
        if drawFromModel == False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = stepModel(sess,xs,action)
                
        reward_sum += reward
        
        ds.append(done*1)
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: 
            
            if drawFromModel == False: 
                real_episodes += 1
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs,drs,ys,ds = [],[],[],[] # reset array memory
            
            if trainTheModel == True:
                actions = np.array(epy[:-1])
                state_prevs = epx[:-1,:]
                state_prevs = np.hstack([state_prevs,actions])
                state_nexts = epx[1:,:]
                rewards = np.array(epr[1:,:])
                dones = np.array(epd[1:,:])
                state_nextsAll = np.hstack([state_nexts,rewards,dones])

                feed_dict={previous_state: state_prevs, true_observation: state_nexts,true_done:dones,true_reward:rewards}
                loss,pState,_ = sess.run([model_loss,predicted_state,updateModel],feed_dict)
                
            if trainThePolicy == True:
                discounted_epr = discount_rewards(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] += grad
 
               
            if episode_number%batch_size == 0: 
                if trainThePolicy == True:
                    sess.run(updateGrads,feed_dict={W1_P_Grad: gradBuffer[0],W2_P_Grad:gradBuffer[1]})
                    gradBuffer = resetGradBuffer(gradBuffer)


                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                if drawFromModel == False:
                    print ('Episode %d. Reward %f. action: %f. mean reward %f.' % (real_episodes,reward_sum/real_bs,action, running_reward/real_bs))

                    if running_reward/batch_size > 200:
                        print ("Task solved in",episode_number,'episodes!')
                        break
                reward_sum = 0

                # Once the model has been trained on N_episodes, we start alternating between training the policy
                # from the model and training the model from the real environment.
            if episode_number > N_episodes:
                drawFromModel = not drawFromModel
                trainTheModel = not drawFromModel
#                trainThePolicy = False

            
            if drawFromModel == True:
                observation = np.random.uniform(-0.1,0.1,[D]) # Generate reasonable starting point
                batch_size = model_bs
            else:
                observation = env.reset()
                batch_size = real_bs
                

print (real_episodes)    
    
    
env.close()    
#upload the result
if CREATE_REPORT:
    key = 'sk_Hp2c6vsJQCbswbDg5DGYA'
    gym.upload(logfile, api_key=key)




plt.figure(figsize=(8, 12))
for i in range(6):
    plt.subplot(6, 2, 2*i + 1)
    plt.plot(pState[:,i])
    plt.subplot(6,2,2*i+1)
    plt.plot(state_nextsAll[:,i])
plt.tight_layout()