import gym,random
import numpy as np
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from statistics import mean,median
from collections import Counter

#LR=1e-30
LR=0.03
#env
#LR=3e-4
env=gym.make('CartPole-v0').env
env.reset()
goal_steps=500
score_requirement=70
initial_games=80000
"""---def random_games():
    for ep in range(5):
        env.reset()
        for t in range(goal_step):
            env.render()
            action=env.action_space.sample()
            obs,rew,done,info=env.step(action)
            if done:
                break  -----"""
#random_games()

def initial_population():
    training_data=[]
    scores=[]
    accepted_score=[]
    for _ in range(initial_games):
        score=0
        game_memory=[]
        previous_obs=[]
        for _ in range(goal_steps):
            action=random.randrange(0,2)
            obs,reward,done,info=env.step(action)
            if len(previous_obs)>0:
                game_memory.append([previous_obs,action])
            previous_obs=obs
            score+=reward
            if done:
                break
        if score>=score_requirement:
            accepted_score.append(score)
            for data in game_memory:
                if data[1]==0:
                    output=[0,1]
                else:
                    output=[1,0]
                training_data.append([data[0],output])
        env.reset()
        scores.append(score)
    training_data_save=np.array([training_data])
    np.save('save.npy',training_data_save)
    print("Average accepted score:",mean(accepted_score))
    print("Median accepted score:",median(accepted_score))
    print(Counter(accepted_score))
    return training_data

def neural_network_model(input_size):
    network=input_data(shape=[None,input_size,1],name='input')
    network=fully_connected(network,128,activation='relu')
    #network=dropout(network,0.8)#its keeprate
    network=fully_connected(network,256,activation='relu')
    #network=dropout(network,0.8)

    network=fully_connected(network,512,activation='relu')
    #network=dropout(network,0.8)

    network=fully_connected(network,256,activation='relu')
    #network=dropout(network,0.8)

    network=fully_connected(network,128,activation='relu')
    network=dropout(network,0.8)

    network=fully_connected(network,2,activation='softmax')

    network=regression(network,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')
    
    model=tflearn.DNN(network,tensorboard_dir='log')
    return model


def train_model(training_data,model=False):
    x=np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    
    y=np.array([i[1] for i in training_data])
    

    if not model:
        model=neural_network_model(input_size=len(x[0]))
    model.fit({'input':x},{'targets':y},n_epoch=5,snapshot_step=500,show_metric=True,run_id='openaistuff')
    return model
training_data=initial_population()
model=train_model(training_data)
#model=train_model(DNN.load("104.model"),)

##
"""for i in range(10):
    x=np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y=np.array([i[1] for i in training_data])
    model=neural_network_model(input_size=len(x[0]))
    model.fit({'input':x},{'targets':y},n_epoch=5,snapshot_step=500,show_metric=True,run_id='openaistuff')
    model=model
"""
##
scores=[]
choices=[]
for each_game in range(50):
    score=0
    game_memory=[]
    prev_obs=[]
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs)==0:
            action=random.randrange(0,2)
        else:
            action=np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
        choices.append(action)

        new_obs,reward,done,info=env.step(action)
        prev_obs=new_obs
        game_memory.append([new_obs,action])
        score+=reward
        if done:
            break
    scores.append(score)
        
print("Average Score:",sum(scores)/len(scores))
print('Choice 1:{} , Choice 2{}:'.format(choices.count(1)/len(choices),choices.count(1)/len(choices)))
