import numpy as np
import tensorflow as tf
import random
from collections import deque
from dqn4 import dqn4 as dqn
from MySocket import MySocket
  
# ed: 입력값은 센서의 갯수만큼
#input_size = 181+180    #181 + vectorLen + headingDiff
input_size = 36+5*2    #181 + vectorLen + headingDiff

# ed: 출력값은 One hot vector로써 #개의 케이스로 나뉜다
output_size = 3*3

# 감가율
dis = 0.9
# 리플레이메모리 사이즈
REPLAY_MEMORY = 50000
# 최대 스텝개수
MAX_STEP = 900000
# ed: 데이터를 Socket으로부터 받아온다
unityRLdata = MySocket.MySocket()

checkpoint_name = "./test1.ckpt"

#reward
reward_cnt = 5
w0 = 1000         #충돌시
w1 = 100000       #goal도착
w2 = 10           #차와 goal간 방향이 잘 맞는 정도에 비례한 reward
w2Exp = 1       #차수
w3 = 350          #차와 goal간 거리가 가까운 정도에 비례한 reward
w3Exp = 3       #차수
w4 = 0.01      # 오래 살아 남으면 -> ----- ==> 빠르게 도달 할 수 있도록

# 실제로 DQN을 학습하는 함수. targetDQN, mainDQN을 분리시켰다
def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # 게임이 끝난 경우
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]


        # 출력값 Q를 y_stack에 쌓는다
        y_stack = np.vstack([y_stack, Q])
        # 입력값 state를 x_stack에 쌓는다
        x_stack = np.vstack([x_stack, state])
		
    # 쌓인 학습데이터를 한번에 업데이트시킨다
    return mainDQN.update(x_stack, y_stack)


# mainDQN ==> targetDQN 가중치를 복사하는 함수 (복잡해보이지만 그냥 복사하는것)
def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    # Copy variables src_scope to dest_scope
    op_holder = []

    # 모든 훈련가능(TRAINABLE_VARIABLES)한 Weight들을 scope에서 가져온다
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    # src_var.value(가중치) 값들을 dest_var에 assign한다
    # 쉽게 말하면 가중치를 dest_var에 복사한다
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder



def main():
    # 최대 에피소드 개수 
    max_episodes = 5000
    least_loss = 99999999
    max_reward = -1000

    replay_buffer = deque()
    last_100_game_reward = deque()


    with tf.Session() as sess:
	
        # mainDQN, targetDQN을 생성한다
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")

        mainDQN.saver.restore(sess, checkpoint_name)
        print("model restore")

        # tf 변수들을 초기화한다
        tf.global_variables_initializer().run()

        # q_net --> target_net 복사한다
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        
        loss = 0
        mean_reward = 0
        # mainDQN == targetDQN이 같은상태에서 학습을 시작한다 (mainDQN만 학습한다)
        for episode in range(max_episodes):
            #episode가 진행 될 수록 random행동을 덜 한다.
            e = 1. / ((episode / 25) + 1)   # 25라는 수가 커지면 초반에 random을 많이 한다.(max episode에 비례하도록 한다)
            done = False
            step_count = 0

            unityRLdata.sendingMsg(0)
            state, _vecLen, _headingDiff, _done = unityRLdata.getStep()
            #reward 초기화
            reward_w = []
            reward_w = np.zeros(reward_cnt) 
            vecLen_pre = 0
            
            while not done:#1: 충돌, 0: 비충돌
                # epsilon greedy 입실론 탐욕알고리즘을 사용한다
                if np.random.rand(1) < e:
                    action = np.random.randint(output_size)
                    #print("rand")
                else:
                    action = np.argmax(mainDQN.predict(state))
                    #print("Q")

                # Unity로 action 값을 전송한다
                unityRLdata.sendingMsg(action)
                #print("action : ", action)
                
                next_state, vecLen, headingDiff, done = unityRLdata.getStep()
                reward = 1
                #print("--------- vecLen : ", vecLen)
                #print("--------- headingDiff : ", headingDiff)
                #print("--------- done : ", done)
                
                #print(next_state)
                #print("a : ",action, "s : ",next_state, "e : ", done)

                # ed: code added
                #reward -= step_count*w5
                if headingDiff == 0:    #분모가 0이 되는것을 방지
                    headingDiff = 1
                if vecLen == 0: #분모가 0이 되는것을 방지
                    vecLen = 1
                    
                if done == 1:   #충돌
                    reward_w[0] = -w0 #+ w5 * (MAX_STEP - (MAX_STEP / step_count))
                elif done == 3: #도착
                    reward_w[1] = w1*w1
                    print("-------------------------------------")
                    print("!!!!!!!!!!!!!! G O A L !!!!!!!!!!!!!!")
                    print("-------------------------------------")
                else:   #평상시
                    reward_w[0] = 0
                    reward_w[1] = 0
                    #headingDiff -> 0 ~ 100

                    #거리 차이에 의한 reward -> 반비례 관계
                    reward_w[3] = pow(w3 / (vecLen), w3Exp) #1 + (pow((127 - vecLen), w3Exp)       * pow(w3, w3Exp)) #far -> 127 / near(goal) -> 0
                    if w1 < reward_w[3]:    #너무 커지지 않도록
                        reward_w[3] = w1

                    #방향 차이에 의한 reward -> 반비례 관계
                    if reward_w[3] < 0: #뒤로가지 않도록
                        reward_w[2] = reward_w[3] * pow(w2 / (abs(headingDiff)), w2Exp)
                    else:
                        reward_w[2] = reward_w[3] * pow(w2 / (headingDiff), w2Exp)
                    if w1 < reward_w[2]:    #너무 커지지 않도록
                        reward_w[2] = w1
                    #reward_w[4] = w4* step_count

#                if vecLen < vecLen_pre:
#                    reward_w[4] = -reward_w[3] - reward_w[2]
#                else:
#                    reward_w[4] = 0

                for i in range(0, reward_cnt):
                    reward += reward_w[i]   #모든 reward를 합한다
                #    print("[",i ,"] : ", reward_w[i])
                #print(" ")
                #print("reward : ", reward)
                #reward /= step_count#시간이 지날수록 reward를 작게 준다. -> 빠르게 도달 할 수 있도록

                # replay_buffer에 SARS를 저장한다
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > REPLAY_MEMORY:
                      replay_buffer.popleft()
                      
                vecLen_pre = vecLen
                state = next_state
                step_count += 1
				
                #print("reward : ", reward)
                #print("Loss: ", loss)
                #print(" ")

                if max_reward < reward:
                      max_reward = reward
                      filename = mainDQN.saver.save(sess, checkpoint_name)
                      print("Episode: {}, steps: {}, mean reward: {}, max reward: {}".format(episode, step_count, int(mean_reward), int(max_reward)))
#                if loss < least_loss:
#                    least_loss = loss
#                    filename = mainDQN.saver.save(sess, checkpoint_name)
                mean_reward += reward

            mean_reward /= step_count
            #print("Episode: {}, steps: {}, mean reward: {}, max reward: {}".format(episode, step_count, int(mean_reward), int(max_reward)))
            
            # 에피소드가 4번될 때마다 1번씩 
            if episode % 4 == 1 and episode > 10:
                # 100번정도 돌면서 replay_buffer에서 10개의 데이터를 가져와서 학습한다
                for _ in range(100):
                    minibatch = random.sample(replay_buffer, 50)
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)

                #loss 낮을 때, reward가 가장 높을 때, step이 가장 많을 때(멀리까지 갔을 때),
                #if loss < least_loss:
                #    least_loss = loss
                #    filename = mainDQN.saver.save(sess, checkpoint_name)
                #print("Loss: ", loss)

                # 특정 주기로만 mainDQN --> targetDQN으로 가중치를 복사한다
                sess.run(copy_ops)
            
#            last_100_game_reward.append(reward)

#            if len(last_100_game_reward) > 100:
#                last_100_game_reward.popleft()

#                avg_reward = np.mean(last_100_game_reward)

#                if avg_reward > MAX_STEP * 0.97:
#                    print("Game Cleared")
#                    break


if __name__ == "__main__":
    main()
