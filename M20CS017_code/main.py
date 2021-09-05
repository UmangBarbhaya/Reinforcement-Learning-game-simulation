import numpy as np
import pygame
import time
import random

pygame.init()
rows, cols, iteration, cost, actions = 8, 8 ,1, 0, ['up', 'right', 'down', 'left']
q_values = np.zeros((rows, cols, 4))
stateReward = np.full((rows, cols), -0.4)

stateReward[0, 2], stateReward[1, 2], stateReward[2, 2], stateReward[3, 2], stateReward[1, 4], stateReward[1, 5], stateReward[3, 4], stateReward[3, 5],stateReward[3, 6], stateReward[4, 4], stateReward[5, 1], stateReward[5, 2], stateReward[5, 4], stateReward[6, 4], stateReward[6, 6],stateReward[6, 7] = -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999


stateReward[4, 0], stateReward[4, 6] = -1 , -1
stateReward[1, 3], stateReward[2, 1], stateReward[7, 7]= 1, 1, 1

screen = pygame.display.set_mode((512, 512))

pygame.display.set_caption("M20CS017_Reinforcement Learning")

agent, agentx, agenty = pygame.image.load('Agent.jpg'), 0, 7*64
goal, goalx, goaly = pygame.image.load('Goal.jpg'), random.randint(5, 7) * 64, random.randint(0, 2) * 64

while goalx == 5 * 64 and goaly == 1 * 64:
    goalx, goaly = random.randint(5, 7) * 64, random.randint(0, 2) * 64

background = pygame.image.load('Environment.jpg')

stateReward[int(goaly / 64), int(goalx / 64)] = 100
print("Real World:\n", stateReward)


def get_shortest_path(start_row_index, start_column_index, iteration):
    x = 0
    if stateReward[start_row_index, start_column_index] != -0.4 and stateReward[
        start_row_index, start_column_index] != -1:
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
        global cost

        while (stateReward[current_row_index, current_column_index] == -0.4 or stateReward[
        current_row_index, current_column_index] == -1) and x <= iteration:

            if np.random.random() < 1.:
                action_index = np.argmax(q_values[current_row_index, current_column_index])
            else:
                action_index = np.random.randint(4)

            new_row_index = current_row_index
            new_column_index = current_column_index
            if actions[action_index] == 'up' and current_row_index > 0:
                new_row_index =new_row_index- 1
                cost += q_values[current_row_index, current_column_index,action_index]
            elif actions[action_index] == 'right' and current_column_index < cols - 1:
                new_column_index =new_column_index + 1
                cost += q_values[current_row_index, current_column_index, action_index]
            elif actions[action_index] == 'down' and current_row_index < rows - 1:
                new_row_index =new_row_index + 1
                cost += q_values[current_row_index, current_column_index, action_index]
            elif actions[action_index] == 'left' and current_column_index > 0:
                new_column_index =new_column_index- 1
                cost += q_values[current_row_index, current_column_index, action_index]
            current_row_index = new_row_index
            x = x + 1
            current_column_index = new_column_index
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path


def train(start_row, start_col):
    for episode in range(1000):
        row_index = start_row
        column_index =  start_col
        while stateReward[row_index, column_index] == -0.4 or stateReward[row_index, column_index] == -1:
            if np.random.random() >= 0.9:
                action_index = np.random.randint(4)
            else:
                action_index = np.argmax(q_values[row_index, column_index])

            old_row_index = row_index
            old_column_index = column_index
            new_row_index = row_index
            new_column_index = column_index
            if actions[action_index] == 'up' and row_index > 0:
                new_row_index = new_row_index - 1
            elif actions[action_index] == 'right' and column_index < cols - 1:
                new_column_index = new_column_index + 1
            elif actions[action_index] == 'down' and row_index < rows - 1:
                new_row_index = new_row_index + 1
            elif actions[action_index] == 'left' and column_index > 0:
                new_column_index = new_column_index - 1
            row_index = new_row_index
            column_index =  new_column_index

            temporal_difference = stateReward[row_index, column_index] + (0.9 * np.max(q_values[row_index, column_index])) - q_values[old_row_index, old_column_index, action_index]
            q_values[old_row_index, old_column_index, action_index] = q_values[old_row_index, old_column_index, action_index] + (0.9 * temporal_difference)

print("Starting the Training")
train(7, 0)
print("Getting the path")
output1,iteration = get_shortest_path(7, 0, iteration),2
output2,iteration = get_shortest_path(7, 0, iteration),3
output3,iteration = get_shortest_path(7, 0, iteration), 500
cost = 0
output4, iteration = get_shortest_path(7, 0, iteration),500

train(3, 7)
output5, iteration = get_shortest_path(3, 7, iteration),1
for g in range(len(output5)):
    output4.append(output5[g])

print("ii. Final Path Cost: ", cost)
print("iii. Path of Iteration-1:", output1)
print("iii. Path of Iteration-2:", output2)
print("iii. Path Iteration-3:", output3)
print("iii. Path of Final Iteration:", output4)

print("iv. Knowledge Base: \n", q_values)


white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
font = pygame.font.Font('freesansbold.ttf', 17)
outputstring = "Iteration-1"
text = font.render(outputstring, True, green, blue)
textRect = text.get_rect()
textRect.center = (64,32)

outputstring = "Iteration-2"
text2 = font.render(outputstring, True, green, blue)
textRect2 = text2.get_rect()
textRect2.center = (64,32)

outputstring = "Iteration-3"
text3 = font.render(outputstring, True, green, blue)
textRect3 = text3.get_rect()
textRect3.center = (64,32)

outputstring = "Final Iteration"
text4 = font.render(outputstring, True, green, blue)
textRect4 = text4.get_rect()
textRect4.center = (64,32)

i = 0
i2 = 0
i3 = 0
i4 = 0
case = 1

running = True



while running:
    # RGB background

    screen.fill((0, 128, 0))
    screen.blit(background, (0, 0))
    screen.blit(goal, (goalx, goaly))
    screen.blit(agent, (agentx, agenty))
    time.sleep(1)
    # iteration 1
    if i < len(output1):
        agenty, agentx = output1[i][0] * 64, output1[i][1] * 64
        screen.blit(text, textRect)
        i = i + 1
    else:
        if i2 < len(output2):
            agenty,agentx = output2[i2][0] * 64, output2[i2][1] * 64
            screen.blit(text2, textRect2)
            i2 = i2 + 1
        else:
            if i3 < len(output3):
                agenty, agentx = output3[i3][0] * 64, output3[i3][1] * 64
                screen.blit(text3, textRect3)
                i3 = i3 + 1
            else:
                if i4 < len(output4):
                    agenty, agentx = output4[i4][0] * 64, output4[i4][1] * 64
                    screen.blit(text4, textRect4)
                    i4 = i4 + 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()


