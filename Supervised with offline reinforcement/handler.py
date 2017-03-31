from utility import *
from MDP import *

"""Event handler"""
def collision(x1,y1,x2,y2):
    if (x1==x2) and (y1==y2):
        return True
    return False

def handle_event(event,pg,x,y):
    if event.type == pg.KEYDOWN:
        if event.key == pg.K_LEFT:
            x = maximum(x-1,0)
        if event.key == pg.K_RIGHT:
            x = minimum(x+1,3)
    return (x,y)


learn_network()
"""Action corresponding to ANN output"""
def handle_event_agent(ball_type,x,y,b_x):
    #simulate three action and take the one with maximum Q value
    curr_state = [ball_type,x,y,b_x]
    max_action = 0
    max_q = 0
    for action in range(-1,2):
        state_reached = state_transition(curr_state,action)
        q = 0
        for new_state in state_reached:
            q = q + transition_probability(curr_state,action,new_state)*get_state_value(new_state)
        if max_q < q:
            max_q = q
            max_action = action
    return (clamp(b_x + int(max_action),0,3),3)
##def handle_event_agent(ball_type,x,y,b_x):
##    return (clamp(b_x + int(action_output[ball_type][x][y][b_x]),0,3),3)
        
