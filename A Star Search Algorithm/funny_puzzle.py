import heapq




def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    n = int(len(from_state)**0.5)
    manhattan_distance = 0
    for i in range(n): # i is the row
        for j in range(n): # j is the column
            tile = from_state[i*n+j]
            if tile == 0:
                continue
            goal_pos = to_state.index(tile)
            goal_i = goal_pos // n
            goal_j = goal_pos % n
            manhattan_distance += abs(goal_i - i) + abs(goal_j - j)
    return manhattan_distance




def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []
    n = int(len(state)**0.5)
    empty_indices = [i for i, tile in enumerate(state) if tile == 0]
    for empty_index in empty_indices:
        i = empty_index // n
        j = empty_index % n

        if i > 0 and state[(i-1)*n+j] != 0: # move a tile down into empty space: there is an empty space not on top row
            successor = state[:]
            successor[empty_index] = successor[(i-1)*n+j] #empty_index-3: the space above
            successor[(i-1)*n+j] = 0
            succ_states.append(successor)
        if i < n-1 and state[(i+1)*n+j] != 0: # move a tile up into empty space: there is an empty space not on bottom row
            successor = state[:]
            successor[empty_index] = successor[(i+1)*n+j] #empty_index+3: the space above
            successor[(i+1)*n+j] = 0
            succ_states.append(successor)
        if j > 0 and state[i*n+(j-1)] != 0: # move a tile right into empty space: there is an empty space not on left
            successor = state[:]
            successor[empty_index] = successor[i*n+(j-1)] # make value of right "0" tile the value of the left one
            successor[i*n+(j-1)] = 0
            succ_states.append(successor)
        if j < n-1 and state[i*n+(j+1)] != 0: # move a tile left into empty space: there is an empty space not on right
                successor = state[:]
                successor[empty_index] = successor[i*n+(j+1)] # make value of left "0" tile the value of the right one
                successor[i*n+(j+1)] = 0
                succ_states.append(successor)
    return sorted(succ_states)


def solve(state, goal_state = [1, 2, 3, 4, 5, 6, 7, 0, 0]):

    # keep track of the g/cost of each state
    g_scores = {tuple(state): 0}
    # keep track of the parent state of each state in the path
    parents = {tuple(state): -1}
    visited = [state]
    path = []
    index = 0
    max_queue_length = 1
    pq = []
    heapq.heappush(pq, (0, state, (g_scores[tuple(state)], get_manhattan_distance(state), -1)))
    
    # Loop until the priority queue is empty or we reach the goal state
    while pq:
        index += 1
        # Pop the state with the lowest f score
        f, current_state, move = heapq.heappop(pq)
        g_scores[tuple(current_state)] = move[0]
        visited.append((current_state,move[2]))

        # Check if we've reached the goal state
        if current_state == goal_state:
            # We've reached the goal state, so print the path and return
            path.append(current_state)
            current_state = visited[move[2]]

            while current_state[1] != -1:
                path.append(current_state[0])
                index = current_state[1]
                current_state = visited[index]
            path.append(current_state[0])
            for i in range(len(path)-1,-1,-1):
                print(path[i], "h={}".format(get_manhattan_distance(path[i])), "moves: {}".format(len(path)-i-1))
            print("Max queue length: {}".format(max_queue_length))
            return
            

        # Generate the successor states of the current state
        successors = get_succ(current_state)
        # Loop through the successor states
        for successor in successors:
            # Calculate the g score of the successor state
            g = g_scores[tuple(current_state)] + 1
           
            # Check if we've already visited the successor state and if this path is better
            if tuple(successor) in g_scores and g >= g_scores[tuple(successor)]:
               continue

            # Update the g score and parent state of the successor state
            parents[tuple(successor)] = index

            # Calculate the f score of the successor state and add it to the priority queue
            f = g + get_manhattan_distance(successor) 
            heapq.heappush(pq, (f, successor, (g, get_manhattan_distance(successor), index)))
            
            max_queue_length = max(max_queue_length, len(pq))

    # If we reach this point, we couldn't find a path to the goal state
    print("No path found.")


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([2,5,1,4,0,6,7,0,3])
    print()
