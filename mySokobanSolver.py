
'''

    2020 CAB320 Sokoban assignment


The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.
No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.


You are NOT allowed to change the defined interfaces.
That is, changing the formal parameters of a function will break the 
interface and results in a fail for the test of your code.
This is not negotiable! 


'''

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
import search 
import sokoban
import itertools

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Global Variables ##

SPACE = ' '
WALL = '#'
BOX = '$'
TARGET_SQUARE = '.'
PLAYER = '@'
PLAYER_ON_TARGET_SQUARE = '!'
BOX_ON_TARGET = '*'
TABOO = 'X'
NEW_LINE = '\n'
EMPTY_STRING = ''

# different types of target squares
TARGETS = [TARGET_SQUARE, PLAYER_ON_TARGET_SQUARE, BOX_ON_TARGET]

# helper for corners
SURROUNDINGS = [(0, -1), (-1, 0), (0, 1), (1, 0)]
ACTIONS = ['Up', 'Left', 'Down', 'Right']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Helper Functions ##

def add_action(state, action, scale=1):
    """
    adds the action tuple to the state tuple and returns
    """
    (s_x, s_y) = state
    (a_x, a_y) = action
    return s_x + (scale * a_x), s_y + (scale * a_y)


def check_if_corner_cell(warehouseMatrix, dst):
    """
    checks the warehouse and determines if the cell is surrounded by a corner
    """
    (row, col) = dst
    for i in range(len(SURROUNDINGS)):
        (a_row, a_col) = SURROUNDINGS[i]
        (b_row, b_col) = SURROUNDINGS[(i+1) % 4]

        # if both are walls, as in is a corner, then return True
        if warehouseMatrix[row + a_row][col + a_col] is WALL and warehouseMatrix[row + b_row][col + b_col] is WALL:
            return True
    return False


def check_if_along_wall(warehouseMatrix, dst):
    """
    checks the warehouse and determines if the cell is along a wall
    """
    (row, col) = dst
    for (a_row, a_col) in SURROUNDINGS:
        # if next to wall then return True
        if warehouseMatrix[row + a_row][col + a_col] is WALL:
            return True
    return False


def matrix_to_string(warehouseMatrix):
    """
    converts a 2D array of chars to a string
    """
    return NEW_LINE.join([EMPTY_STRING.join(row) for row in warehouseMatrix])

def string_to_matrix(warehouseStr):
    return [list(line) for line in warehouseStr.split(NEW_LINE)]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Helper Classes ##

class PathProblem(search.Problem):
    # initialises the problem
    def __init__(self, initial, warehouse, goal):
        self.initial = initial
        self.warehouse = warehouse
        self.goal = goal

    # signifies the cost of the movement
    def value(self, state):
        return 1

    # list of possible actions
    def actions(self, state):
        boxes = self.warehouse.boxes
        walls = self.warehouse.walls

        for action in SURROUNDINGS:
            new_state = add_action(state, action)
            # check that the action doesn't result in a wall or box collision
            if new_state not in boxes and new_state not in walls:
                yield action

    # Return the old state, with the action applied.
    def result(self, state, action):
        return add_action(state, action)

class Heuristic():
    def __init__(self, col, row):
        self.col = col
        self.row = row

    def manhattan_distance(self, n):
        """
        heuristic using manhattan distance for a* graph search |x2 - x1| + |y2 - y1|
        """
        (s_row, s_col) = n.state
        return abs(s_row - self.row) + abs(s_col - self.col)
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    '''
    return [ (10212361, 'Jamie', 'Martin'), (9737197, 'Tolga', 'Pasin'), (000000, 'xxxx', 'xxxx') ]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A cell inside a warehouse is 
    called 'taboo'  if whenever a box get pushed on such a cell then the puzzle 
    becomes unsolvable. Cells outside the warehouse should not be tagged as taboo.
    When determining the taboo cells, you must ignore all the existing boxes, 
    only consider the walls and the target  cells.  
    Use only the following two rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of 
             these cells is a target.
    
    @param warehouse: 
        a Warehouse object with a worker inside the warehouse

    @return
       A string representing the puzzle with only the wall cells marked with 
       a '#' and the taboo cells marked with a 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''

    # get string
    warehouseStr = warehouse.__str__()

    irrelevantSquares = [BOX, PLAYER]

    for square in irrelevantSquares:
        warehouseStr = warehouseStr.replace(square, SPACE)

    # convert warehouse string into Array<Array<char>>
    warehouseMatrix = string_to_matrix(warehouseStr)

    # ignore boxes for can_go_there method
    warehouse.boxes = []

    ''' old method '''
    # inside = False

    # find the inside of the playing area
    # if not inside and cell is WALL:
    #     inside = True
    # elif inside:
        
    #     # if rest of row is outside of playing area then break the loop
    #     if all([cell is SPACE for cell in row[x:]]):
    #         break
    ''' end old method '''

    # rule 1: if a cell is a corner and not a target, then it is a taboo cell.
    for row_index, row in enumerate(warehouseMatrix):
        for col_index, cell in enumerate(row):
            if cell is not WALL and cell not in TARGETS:
                if can_go_there(warehouse, (row_index, col_index)) or (row_index, col_index) == warehouse.worker:
                    # find corners to set as taboo, breaks when found
                    if check_if_corner_cell(warehouseMatrix, (row_index, col_index)):
                        warehouseMatrix[row_index][col_index] = TABOO

    # rule 2: all the cells between two corners along a wall are taboo if none of these cells is a target.
    for row_index, row in enumerate(warehouseMatrix):
        for col_index, cell in enumerate(row):
            # find a taboo cell and check rows and columns that apply to rule 2
            if cell is TABOO and check_if_corner_cell(warehouseMatrix, (row_index, col_index)):
                # from the taboo point get the rest of the row to the right of it and enumerate
                #row_x
                for taboo_col_index, test_cell in enumerate(warehouseMatrix[row_index][col_index + 1:]):
                    # if there's any targets or walls break
                    if test_cell in TARGETS or test_cell is WALL:
                        break

                    # this is the next point in the row, we use x because the rest of the row may be cut off to enumerate
                    next_taboo_col_index = col_index + (taboo_col_index + 1)

                    # find another taboo cell or corner
                    if test_cell is TABOO and check_if_corner_cell(warehouseMatrix, (row_index, next_taboo_col_index)):
                        # if the entire row is along a wall then the entire row is taboo
                        if all([check_if_along_wall(warehouseMatrix, (row_index, i)) for i in range(col_index + 1, next_taboo_col_index)]):
                            # fill with taboo
                            for taboo_index in range(col_index + 1, next_taboo_col_index):
                                warehouseMatrix[row_index][taboo_index] = TABOO

                # from the taboo point get the rest of the column below it and enumerate over
                for taboo_row_index, test_cell in enumerate([row[col_index] for row in warehouseMatrix[row_index + 1:][:]]):
                    # if there's any targets or walls break
                    if test_cell in TARGETS or test_cell is WALL:
                        break

                    # this is the next point in the column, we use x because the rest of the col may be cut off to enumerate
                    next_taboo_row_index = row_index + (taboo_row_index + 1)
                    
                    # find another taboo cell or corner
                    if test_cell is TABOO and check_if_corner_cell(warehouseMatrix, (next_taboo_row_index, col_index)):
                        # if the entire column is along a wall then the entire column is taboo
                        if all([check_if_along_wall(warehouseMatrix, (i, col_index)) for i in range(row_index + 1, next_taboo_row_index)]):
                            # fill with taboo
                            for taboo_index in range(row_index + 1, next_taboo_row_index):
                                warehouseMatrix[taboo_index][col_index] = TABOO

    # return to string variable
    warehouseStr = matrix_to_string(warehouseMatrix)

    # remove target chars
    for square in TARGETS:
        warehouseStr = warehouseStr.replace(square, SPACE)

    return warehouseStr

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 
    
    Each SokobanPuzzle instance should have at least the following attributes
    - self.allow_taboo_push
    - self.macro
    
    When self.allow_taboo_push is set to True, the 'actions' function should 
    return all possible legal moves including those that move a box on a taboo 
    cell. If self.allow_taboo_push is set to False, those moves should not be
    included in the returned list of actions.
    
    If self.macro is set True, the 'actions' function should return 
    macro actions. If self.macro is set False, the 'actions' function should 
    return elementary actions.        
    '''
    
    def __init__(self, warehouse):
        """
        initialisation function
        """
        self.initial = warehouse.copy()
        # get a list of taboo_cells for usage
        self.taboo_cells = set(sokoban.find_2D_iterator(taboo_cells(warehouse).split(sep='\n'), "X"))
        # remove the player from the goal or target_square and move the boxes to the targets
        self.goal = warehouse.__str__().replace(PLAYER, SPACE).replace(PLAYER_ON_TARGET_SQUARE, BOX_ON_TARGET).replace(BOX, SPACE).replace(TARGET_SQUARE, BOX_ON_TARGET)

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        worker = state.worker
        walls = state.walls
        boxes = state.boxes

        # enumerate through possible surroundings
        for i, surr in enumerate(SURROUNDINGS):
            # get the new position of adding the move to the worker
            test_pos = add_action(worker, surr)
            # test it's not a wall
            if test_pos not in walls:
                # if it's within a box position test new position of box
                if test_pos in boxes:
                    test_box = add_action(worker, surr, 2)
                    # ensure the new box position doesn't merge with a wall, box or taboo cell
                    if test_box not in self.taboo_cells and test_box not in boxes and test_box not in walls:
                        yield ACTIONS[i]
                else:
                    yield ACTIONS[i]        

    def goal_test(self, state):
        # goal test to ensure all boxes are in a target_square
        # player position is irrelevant so remove
        return state.__str__().replace("@", " ") == self.goal

    def result(self, state, action):
        """
        action upon the given action and return the new state
        """
        # convert action ie 'Left' into tuple (-1, 0)
        next_pos = SURROUNDINGS[ACTIONS.index(action)]
        worker = state.worker
        boxes = state.boxes

        new_worker = add_action(worker, next_pos)

        # copy the state and move the worker to the next position
        # for any box in the position of the new worker position,
        # push it twice the current position of the worker to allow the worker to move forward
        # if the box isn't in the resultant position return the same position of the box       
        return state.copy(
            worker = new_worker, 
            boxes = [add_action(worker, next_pos, 2) 
                    if box_pos == new_worker
                    else box_pos 
                    for box_pos in boxes])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Impossible', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    # copies warehouse into a new Sokoban puzzle
    puzzle = SokobanPuzzle(warehouse.copy())

    Failed = 'Impossible'

    # iterates over the actions
    for action in action_seq:
        # employs the actions and returns the resultant
        # we can use the result() to get the acted up result of each action
        warehouse=puzzle.result(warehouse, action)
        walls = warehouse.walls
        worker = warehouse.worker
        boxes = warehouse.boxes

        # iterates over walls and ensures the worker hasn't clipped a wall
        for wall in walls:
            if worker == wall:
                return Failed

        # helper to ensure boxes aren't stacked
        box_stack = set()

        # iterates over boxes
        for box in boxes:
            # iterates over walls and ensures no boxes have clipped any walls
            for wall in walls:
                if wall == box:
                    return Failed

            # ensures no boxes clip one another
            if box in box_stack:
                return Failed
            # adds the box to set for next box test
            else:
                box_stack.add(box)

    return warehouse.__str__()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_elem(warehouse):
    '''    
    This function should solve using A* algorithm and elementary actions
    the puzzle defined in the parameter 'warehouse'.
    
    In this scenario, the cost of all (elementary) actions is one unit.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''

    ######################## NEED TO CHANGE OR REFORMAT BELOW ##########################
    def heuristic(n):
        """
        heuristic using manhattan distance for a* graph search |x2 - x1| + |y2 - y1|
        """

        for targets_perm in itertools.permutations(n.state.targets):
            abs_value = 0
            for boxes_perm in itertools.permutations(n.state.boxes):
                print(targets_perm, boxes_perm)
        print('\n')
        # create every possible permutation of targets
        permu_targets = list(itertools.permutations(n.state.targets))
        heuristic_list = []
        # loop through every permutation
        for i in range(len(permu_targets)):
             # zip permutation and boxes for comparison
             zipped = list(zip(n.state.boxes, permu_targets[i]))
             total_abs_value = 0
             # loop through lists and find total absolute distance from targets to boxes
             for j in range(len(zipped)):
                 abs_value = abs(zipped[j][0][0]-zipped[j][1][0]) + abs(zipped[j][0][1]-zipped[j][1][1])
                 total_abs_value = total_abs_value + abs_value
             heuristic_list.append(total_abs_value)   
        
        # take the minimum absolute distance of boxes to targets
        boxtotarget_distance = min(heuristic_list)
        
        heuristic_list = []
        # store the absolute distance between the worker and the closest goal in a list
        for (x,y) in n.state.boxes:
            abs_distance = abs(x-n.state.worker[0])+abs(y-n.state.worker[1])
            heuristic_list.append(abs_distance)
        #get the minimum distance between worker and closest goal    
        workertobox_distance = min(heuristic_list)

        return boxtotarget_distance + workertobox_distance
    ######################## NEED TO CHANGE OR REFORMAT ABOVE ##########################


    path = search.astar_graph_search(SokobanPuzzle(warehouse), heuristic)

    def getPath(path):
        for node in path.path():
            if node.action is not None:
                if isinstance(node.action, list):
                    for action in node.action:
                        yield action
                else:
                    yield node.action
                    
    if path is not None:
        return [action for action in getPath(path)]
    else: 
        return 'Impossible'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def can_go_there(warehouse, dst):
    '''    
    Determine whether the worker can walk to the cell dst=(row,column) 
    without pushing any box.
    
    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    '''
    # separate row, col for usage below
    (row, col) = dst

    # the player is only able to move to a space and a target square
    ALLOWED_CELLS = [SPACE, TARGET_SQUARE] 

    # convert the warehouse to a Array<Array<char>>
    warehouseMatrix = string_to_matrix(warehouse.__str__())

    # check if the worker is allowed onto the given coordinates before checking if a valid path exists
    cell = warehouseMatrix[row][col]
    if cell not in ALLOWED_CELLS:
        return False
  
    h = Heuristic(col, row)

    # check if a valid path from the worker to the coordinate provided exists
    path = search.astar_graph_search(
                PathProblem(
                        warehouse.worker, 
                        warehouse, 
                        (col, row)),
                        h.manhattan_distance)

    return path is not None
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_macro(warehouse):
    '''    
    Solve using using A* algorithm and macro actions the puzzle defined in 
    the parameter 'warehouse'. 
    
    A sequence of macro actions should be 
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ] 
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes to the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.
    
    In this scenario, the cost of all (macro) actions is one unit. 

    @param warehouse: a valid Warehouse object

    @return
        If the puzzle cannot be solved return the string 'Impossible'
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    '''
    
    ##         "INSERT YOUR CODE HERE"
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban_elem(warehouse, push_costs):
    '''
    In this scenario, we assign a pushing cost to each box, whereas for the
    functions 'solve_sokoban_elem' and 'solve_sokoban_macro', we were 
    simply counting the number of actions (either elementary or macro) executed.
    
    When the worker is moving without pushing a box, we incur a
    cost of one unit per step. Pushing the ith box to an adjacent cell 
    now costs 'push_costs[i]'.
    
    The ith box is initially at position 'warehouse.boxes[i]'.
        
    This function should solve using A* algorithm and elementary actions
    the puzzle 'warehouse' while minimizing the total cost described above.
    
    @param 
     warehouse: a valid Warehouse object
     push_costs: list of the weights of the boxes (pushing cost)

    @return
        If puzzle cannot be solved return 'Impossible'
        If a solution exists, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''
    
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

