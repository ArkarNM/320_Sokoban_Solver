
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

# helper for corners (x, y)
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

flip_tuple = lambda tup : (tup[1], tup[0])
flip_tuple.__doc__ = """flips the tuple from x,y to row, col (y, x)"""

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

matrix_to_string = lambda warehouseMatrix : NEW_LINE.join([EMPTY_STRING.join(row) for row in warehouseMatrix])
matrix_to_string.__doc__ = """converts a 2D array of chars to a string"""

string_to_matrix = lambda warehouseStr : [list(line) for line in warehouseStr.split(NEW_LINE)]
string_to_matrix.__doc__ = """converts a string to a 2D array of chars"""

def manhattan_distance(init, end):
        """
        manhattan distance |x2 - x1| + |y2 - y1|
        """
        (i_x, i_y) = init
        (e_x, e_y) = end
        return abs(e_x - i_x) + abs(e_y - i_y)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Helper Classes ##

class PathProblem(search.Problem):
    # initialises the problem
    def __init__(self, initial, warehouse, goal):
        self.initial = initial
        self.warehouse = warehouse
        self.goal = goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

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
    result = lambda self, state, action : add_action(state, action)

    h = lambda self, n : manhattan_distance(self.goal, n.state)
    h.__doc__ = """heuristic using manhattan distance for a* graph search |x2 - x1| + |y2 - y1|"""

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

class SokobanPuzzleState(sokoban.Warehouse):

    def __init__(self, warehouse):
        self.worker = warehouse.worker
        self.boxes = warehouse.boxes
        self.targets = warehouse.targets
        self.walls = warehouse.walls
        self.ncols = warehouse.ncols
        self.nrows = warehouse.nrows

    __lt__ = lambda self, a : (self.worker, self.boxes) > (a.worker, a.boxes)
    __lt__.__doc__ = """ """


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
    
    def __init__(self, warehouse, macro=False, allow_taboo_push=False):
        """
        initialisation function
        """
        self.initial = SokobanPuzzleState(warehouse)
        self.macro = macro
        self.allow_taboo_push = allow_taboo_push
        # get a list of taboo_cells for usage
        self.taboo_cells = set(sokoban.find_2D_iterator(taboo_cells(warehouse).split(sep='\n'), "X"))
        # remove the player from the goal or target_square and move the boxes to the targets
        self.goal = self.initial.__str__().replace(PLAYER, SPACE).replace(PLAYER_ON_TARGET_SQUARE, BOX_ON_TARGET).replace(BOX, SPACE).replace(TARGET_SQUARE, BOX_ON_TARGET)

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        walls, worker, boxes = state.walls, state.worker, state.boxes

        if self.macro:
            # macro actions
            # go through boxes and determine what worker can do to them
            for box in boxes:
                # enumerate through possible surroundings
                for i, surr in enumerate(SURROUNDINGS):
                    # new position of the box when pushed
                    test_pos = add_action(box, surr)
                    # if we can't go there then it's not a valid move
                    if can_go_there(state, flip_tuple(test_pos)) or worker == test_pos:
                        # new position of the box when pushed, opposition direction of current surrounding
                        new_box_pos = add_action(box, surr, -1)
                        if new_box_pos not in boxes and new_box_pos not in walls:
                         # if allow taboo push, yield action or if test box not in taboo_cells
                            if self.allow_taboo_push or new_box_pos not in self.taboo_cells:
                                # get the opposite of the current action as in worker goes 'Left' but pushes box 'Right'
                                yield flip_tuple(box), ACTIONS[(i+2) % 4]                    
        else:
            # elementary actions
            # enumerate through possible surroundings
            for i, surr in enumerate(SURROUNDINGS):
                # get the new position of adding the move to the worker
                test_pos = add_action(worker, surr)
                # test it's not a wall
                if test_pos not in walls:
                    # if it's within a box position test new position of box
                    if test_pos in boxes:
                        test_box = add_action(worker, surr, 2)
                        # ensure the new box position doesn't merge with a wall, box
                        if test_box not in boxes and test_box not in walls:
                            # if allow taboo push, yield action or if not allowing, test box not in taboo_cells
                            if self.allow_taboo_push or test_box not in self.taboo_cells:
                                yield ACTIONS[i]
                    else:
                        yield ACTIONS[i]

    path_cost = lambda self, c, state1, action, state2 : c + 1
    path_cost.__doc__ = """Return the cost of a solution path that arrives at state2 from
                            state1 via action, assuming cost c to get up to state1. If the problem
                            is such that the path doesn't matter, this function will only look at
                            state2.  If the path does matter, it will consider c and maybe state1
                            and action. The default method costs 1 for every step in the path."""

    goal_test = lambda self, state : state.__str__().replace("@", " ") == self.goal
    goal_test.__doc__ = """goal test to ensure all boxes are in a target_square, player position is irrelevant so remove"""

    def result(self, state, action):
        """
        action upon the given action and return the new state
        """
        worker, boxes = state.worker, state.boxes

        if self.macro:
            # convert action ie 'Left' into tuple (-1, 0)
            next_pos = SURROUNDINGS[ACTIONS.index(action[1])]
            # get the new worker position, flip the action because it's row, col (y, x) not x, y
            new_worker = flip_tuple(action[0])
        else:
            # convert action ie 'Left' into tuple (-1, 0)
            next_pos = SURROUNDINGS[ACTIONS.index(action)]
            # get the new worker position
            new_worker = add_action(worker, next_pos)

        # copy the state and move the worker to the next position
        # for any box in the position of the new worker position,
        # push it twice the current position of the worker to allow the worker to move forward
        # if the box isn't in the resultant position return the same position of the box    
        return SokobanPuzzleState(state.copy(
            worker = new_worker, 
            boxes = [add_action(box_pos, next_pos) 
                    if box_pos == new_worker
                    else box_pos 
                    for box_pos in boxes]))

    def h(self, n):
        """
        heuristic using manhattan distance for a* graph search |x2 - x1| + |y2 - y1|
        """
        # initialise new warehouse to work on and get new tuples
        current_warehouse = n.state
        
        worker, boxes, targets = current_warehouse.worker, current_warehouse.boxes, current_warehouse.targets

        worker_to_box_distances, box_to_target_totals = list(), list()
        
        # iterate through boxes to find the distance for each from worker
        for box in boxes:
            worker_to_box_distances.append(manhattan_distance(worker, box))

        # iterate through each perm of targets to find the distance between each box
        for targets_perm in itertools.permutations(targets):
            total_distance = 0
            # combines targets and boxes in tuples as in (target, box) 
            zipped_tuples = zip(targets_perm, boxes)
            # for each target and box get the manhattan distance for each and add that to a total 
            # so we have the total distance of all boxes to targets in this permuation
            for target, box in zipped_tuples:
                total_distance += manhattan_distance(target, box)
            box_to_target_totals.append(total_distance)

        # return the smallest worker to box distance and smallest box to target total distance
        return min(worker_to_box_distances) + min(box_to_target_totals)


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
    wh = SokobanPuzzleState(warehouse)
    puzzle = SokobanPuzzle(warehouse)

    Failed = 'Impossible'

    # iterates over the actions
    for action in action_seq:

        # get the original location of walls and boxes
        walls, boxes = wh.walls, wh.boxes

        # employs the actions and returns the resultant string
        # we can use the result() to get the acted up result of each action
        wh=puzzle.result(wh, action)

        # get the worker from the new result
        worker = wh.worker

        # iterates over walls and ensures the worker hasn't clipped a wall
        for wall in walls:
            if worker == wall:
                return Failed

        # helper to ensure boxes aren't stacked
        box_stack = set()

        # iterates over boxes
        for box in boxes:
            # ensures no boxes clip one another
            if box in box_stack:
                return Failed
            # adds the box to set for next box test
            else:
                box_stack.add(box)
            # iterates over walls and ensures no boxes have clipped any walls
            for wall in walls:
                if box == wall:
                    return Failed

    return wh.__str__()

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

    path = search.astar_graph_search(SokobanPuzzle(warehouse))

    if path is not None:
        return path.solution()
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

    # check if a valid path from the worker to the coordinate provided exists
    path = search.astar_graph_search(PathProblem(warehouse.worker, warehouse, (col, row)))

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

    path = search.astar_graph_search(SokobanPuzzle(warehouse, True))

    if path is not None:
        return path.solution()
    else: 
        return 'Impossible'

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

