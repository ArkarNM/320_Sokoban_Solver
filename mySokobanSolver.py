
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
import math
import time
import timeit

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# global variables
SPACE = ' '
WALL = '#'
BOX = '$'
TARGET_SQUARE = '.'
PLAYER = '@'
PLAYER_ON_TARGET_SQUARE = '!'
BOX_ON_TARGET = '*'
TABOO = 'X'
# helper for corners
SURROUNDINGS = [(0, -1), (-1, 0), (0, 1), (1, 0)]

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
    
    SYMBOLS_TO_REMOVE = ['$', '@']

    TARGETS = [TARGET_SQUARE, PLAYER_ON_TARGET_SQUARE, BOX_ON_TARGET]

    def is_corner_cell(warehouse2D, x, y):
        for i, _ in enumerate(SURROUNDINGS):
            (ax, ay) = SURROUNDINGS[i]
            (bx, by) = SURROUNDINGS[(i+1) % 4]
            # if both are walls, as in is a corner, then return True
            if warehouse2D[y + ay][x + ax] is WALL and warehouse2D[y + by][x + bx] is WALL:
                return True
        return False

    def is_along_wall(warehouse2D, x, y):
        for i, (ax, ay) in enumerate(SURROUNDINGS):
            # if next to wall then return True
            if warehouse2D[y + ay][x + ax] is WALL:
                return True
        return False

    
    # get string
    warehouseStr = warehouse.__str__()
    
    '''for using can_go_there() method'''
    # whStr = warehouseStr
    # wh = sokoban.Warehouse()
    # for char in [BOX, BOX_ON_TARGET, TARGET_SQUARE]:
    #     whStr = whStr.replace(char, SPACE)
    # wh.from_lines(whStr.split(sep='\n'))
    '''end for using can_go_there() method'''

    # remove unneccessary things
    for char in SYMBOLS_TO_REMOVE:
        warehouseStr = warehouseStr.replace(char, SPACE)

    # convert warehouse string into Array<Array<char>>
    warehouse2D = [list(line) for line in warehouseStr.split('\n')]

    # rule 1: if a cell is a corner and not a target, then it is a taboo cell.
    for y, row in enumerate(warehouse2D):
        ''' old method '''
        inside = False
        ''' end old method '''
        for x, cell in enumerate(row):

            ''' can_go_there() method '''
            # print(can_go_there(wh, (y, x)), wh.worker, y, x, cell)
            # if can_go_there(wh, (y, x)) or (y, x) == wh.worker:
            '''  can_go_there() method '''

            ''' old method '''
            # find the inside of the playing area
            if not inside and cell is WALL:
                inside = True
            elif inside:
                ''' end old method '''

                # if rest of row is outside of playing area then break the loop
                if all([cell is SPACE for cell in row[x:]]):
                    break
                elif cell is not WALL and cell not in TARGETS:
                    # find corners to set as taboo, breaks when found
                    if is_corner_cell(warehouse2D, x, y):
                        warehouse2D[y][x] = TABOO

    # rule 2: all the cells between two corners along a wall are taboo if none of these cells is a target.
    for y, row in enumerate(warehouse2D):
        for x, cell in enumerate(row):
            # find a taboo cell and check rows and columns that apply to rule 2
            if cell is TABOO and is_corner_cell(warehouse2D, x, y):
                # from the taboo point get the rest of the row to the right of it and enumerate
                for row_x, row_cell in enumerate(warehouse2D[y][x + 1:]):
                    # if there's any targets or walls break
                    if row_cell in TARGETS or row_cell is WALL:
                        break

                    # this is the next point in the row, we use x because the rest of the row may be cut off to enumerate
                    next_in_row_from_taboo = x + (row_x + 1)

                    # find another taboo cell or corner
                    if row_cell is TABOO and is_corner_cell(warehouse2D, next_in_row_from_taboo, y):
                        # if the entire row is along a wall then the entire row is taboo
                        if all([is_along_wall(warehouse2D, i, y) for i in range(x + 1, next_in_row_from_taboo)]):
                            # fill with taboo
                            for x4 in range(x + 1, next_in_row_from_taboo):
                                warehouse2D[y][x4] = TABOO

                # from the taboo point get the rest of the column below it and enumerate over
                for col_y, col_cell in enumerate([row[x] for row in warehouse2D[y + 1:][:]]):
                    # if there's any targets or walls break
                    if col_cell in TARGETS or col_cell is WALL:
                        break

                    # this is the next point in the column, we use x because the rest of the col may be cut off to enumerate
                    next_in_col_from_taboo = y + (col_y + 1)
                    
                    # find another taboo cell or corner
                    if col_cell is TABOO and is_corner_cell(warehouse2D, x, next_in_col_from_taboo):
                        # if the entire column is along a wall then the entire column is taboo
                        if all([is_along_wall(warehouse2D, x, i) for i in range(y + 1, next_in_col_from_taboo)]):
                            # fill with taboo
                            for y4 in range(y + 1, next_in_col_from_taboo):
                                warehouse2D[y4][x] = TABOO


    # return to string variable
    warehouseStr = '\n'.join([''.join(line) for line in warehouse2D])

    # remove target chars
    for char in TARGETS:
        warehouseStr = warehouseStr.replace(char, SPACE)

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
    
    #
    #         "INSERT YOUR CODE HERE"
    #
    #     Revisit the sliding puzzle and the pancake puzzle for inspiration!
    #
    #     Note that you will need to add several functions to 
    #     complete this class. For example, a 'result' function is needed
    #     to satisfy the interface of 'search.Problem'.

    
    def __init__(self, warehouse):
        """

        """
        raise NotImplementedError()

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        raise NotImplementedError

    def result(self, state, action):
        """
        
        """
        pass

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
    
    ##         "INSERT YOUR CODE HERE"
    
    raise NotImplementedError()


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
    
    ##         "INSERT YOUR CODE HERE"
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def add_tuples(tuple1, tuple2):
    return tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]


class FindPathProblem(search.Problem):
    def __init__(self, initial, warehouse, goal=None):
        self.initial = initial
        self.goal = goal
        self.warehouse = warehouse

    def value(self, state):
        return 1  # Single movements have a cost of 1

    def result(self, state, action):
        # The result is the old state, with the action applied.
        new_state = add_tuples(state, action)
        return new_state

    def actions(self, state):
        for offset in SURROUNDINGS:
            new_state = add_tuples(state, offset)
            # Check that the location isn't a wall or box
            if new_state not in self.warehouse.boxes \
                    and new_state not in self.warehouse.walls:
                yield offset


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
    # separate x, y for usage, dst=(row = y, col = x)
    (row, col) = dst

    # the player is only able to move to a space and a target square
    ALLOWED_CELLS = [SPACE, TARGET_SQUARE] 

    # convert the warehouse to a string
    warehouseStr = warehouse.__str__()

    # convert warehouse string into Array<Array<char>>
    warehouse2D = [list(line) for line in warehouseStr.split('\n')]
    coordinates = warehouse2D[row][col]

    # check if the worker is allowed onto the given coordinates before checking if a valid path exists
    if coordinates not in ALLOWED_CELLS:
        return False

    # use manhattan distance for a* graph search |x2 - x1| + |y2 - y1|
    def heuristic(n):
        (s_row, s_col) = n.state
        return abs(s_row - row) + abs(s_col - col)

    # h = h()
    # path = search.astar_graph_search()
    # return path is not None

    # can just return the search.astar_graph_search() is not None ^^

    node = search.astar_graph_search(FindPathProblem(warehouse.worker, warehouse, (col, row)),
                       heuristic)

    # If a node was found, this is a valid destination
    return node is not None
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

