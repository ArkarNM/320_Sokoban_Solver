
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Helper Functions ##

def concat_tuples(firstTuple, secondTuple):
    """

    """

    return firstTuple[0] + secondTuple[0], firstTuple[1] + secondTuple[1]

def check_if_corner_cell(warehouseMatrix, cell):
    """

    """
    
    for i, _ in enumerate(SURROUNDINGS):
        (ax, ay) = SURROUNDINGS[i]
        (bx, by) = SURROUNDINGS[(i+1) % 4]

        # if both are walls, as in is a corner, then return True
        if warehouseMatrix[cell[1] + ay][cell[0] + ax] is WALL and warehouseMatrix[cell[1] + by][cell[0] + bx] is WALL:
            return True
    return False

def check_if_along_wall(warehouseMatrix, cell):
    """

    """
 
    for (ax, ay) in enumerate(SURROUNDINGS):

        # if next to wall then return True
        if warehouseMatrix[cell[1] + ay][cell[0] + ax] is WALL:
            return True
    return False

def matrix_to_string(warehouseMatrix):
    """

    """

    return NEW_LINE.join([EMPTY_STRING.join(row) for row in warehouseMatrix])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Helper Classes ##

## NEED TO REFORMAT BELOW FUNCTION ##
class FindPathProblem(search.Problem):
    def __init__(self, initial, warehouse, goal=None):
        self.initial = initial
        self.goal = goal
        self.warehouse = warehouse

    def value(self, state):
        return 1  # Single movements have a cost of 1

    def result(self, state, action):
        # The result is the old state, with the action applied.
        new_state = concat_tuples(state, action)
        return new_state

    def actions(self, state):
        for offset in SURROUNDINGS:
            new_state = concat_tuples(state, offset)
            # Check that the location isn't a wall or box
            if new_state not in self.warehouse.boxes \
                    and new_state not in self.warehouse.walls:
                yield offset

class Heuristic():
    def __init__(self, col, row):
        self.col = col
        self.row = row

    def manhattan_distance(self, n):
        """

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
    warehouseMatrix = [list(line) for line in warehouseStr.split(NEW_LINE)]

    # ignore boxes for can_go_there method
    warehouse.boxes = []

    # rule 1: if a cell is a corner and not a target, then it is a taboo cell.
    for y, row in enumerate(warehouseMatrix):
        ''' old method '''
        inside = False
        ''' end old method '''
        for x, cell in enumerate(row):

            ''' can_go_there() method '''
            # print(can_go_there(warehouse, (y, x)), warehouse.worker, y, x, cell)
            # if can_go_there(warehouse, (y, x)) or cell is PLAYER:
            '''  can_go_there() method '''

            ''' old method '''
            # find the inside of the playing area
            if not inside and cell is WALL:
                inside = True
            elif inside:
                
                # if rest of row is outside of playing area then break the loop
                if all([cell is SPACE for cell in row[x:]]):
                    break
                ''' end old method '''
                if cell is not WALL and cell not in TARGETS:
                    # find corners to set as taboo, breaks when found
                    if check_if_corner_cell(warehouseMatrix, (x, y)):
                        warehouseMatrix[y][x] = TABOO

    # rule 2: all the cells between two corners along a wall are taboo if none of these cells is a target.
    for y, row in enumerate(warehouseMatrix):
        for x, cell in enumerate(row):
            # find a taboo cell and check rows and columns that apply to rule 2
            if cell is TABOO and check_if_corner_cell(warehouseMatrix, (x, y)):
                # from the taboo point get the rest of the row to the right of it and enumerate
                for row_x, row_cell in enumerate(warehouseMatrix[y][x + 1:]):
                    # if there's any targets or walls break
                    if row_cell in TARGETS or row_cell is WALL:
                        break

                    # this is the next point in the row, we use x because the rest of the row may be cut off to enumerate
                    next_in_row_from_taboo = x + (row_x + 1)

                    # find another taboo cell or corner
                    if row_cell is TABOO and check_if_corner_cell(warehouseMatrix, (next_in_row_from_taboo, y)):
                        # if the entire row is along a wall then the entire row is taboo
                        if all([check_if_along_wall(warehouseMatrix, (i, y)) for i in range(x + 1, next_in_row_from_taboo)]):
                            # fill with taboo
                            for x4 in range(x + 1, next_in_row_from_taboo):
                                warehouseMatrix[y][x4] = TABOO

                # from the taboo point get the rest of the column below it and enumerate over
                for col_y, col_cell in enumerate([row[x] for row in warehouseMatrix[y + 1:][:]]):
                    # if there's any targets or walls break
                    if col_cell in TARGETS or col_cell is WALL:
                        break

                    # this is the next point in the column, we use x because the rest of the col may be cut off to enumerate
                    next_in_col_from_taboo = y + (col_y + 1)
                    
                    # find another taboo cell or corner
                    if col_cell is TABOO and check_if_corner_cell(warehouseMatrix, (x, next_in_col_from_taboo)):
                        # if the entire column is along a wall then the entire column is taboo
                        if all([check_if_along_wall(warehouseMatrix, (x, i)) for i in range(y + 1, next_in_col_from_taboo)]):
                            # fill with taboo
                            for y4 in range(y + 1, next_in_col_from_taboo):
                                warehouseMatrix[y4][x] = TABOO


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
    warehouseMatrix = [list(line) for line in warehouseStr.split(NEW_LINE)]
    coordinates = warehouseMatrix[row][col]

    # check if the worker is allowed onto the given coordinates before checking if a valid path exists
    if coordinates not in ALLOWED_CELLS:
        return False
    
    h = Heuristic(col, row)

    # check if a valid path from the worker to the coordinate provided exists
    path = search.astar_graph_search(
                FindPathProblem(
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

