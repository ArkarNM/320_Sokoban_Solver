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
import math

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Global Variables ##

# sokoban squares
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

# game outcome
FAILED = 'Impossible'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Helper Functions ##

def add_action(state, action, scale=1):
    """
    adds the action tuple to the state tuple and returns
    """
    return state[0] + (scale * action[0]), state[1] + (scale * action[1])


def check_if_corner_cell(walls, dst):
    """
    checks the warehouse and determines if the cell is surrounded by a corner
    """
    for i in range(len(SURROUNDINGS)):
        (a_x, a_y) = SURROUNDINGS[i]
        (b_x, b_y) = SURROUNDINGS[(i + 1) % 4]

        # if both are walls, as in is a corner, then return True
        if (dst[1] + a_x, dst[0] + a_y) in walls and (dst[1] + b_x, dst[0] + b_y) in walls:
            return True
    return False


def check_if_along_wall(walls, dst):
    """
    checks the warehouse and determines if the cell is along a wall
    """
    (row, col) = dst
    for (a_x, a_y) in SURROUNDINGS:
        # if next to wall then return True
        if (col + a_x, row + a_y) in walls:
            return True
    return False


def matrix_to_string(warehouse_m):
    """
    converts a 2D array of chars to a string
    """
    return NEW_LINE.join([EMPTY_STRING.join(row) for row in warehouse_m])


def warehouse_to_matrix(warehouse):
    """
    converts a string to a 2D array of chars
    """
    return [list(line) for line in warehouse.__str__().split(NEW_LINE)]


def manhattan_distance(init, end):
    """
    manhattan distance |x2 - x1| + |y2 - y1|
    """
    return abs(end[0] - init[0]) + abs(end[1] - init[1])
    # return math.sqrt((end[0] - init[0])**2 + (end[1] - init[1])**2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Helper Classes ##

class PathProblem(search.Problem):
    # initialises the problem
    def __init__(self, warehouse, goal):
        self.initial = warehouse.worker
        self.boxes = set(warehouse.boxes)
        self.walls = set(warehouse.walls)
        self.goal = goal

    # list of possible actions
    def actions(self, state):
        for action in SURROUNDINGS:
            new_state = add_action(state, action)
            # check that the action doesn't result in a wall or box collision
            if new_state not in self.boxes and new_state not in self.walls:
                yield action

    # Return the old state, with the action applied.
    def result(self, state, action):
        return add_action(state, action)

    def h(self, n):
        """heuristic using manhattan distance for a* graph search |x2 - x1| + |y2 - y1|"""
        return manhattan_distance(self.goal, n.state)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def my_team():
    """
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    """
    return [(10212361, 'Jamie', 'Martin'), (9737197, 'Tolga', 'Pasin'), (000000, 'xxxx', 'xxxx')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def taboo_cells(warehouse):
    """
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
    """

    # convert warehouse into Array<Array<char>>
    warehouse_matrix = warehouse_to_matrix(warehouse)

    worker, walls = warehouse.worker, set(warehouse.walls)

    # ignore boxes for can_go_there method
    warehouse.boxes = []

    # iterate through rows and cols
    for row_index in range(warehouse.nrows):
        for col_index in range(warehouse.ncols):
            position = (row_index, col_index)
            cell = warehouse_matrix[row_index][col_index]

            # remove unnecessary chars
            if cell is PLAYER or cell is BOX:
                warehouse_matrix[row_index][col_index] = SPACE

            # rule 1: if a cell is a corner and not a target, then it is a taboo cell.
            if cell is not WALL and cell not in TARGETS:
                if check_if_corner_cell(walls, position) and (
                        position == worker or can_go_there(warehouse.copy(), position)):
                    warehouse_matrix[row_index][col_index] = TABOO

                    # rule 2: all the cells between two corners along a wall are taboo if none of these cells is a
                    # target. from the taboo point get the rest of the row to the right of it and enumerate
                    for taboo_col_index in range((col_index + 1), warehouse.ncols):
                        taboo_cell = warehouse_matrix[row_index][taboo_col_index]
                        # if there's any targets or walls break
                        if taboo_cell in TARGETS or taboo_cell is WALL:
                            break

                        taboo_position = (row_index, taboo_col_index)

                        # find another taboo cell or corner
                        if check_if_corner_cell(walls, taboo_position):
                            # if the entire row is along a wall then the entire row is taboo
                            rest_of_cells_along_wall = [check_if_along_wall(walls, (row_index, i)) for i in
                                                        range(col_index + 1, taboo_col_index)]
                            if all(rest_of_cells_along_wall):
                                # fill with taboo
                                for taboo_index in range(col_index + 1, taboo_col_index):
                                    warehouse_matrix[row_index][taboo_index] = TABOO

                    # from the taboo point get the rest of the column below it and enumerate over
                    for taboo_row_index in range((row_index + 1), warehouse.nrows):
                        taboo_cell = warehouse_matrix[taboo_row_index][col_index]
                        # if there's any targets or walls break
                        if taboo_cell in TARGETS or taboo_cell is WALL:
                            break

                        taboo_position = (taboo_row_index, col_index)

                        # find another taboo cell or corner
                        if check_if_corner_cell(walls, taboo_position):
                            # if the entire column is along a wall then the entire column is taboo
                            rest_of_cells_along_wall = [check_if_along_wall(walls, (i, col_index)) for i in
                                                        range(row_index + 1, taboo_row_index)]
                            if all(rest_of_cells_along_wall):
                                # fill with taboo
                                for taboo_index in range(row_index + 1, taboo_row_index):
                                    warehouse_matrix[taboo_index][col_index] = TABOO

    # return to string variable
    warehouse_str = matrix_to_string(warehouse_matrix)

    # remove target chars
    for square in TARGETS:
        warehouse_str = warehouse_str.replace(square, SPACE)

    return warehouse_str


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):
    """
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
    """

    def __init__(self, warehouse, macro=False, allow_taboo_push=False, push_costs=None):
        """
        initialisation function
        """
        if push_costs is not None:
            self.initial = str(tuple((warehouse.worker, list(zip(warehouse.boxes, push_costs)))))
        else:
            self.initial = str(tuple((warehouse.worker, [(box, 0) for box in warehouse.boxes])))

        self.macro = macro
        self.allow_taboo_push = allow_taboo_push
        # get a list of taboo_cells for usage
        self.taboo_cells = set(sokoban.find_2D_iterator(taboo_cells(warehouse).split(sep='\n'), "X"))
        # remove the player from the goal or target_square and move the boxes to the targets
        self.walls = set(warehouse.walls)
        self.goal = warehouse.targets

        # for #ing/debug purposes
        self.warehouse = warehouse

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        state = eval(state)

        boxes = [box for (box, cost) in state[1]]
        worker = state[0]

        #(self.warehouse.copy(worker=worker, boxes=boxes))

        if self.macro:
            # macro actions
            # go through boxes and determine what worker can do to them
            for box in boxes:
                # enumerate through possible surroundings
                for i, surr in enumerate(SURROUNDINGS):
                    # new position of the box when pushed
                    test_pos = add_action(box, surr)
                    # if we can't go there then it's not a valid move
                    warehouse = self.warehouse.copy(worker=worker, boxes=boxes)
                    if can_go_there(warehouse.copy(), tuple(reversed(test_pos))) or worker == test_pos:
                        # new position of the box when pushed, opposition direction of current surrounding
                        new_box_pos = add_action(box, surr, -1)
                        if new_box_pos not in boxes and new_box_pos not in self.walls:
                            # if allow taboo push, yield action or if test box not in taboo_cells
                            if self.allow_taboo_push or new_box_pos not in self.taboo_cells:
                                # get the opposite of the current action as in worker goes 'Left' but pushes box 'Right'
                                yield tuple(reversed(box)), ACTIONS[(i + 2) % 4]
        else:
            # elementary actions
            # enumerate through possible surroundings
            for i, surr in enumerate(SURROUNDINGS):
                # get the new position of adding the move to the worker
                test_pos = add_action(worker, surr)
                # test it's not a wall
                if test_pos not in self.walls:
                    # if it's within a box position test new position of box
                    if test_pos in boxes:
                        test_box = add_action(test_pos, surr)
                        # ensure the new box position doesn't merge with a wall, box
                        if test_box not in boxes and test_box not in self.walls:
                            # if allow taboo push, yield action or if not allowing, test box not in taboo_cells
                            if self.allow_taboo_push or test_box not in self.taboo_cells:
                                yield ACTIONS[i]
                    else:
                        yield ACTIONS[i]


    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
            state1 via action, assuming cost c to get up to state1. If the problem
            is such that the path doesn't matter, this function will only look at
            state2.  If the path does matter, it will consider c and maybe state1
            and action. The default method costs 1 for every step in the path."""
        push_cost = 0

        state1 = eval(state1)
        state2 = eval(state2)

        old_boxes, new_boxes = state1[1], state2[1]
        old_worker, new_worker = state1[0], state2[0]

        # checking we don't move more spaces than possible
        if not self.macro:
            assert(manhattan_distance(old_worker, new_worker) == 1)

        if new_boxes != old_boxes:
            # go through the old boxes to find the box not there
            for box_index, (box, cost) in enumerate(new_boxes):
                if (box, cost) not in old_boxes:
                    # go through to find the index of the old box to apply it's push_cost
                    push_cost = cost

        return c + 1 + push_cost

    def goal_test(self, state):
        """goal test to ensure all boxes are in a target_square, player position is irrelevant so remove"""
        state = eval(state)
        return [box for (box, cost) in state[1]] == self.goal

    def result(self, state, action):
        """
        action upon the given action and return the new state
        """
        (worker, boxes) = eval(state)

        if self.macro:
            # convert action ie 'Left' into tuple (-1, 0)
            next_pos = SURROUNDINGS[ACTIONS.index(action[1])]
            # get the new worker position, flip the action because it's row, col (y, x) not x, y
            worker = tuple(reversed(action[0]))
        else:
            # convert action ie 'Left' into tuple (-1, 0)
            next_pos = SURROUNDINGS[ACTIONS.index(action)]
            # get the new worker position
            worker = add_action(worker, next_pos)

        for i, (box, cost) in enumerate(boxes):
            if worker == box:
                new_box_pos = add_action(box, next_pos)
                assert(manhattan_distance(box, new_box_pos) == 1)
                boxes[i] = (new_box_pos, cost)

        return str(tuple((worker, boxes)))

    def h(self, n):
        """
        heuristic using manhattan distance for a* graph search |x2 - x1| + |y2 - y1|
        """
        state = eval(n.state)
        # initialise new warehouse to work on and get new tuples
        boxes = state[1]
        worker = state[0]

        worker_to_box_distances, box_to_target_totals = list(), list()

        # iterate through boxes to find the distance for each from worker
        for (box, _) in boxes:
            distance = manhattan_distance(worker, box)
            worker_to_box_distances.append(distance)
        # iterate through each perm of targets to find the distance between each box
        for targets_perm in itertools.permutations(self.goal):
            total_distance = 0
            # combines targets and boxes in tuples as in (target, box)
            zipped_tuples = zip(targets_perm, boxes)
            # for each target and box get the manhattan distance for each and add that to a total
            # so we have the total distance of all boxes to targets in this permuation
            for target, (box, cost) in zipped_tuples:
                distance = manhattan_distance(target, box)
                if cost != 0:
                    total_distance += (distance * cost)
                else:
                    total_distance += distance
            box_to_target_totals.append(total_distance)
        # return the smallest worker to box distance and smallest box to target total distance
        return min(worker_to_box_distances) + 10 * min(box_to_target_totals)
    # def h(self, n):
    #     # Perform a manhattan distance heuristic
    #     state = eval(n.state)
    #     # initialise new warehouse to work on and get new tuples
    #     boxes = state[1]
    #     worker = state[0]
    #
    #     num_targets = len(self.goal)
    #     heuristic = 0
    #     test = 1
    #     for (box, cost) in boxes:
    #         # dist = 0
    #         # for target in wh.targets:
    #         #     dist+= manhattan_distance(box, target)
    #         # heuristic += (dist/num_targets)
    #         if test == 1:
    #             dist = 0
    #             for target in self.goal:
    #                 dist += manhattan_distance(box, target)
    #             heuristic += 0.8 * (dist / num_targets) + 0.5 * manhattan_distance(worker, box)
    #         else:
    #             dist1 = []
    #             for target in self.goal:
    #                 dist1.append(manhattan_distance(box, target))
    #             heuristic += 0.8 * min(dist1) + 0.5 * manhattan_distance(worker, box)
    #     return heuristic


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    """

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
    """
    # copies warehouse into a new Sokoban puzzle
    puzzle = SokobanPuzzle(warehouse.copy())

    walls = set(warehouse.walls)

    boxes = warehouse.boxes
    worker = warehouse.worker

    # iterates over the actions
    for action in action_seq:
        # we can use the result() to get the acted upon result of each action
        state = puzzle.result(str(tuple((worker, [(box, 0) for box in boxes]))), action)

        state = eval(state)

        worker = state[0]

        boxes = [box for (box, cost) in state[1]]

        # ensures the worker hasn't clipped a wall
        if worker in walls:
            return FAILED

        # helper to ensure boxes aren't stacked

        box_stack = set()

        # ensures no boxes have clipped any walls
        for box in boxes:
            if box in box_stack or box in walls:
                return FAILED
            else:
                box_stack.add(box)

    return warehouse.copy(worker=worker, boxes=boxes).__str__()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_elem(warehouse):
    """
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
    """
    puzzle = SokobanPuzzle(warehouse)
    path = search.astar_graph_search(puzzle, puzzle.h)

    if path is not None:
        return path.solution()
    else:
        return FAILED


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def can_go_there(warehouse, dst):
    """
    Determine whether the worker can walk to the cell dst=(row,column)
    without pushing any box.

    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    """
    # separate row, col for usage below
    (row, col) = dst

    # the player is only able to move to a space and a target square
    ALLOWED_CELLS = set([SPACE, TARGET_SQUARE])

    # convert the warehouse to a Array<Array<char>>
    warehouse_matrix = warehouse_to_matrix(warehouse)

    # check if the worker is allowed onto the given coordinates before checking if a valid path exists
    if warehouse_matrix[row][col] not in ALLOWED_CELLS:
        return False

    # check if a valid path from the worker to the coordinate provided exists
    path = search.astar_graph_search(PathProblem(warehouse, (col, row)))

    return path is not None


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def solve_sokoban_macro(warehouse):
    """
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
    """
    puzzle = SokobanPuzzle(warehouse, macro=True)
    path = search.astar_graph_search(puzzle, puzzle.h)

    if path is not None:
        return path.solution()
    else:
        return FAILED


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban_elem(warehouse, push_costs):
    """
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
    """

    puzzle = SokobanPuzzle(warehouse, push_costs=push_costs)
    path = search.astar_graph_search(puzzle, puzzle.h)

    if path is not None:
        return path.solution()
    else:
        return FAILED

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
