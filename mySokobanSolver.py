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

# -- Global Variables -- #

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

# helper for corners, stored in (x, y)
SURROUNDINGS = [(0, -1), (-1, 0), (0, 1), (1, 0)]
ACTIONS = ['Up', 'Left', 'Down', 'Right']

# game outcome
FAILED = 'Impossible'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# -- Helper Functions -- ##

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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def my_team():
    """
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    """
    return [(10212361, 'Jamie', 'Martin'), (9737197, 'Tolga', 'Pasin')]


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

    # convert warehouse into 2D array of characters
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
                # if the position is a corner cell and is either where the worker is
                # or it can go there (we don't care about stuff outside of the playing field)
                # then set the character to a taboo cell
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

        stores the state as a (worker, str([(box, cost),...]) tuple.

        it's necessary to use the string as the search.py uses a hashset and lists aren't hashable
        """
        self.initial = (warehouse.worker, frozenset(zip(warehouse.boxes, push_costs))) \
            if push_costs is not None \
            else (warehouse.worker, frozenset((box, 0) for box in warehouse.boxes))

        # custom variable inputs
        self.push_costs = push_costs
        self.macro = macro
        self.allow_taboo_push = allow_taboo_push

        # helpers
        self.taboo_cells = set(sokoban.find_2D_iterator(taboo_cells(warehouse).split(sep='\n'), "X"))
        self.walls = set(warehouse.walls)
        self.goal = set(warehouse.targets)

        # for macro actions can_go_there purposes
        self.warehouse = warehouse

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        """
        (worker, boxes) = state
        boxes = set(box for (box, _) in boxes)

        # macro actions
        if self.macro:
            # go through boxes and determine what worker can do to them
            for box in boxes:
                # enumerate through possible surroundings of each box
                for i, surr in enumerate(SURROUNDINGS):
                    # test the possible surroundings for the worker to move to
                    test_pos = add_action(box, surr)
                    # if the worker can't go there then it's not a valid move
                    if worker == test_pos or \
                            can_go_there(self.warehouse.copy(worker=worker, boxes=boxes), tuple(reversed(test_pos))):
                        # new position of the box when pushed, opposition direction of current surrounding
                        new_box_pos = add_action(box, surr, -1)

                        # ensure the new box position doesn't merge with a wall, box and
                        # that allow taboo push is true or the test box not in taboo_cells
                        if new_box_pos not in boxes and new_box_pos not in self.walls \
                                and (self.allow_taboo_push or new_box_pos not in self.taboo_cells):
                            # get the opposite of the surrounding as in,
                            # worker goes to the 'Left' and pushes the box 'Right'
                            yield tuple(reversed(box)), ACTIONS[(i + 2) % 4]
        # elementary actions
        else:
            # enumerate through possible surroundings of the worker
            for i, surr in enumerate(SURROUNDINGS):
                # add the surrounding to the workers current position to test if it's viable
                test_pos = add_action(worker, surr)

                # ensure it's not in a wall
                if test_pos not in self.walls:
                    # if it's not in a box then the worker can move their
                    if test_pos not in boxes:
                        yield ACTIONS[i]

                    # if it's within a box test new position of the box
                    else:
                        # this is the position 2 spaces from the current worker
                        test_pos = add_action(test_pos, surr)
                        # ensure the new box position doesn't merge with a wall, box and
                        # that allow taboo push is true or the test box not in taboo_cells
                        if test_pos not in boxes and test_pos not in self.walls \
                                and (self.allow_taboo_push or test_pos not in self.taboo_cells):
                            yield ACTIONS[i]

    def path_cost(self, c, state1, action, state2):
        """
        Return the cost of the solution path that arrives at state2 from state1 via action
        """
        push_cost = 0

        # determines if we need to worry about push_costs
        if self.push_costs is not None:
            # copy the two states into workable variables
            (old_worker, old_boxes), (new_worker, new_boxes) = state1, state2
            # set comparison is unordered + we shouldn't have a case of box_stack up as this has already been checked
            old_boxes, new_boxes = set(old_boxes), set(new_boxes)

            # if the two are different try find the box that moved
            if new_boxes != old_boxes:
                for box_index, (box, cost) in enumerate(new_boxes):
                    # assign push_cost the cost of the box movement
                    if (box, cost) not in old_boxes:
                        push_cost = cost

        # returns the current cost + 1 for an action + the push cost
        return c + 1 + push_cost

    def goal_test(self, state):
        """
        goal test to ensure all boxes are in a target_square
        """
        (_, boxes) = state
        return set(box for (box, _) in boxes) == self.goal

    def result(self, state, action):
        """
        action upon the given action and return the new state
        """
        # copy the state into workable variables
        (worker, boxes) = state
        boxes = list(boxes)

        # macro result
        if self.macro:
            # convert action ie 'Left' into tuple (-1, 0)
            next_pos = SURROUNDINGS[ACTIONS.index(action[1])]
            # assigns the worker their new position
            # flip the action because it's in row, col (y, x) not x, y
            worker = tuple(reversed(action[0]))
        # elementary result
        else:
            # convert action ie 'Left' into tuple (-1, 0)
            next_pos = SURROUNDINGS[ACTIONS.index(action)]
            # assigns the worker their new position
            worker = add_action(worker, next_pos)

        # update the box if one is pushed
        for i, (box, cost) in enumerate(boxes):
            if worker == box:
                boxes[i] = (add_action(box, next_pos), cost)

        return worker, frozenset(boxes)

    def h(self, n):
        """
        heuristic using that defines the closest box to the worker
        and also the closest box to target combination,
        incoporating push_costs if necessary
        """
        # copy the state into workable variables
        (worker, boxes) = n.state
        boxes = list(boxes)

        # initialise the list of distances
        # we don't care about double ups we just want the smallest possible answer
        worker_to_box_distances, box_to_target_totals = set(), set()

        # iterate through boxes and append the distance for each from worker
        for (box, _) in boxes:
            worker_to_box_distances.add(manhattan_distance(worker, box))

        # iterate through each permutation of targets to find the distance between each box
        for targets_perm in itertools.permutations(self.goal):
            total_distance = 0

            # combines targets and boxes in tuples as in (target, box)
            zipped_tuples = zip(targets_perm, boxes)

            # for each target and box get the manhattan distance for each
            for target, (box, cost) in zipped_tuples:
                # cost is incorporated to ensure the worker understands
                # the effort required to push this box.
                # if it's 0 make it 1 because it still costs the worker to move there
                cost = cost if cost > 0 else 1
                # append the total distance of all boxes to targets in this permutation
                total_distance += manhattan_distance(target, box) * cost

            # add the total so we have
            box_to_target_totals.add(total_distance)

        # return the smallest worker to box distance and smallest box to target total distance
        return min(worker_to_box_distances) + min(box_to_target_totals)


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

    # initialises the walls, boxes and worker for use in the action sequence
    boxes, worker = warehouse.boxes, warehouse.worker

    # iterates over the actions
    for action in action_seq:
        # we can use the result() to get the state of the acted upon result of each action
        (worker, boxes) = puzzle.result((worker, frozenset((box, 0) for box in boxes)), action)
        # get the list of just the boxes, no costs
        boxes = list(box for (box, _) in boxes)

        # ensures the worker hasn't clipped a wall
        if worker in puzzle.walls:
            return FAILED

        # ensure boxes aren't stacked
        if len(boxes) != len(set(boxes)):
            return FAILED

        # ensures no boxes have clipped any walls
        for box in boxes:
            if box in puzzle.walls:
                return FAILED

    # return a copy of the warehouse with the new worker and boxes
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

    path = search.astar_graph_search(SokobanPuzzle(warehouse))

    if path is not None:
        return path.solution()

    return FAILED


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class PathProblem(search.Problem):

    def __init__(self, warehouse, goal):
        """initialises the problem"""
        self.initial = warehouse.worker
        self.boxes_and_walls = set(itertools.chain(warehouse.walls, warehouse.boxes))
        self.goal = goal

    def actions(self, state):
        """yield of all possible actions"""
        for action in SURROUNDINGS:
            # check that the new state from the given action doesn't result in a wall or box collision
            if add_action(state, action) not in self.boxes_and_walls:
                yield action

    def result(self, state, action):
        """return the old state with the action applied"""
        return add_action(state, action)

    def h(self, n):
        """heuristic using manhattan distance for a* graph search |x2 - x1| + |y2 - y1|"""
        return manhattan_distance(self.goal, n.state)


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
    allowed_cells = {SPACE, TARGET_SQUARE}

    # convert the warehouse to a Array<Array<char>>
    warehouse_matrix = warehouse_to_matrix(warehouse)

    # check if the worker is allowed onto the given coordinates before checking if a valid path exists
    if warehouse_matrix[row][col] not in allowed_cells:
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

    path = search.astar_graph_search(SokobanPuzzle(warehouse, macro=True))

    if path is not None:
        return path.solution()

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

    path = search.astar_graph_search(SokobanPuzzle(warehouse, push_costs=push_costs))

    if path is not None:
        return path.solution()

    return FAILED

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
