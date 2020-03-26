from __future__ import print_function
from __future__ import division


from sokoban import Warehouse, find_2D_iterator
from mySokobanSolver import solve_sokoban_elem


def test(n):
    problem_file = "./warehouses/warehouse_%s.txt" % str(n)
    print("Testing:", problem_file)
    wh = Warehouse()
    wh.load_warehouse(problem_file)
    answer = solve_sokoban_elem(wh)
    print(answer)

if __name__ == "__main__":

    print('Test a Custom Puzzle')
    print("enter 'quit' to exit\n")
    c = input('Warehouse number: ')
    while c != 'quit':
        try:
            test(c)
        except FileNotFoundError as e:
            print("Warehouse %s does not exist\n" % str(c))
        c = input('Warehouse number: ')