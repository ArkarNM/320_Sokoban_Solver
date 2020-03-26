from __future__ import print_function
from __future__ import division


from sokoban import Warehouse, find_2D_iterator
from mySokobanSolver import solve_sokoban_elem


t1 = '''
####
# .#  
#  ###
#*@  #
#  $ #
#  ###
####  
'''

t1_expected = '''
####  
#X #  
#  ###
#   X#
#   X#
#XX###
####  
'''

t3 ='''
#######
#@ $ .#
#. $  #
#######'''

t3_expected ='''
#######
#X    #
#    X#
#######'''

def same_multi_line_strings(s1,s2):
    '''
    Auxiliary function to test two multi line string representing warehouses
    '''
    L1 = [s.rstrip() for s in s1.strip().split('\n')]
    L2 = [s.rstrip() for s in s2.strip().split('\n')]
    S1 = '\n'.join(L1)
    S2 = '\n'.join(L2)
    return S1==S2

def test_taboo_cells(n):
    problem_file = "./warehouses/warehouse_%s.txt" % str(n)
    print("Testing:", problem_file)
    wh = Warehouse()
    wh.load_warehouse(problem_file)
    answer = solve_sokoban_elem(wh)
    print(answer)

def test_taboo(test):
    wh = Warehouse()
    print(test)
    # removes unneccessary \n
    wh.from_lines(test.split(sep='\n'))
    answer = solve_sokoban_elem(wh)
    print(answer)

if __name__ == "__main__":

    print('Test a Custom Puzzle')
    print("enter 'quit' to exit\n")
    c = input('Warehouse number: ')
    while c != 'quit':
        try:
            test_taboo_cells(c)
        except FileNotFoundError as e:
            print("Warehouse %s does not exist\n" % str(c))
        c = input('Warehouse number: ')