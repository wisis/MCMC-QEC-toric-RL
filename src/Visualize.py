from .toric_model import Toric_code
from .mcmc import *
from .util import Action

def visual(size):
    tc=Toric_code(size)
    d=tc.system_size
    tc.plot_toric_code(tc.next_state,'Visual')
    while True:
        choice=int(input('Error(1)? Stabilizer(2)? or non-trivial loop(3)?'))
        if choice==1:
            operator=int(input('Operator?(I=0,X=1,Y=2,Z=3):'))
            row=(int(input('Row:')))%d
            col=(int(input('Column:')))%d
            action= Action(position = np.array([1, row, col]), action = operator)
            tc.step(action)
            tc.plot_toric_code(tc.next_state,'Visual')
        elif choice==2:
            operator=int(input('Operator?(X=2,Y=3):'))
            row=(int(input('Row:')))%d
            col=(int(input('Column:')))%d
            apply_stabilizer(tc,operator,row,col)
            tc.plot_toric_code(tc.next_state,'Visual')
        elif choice==3:
            operator=int(input('Operator?(I=0,X=1,Z=3):'))
            orientation=int(input('Horisontal(1) or Vertical(2)?'))
            if orientation*operator==1:
                row=(int(input('Row:')))%d
                for i in range(d):
                    action= Action(position = np.array([0, row, i]), action = operator)
                    tc.step(action)
            elif orientation*operator==2:
                col=(int(input('Column:')))%d
                for i in range(d):
                    action= Action(position = np.array([1, i, col]), action = operator)
                    tc.step(action)
            elif orientation*operator==3:
                row=(int(input('Row:')))%d
                for i in range(d):
                    action= Action(position = np.array([1, row, i]), action = operator)
                    tc.step(action)
            
            elif orientation*operator==6:
                col=(int(input('Column:')))%d
                for i in range(d):
                    action= Action(position = np.array([0, i, col]), action = operator)
                    tc.step(action)
            tc.plot_toric_code(tc.next_state,'Visual')


visual(3)           
        