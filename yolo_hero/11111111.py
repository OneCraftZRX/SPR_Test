
def update(list_,new_value,length):
    print(length)
    for i in range(length-1):
        list_[i]=list_[i+1]
    list[length-1]=new_value
    
    
list=[1,2,3,4,5,6,7,8,9]
update(list,15,len(list))
print(list)



    
#     # port_2.readAngle()
#     port.close_port()
    # port_1.close_port()
# port.close_port()
# from position_solver import PositionSolver
# from my_serial import SerialPort
# import time

# po=PositionSolver()

# po.solve_pitch(2.000,2.5000)

