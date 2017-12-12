# from pytocl.main import main
# from my_driver import MyDriver
#
# def run():
#     main(MyDriver(logdata=False))
#
# if __name__ == '__main__':
#     run()
#     #main(Driver())


from pytocl.main import main
# import my_driver_swarm
# from my_driver import MyDriver
from my_driver import MyDriver
import my_driver_noswarm
import os

if __name__ == '__main__':
    swarm = True

    if swarm:

        filename1 = 'swarm_file1.marshal'
        filename2 = 'swarm_file2.marshal'
        filename3 = 'deleted.txt'

        if os.path.isfile(filename3) is not True:
            open(filename3, 'w+')
            if os.path.isfile(filename1):
                os.remove(filename1)
            if os.path.isfile(filename2):
                os.remove(filename2)
        main(MyDriver(logdata=False))
    else:
        my_driver = my_driver_noswarm.MyDriver(logdata=False)
        main(my_driver)
