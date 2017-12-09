

from pytocl.main import main
from my_driver import MyDriver

def run():
    main(MyDriver(logdata=False))
if __name__ == '__main__':
    run()
    #main(Driver())
