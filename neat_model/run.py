#! /usr/bin/env python3
from config_parser import Parser
from pytocl.main import main
from pytocl.driver import Driver
from pytocl.protocol import Client
import argparse
import signal
import logging
import functools

# from __future__ import print_function
from neat.checkpoint import Checkpointer
import os
import pickle

# import cart_pole
import time
import neat
# import visualize
#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
import subprocess
import neat
driver = None
fitness_path = None
first = 0
argsg = None
parserg = None
models = None
def eval_genome(genome, config, fitness_save_path):
    # net=neat.ctrnn.CTRNN.create(genome, config, 10)
    net=neat.nn.FeedForwardNetwork.create(genome, config)
    def signal_handler(signal, frame):
        print("Timeout reached!")
        exit(42)
    global driver
    global first
    global argsg
    global parserg
    global models
    driver = MyDriver(parserg, models=models, net=net)
    print('Driver is created')

    client = Client(parserg, driver=driver, **argsg.__dict__)
    print('Client is created')
    signal.signal(signal.SIGABRT, signal_handler)
    signal.alarm(150)
    client.run()
    signal.alarm(0)
    print('Client has finished running')
    #     first = 1
    # else:
    #     driver.set_net(net)

    with open(fitness_save_path,'r') as f:
        fitt=f.read()

    print("fitness *******************   ",fitt)

    return float(fitt)


# def eval_genomes(genomes, config):
#     for genome_id, genome in genomes:
#         genome.fitness = eval_genome(genome, config)
#

    # def eval_genome_func(genome, conf, fitness_save_path=os.path.join(config_parser.save_path, 'fitness_value.txt')):
    #     eval_genome(genome, conf, fitness_save_path)

def run():

    parser = argparse.ArgumentParser(
        description='Client for TORCS racing car simulation with SCRC network'
                    ' server.'
    )
    parser.add_argument(
        '--hostname',
        help='Racing server host name.',
        default='localhost'
    )
    parser.add_argument(
        '-p',
        '--port',
        help='Port to connect, 3001 - 3010 for clients 1 - 10.',
        type=int,
        default=3001
    )
    parser.add_argument(
        '-с',
        '--conf',
        help='Configuration file path.',
        default='../config/config_template.yaml'
    )
    parser.add_argument(
        '-r',
        '--rel',
        help='Winner file path.',
        default='None'
    )

    parser.add_argument('-v', help='Debug log level.', action='store_true')
    args = parser.parse_args()
    config_file = args.conf
    del args.conf
    # switch log level:
    if args.v:
        level = logging.DEBUG
    else:
        level = logging.INFO
    del args.v
    # del args.rel
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)7s %(name)s %(message)s"
    )

    r = args.rel
    del args.rel


    config_parser = Parser(config_file)
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    global driver
    global argsg
    global parserg
    parserg = config_parser
    argsg = args
    # net = neat.ctrnn.CTRNN.create(genome, config, 10)
    # driver = MyDriver(config_parser)
    #
    # main(driver)
    global models
    i2o = {0: 'accel', 1: 'brake', 2: 'steer'}
    models = {}
    for i in range(len(config_parser.output_model)):
        # print(self.parser.output_model[i])
        if config_parser.output_model[i]['name'] != 'None':
            models[i2o[i]] = pickle.load(open(config_parser.output_model[i]['name'], 'rb'))

    # pop = Checkpointer().restore_checkpoint("neat-checkpoint-33")



    eval_genome_func = functools.partial(eval_genome, fitness_save_path=os.path.join(config_parser.save_path, 'fitness_value.txt'))




    from os import listdir
    from os.path import isfile, join
    filenames = [f for f in listdir(config_parser.save_path) if isfile(join(config_parser.save_path, f))]
    maxx = 0
    for f in filenames:
        split = f.split('_')
        if len(split) == 2:
            if split[0] == 'checkpoint':
                num = int(split[1])
                if num > maxx:
                    maxx = num
    if maxx > 0:
        pop = Checkpointer().restore_checkpoint(os.path.join(config_parser.save_path, 'checkpoint_' + str(maxx)))
        print('reloaded checkpoint ', maxx)
    else:
        # config_path = os.path.join(local_dir, neat_config_path)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_parser.neat_config_file)
        pop = neat.Population(config)




    if r != 'None':
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_parser.neat_config_file)
        winner = pickle.load(open(r, 'rb'))
        net = neat.nn.FeedForwardNetwork.create(winner, config)

        driver = MyDriver(parserg, models=models, net=net)
        # del args.rel
        client = Client(parserg, driver=driver, **argsg.__dict__)
        client.run()


    else:




        stats = neat.StatisticsReporter()
        check = Checkpointer(1,time_interval_seconds=None,filename_prefix=os.path.join(config_parser.save_path, 'checkpoint_'))

        pop.add_reporter(stats)
        pop.add_reporter(check)
        pop.add_reporter(neat.StdOutReporter(True))

        pe = neat.ParallelEvaluator(1, eval_genome_func)
        winner = pop.run(pe.evaluate, 1)

        # Save the winner.
        with open(config_parser.save_path + '_winner.pkl', 'wb') as f:
            pickle.dump(winner, f)

        print(winner)

    # visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    # visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    # node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    # visualize.draw_net(config, winner, True, node_names=node_names)

    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward.gv")
    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward-enabled.gv", show_disabled=False)
    # visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                    filename="winner-feedforward-enabled-pruned.gv", show_di
        # print(x)




if __name__ == '__main__':
    run()
    #main(Driver())
