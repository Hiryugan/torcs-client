"""Application entry point."""
import argparse
import logging

import pickle

from pytocl.protocol import Client
import my_driver


def main(driver):
    """Main entry point of application."""
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
        '-t',
        '--track',
        help='Track data.',
        default='../config/config_template.yaml'
    )
    parser.add_argument('-v', help='Debug log level.', action='store_true')
    args = parser.parse_args()
    config_file = args.conf
    del args.conf
    track_info = args.track
    track_name = track_info.split('|')[0]
    del args.track
    driver.track_name = track_name
    # switch log level:
    if args.v:
        level = logging.DEBUG
    else:
        level = logging.INFO
    del args.v
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)7s %(name)s %(message)s"
    )

    # start client loop:
    client = Client(driver=driver, **args.__dict__)
    client.run()

    pickle.dump(driver.data, open('add_data/' + track_name + '.pkl', 'wb+'))
    exit(0)


if __name__ == '__main__':
    from pytocl.driver import Driver

    main(Driver())