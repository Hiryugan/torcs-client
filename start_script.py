"""Application entry point."""
import argparse
import subprocess
from config_parser import Configurator
import shlex
import psutil

def main():
    """Main entry point of application."""
    parser = argparse.ArgumentParser(
        description='Configurator for torcs server.'
    )
    parser.add_argument(
        '-c',
        '--conf',
        help='Configuration file path.'
    )
    args = parser.parse_args()
    config_file = args.conf
    del args.conf
    configurator = Configurator(config_file)

    configurator.configure_server()
    configurator.configure_client()

    parser = configurator.parser
    script = 'torcs -r {} & python pytocl/run.py --conf {} -p {}'\
        .format(parser.server_config_file,
                parser.port + 1,
                config_file)

    args = shlex.split(script)
    proc = subprocess.Popen(args)
    process = psutil.Process(proc.pid)
    for pr in process.children(recursive=True):
        pr.kill()
    process.kill()

if __name__ == '__main__':
    main()
