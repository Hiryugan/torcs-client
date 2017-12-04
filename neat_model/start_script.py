"""Application entry point."""
import argparse
import subprocess
from config_parser import Configurator
import shlex
import psutil
def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    pid = process.pid
    return rc, pid

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
    torcs_args = parser.parse_args()
    config_file = torcs_args.conf
    del torcs_args.conf
    configurator = Configurator(config_file)

    configurator.configure_server()
    configurator.configure_client()

    parser = configurator.parser
    torcs_script = 'torcs -r {} &'.format(parser.server_config_file)
    python_script = 'python run.py -p {} --conf {} '.format(3000 + parser.port + 1, config_file)

    torcs_args = shlex.split(torcs_script)
    torcs_proc = subprocess.Popen(torcs_args, stdout=subprocess.PIPE)
    rc = 42
    while rc == 42:
        rc, python_pid = run_command(python_script)
        process = psutil.Process(python_pid)
        for pr in process.children(recursive=True):
            pr.kill()
        process.kill()

    process = psutil.Process(torcs_proc.pid)
    for pr in process.children(recursive=True):
        pr.kill()
    process.kill()


if __name__ == '__main__':
    main()