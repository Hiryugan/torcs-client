"""Application entry point."""
import argparse
import logging

from pytocl.protocol import Client


def main(driver):
    """Main entry point of application."""


    # start client loop:
    client = Client(driver=driver, **args.__dict__)
    client.run()


if __name__ == '__main__':
    from pytocl.driver import Driver

    main(Driver())
