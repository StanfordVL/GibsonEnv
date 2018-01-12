#!/usr/bin/env python

import os
import subprocess
import time
import sys
import signal

if __name__ == '__main__':
    if os.environ.has_key('APP'):
        graphical_app = os.environ['APP']
        if os.environ.has_key('ARGS'):
            extra_args = os.environ['ARGS']
            command = graphical_app + ' ' + extra_args
        else:
            command = graphical_app

        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def print_return_code_and_shutdown(return_code):
            for line in process.stdout.readlines():
                sys.stdout.write(line)
            with open('/tmp/graphical-app.return_code', 'w') as fp:
                fp.write(str(return_code))
            subprocess.call(['sudo', 'supervisorctl', 'shutdown'],
                            stdout=subprocess.PIPE)
            sys.exit(return_code)

        def signal_handler(signum, frame):
            print_return_code_and_shutdown(128 + signum)
        signal.signal(signal.SIGTERM, signal_handler)

        while True:
            output = process.stdout.readline()
            if sys.version_info[0] >= 3:
                output = output.decode()
            if output == '' and process.poll() is not None:
                break
            print(output.rstrip())

        print_return_code_and_shutdown(process.returncode)
