import time
import threading
import itertools


class Spinner(threading.Thread):

    def __init__(self, delay=0.15):
        super(Spinner, self).__init__()
        self.__seq = itertools.cycle(['-', '\\', '|', '/'])
        self.delay = delay
        self.waiting_time = 0.1
        self._stop_event = threading.Event()

        # Thread states
        self.started = False
        self.spinning = False
        self.is_empty = True
        self.is_spinning = False
        self.is_stopped = True

    def start_(self):
        self._stop_event.clear()
        self.is_stopped = False
        self.started = True
        self.spinning = True
        threading.Thread(target=self.spinner_task, daemon=True).start()  # Start daemon

        # Wait on the daemon
        while not self.is_spinning:
            time.sleep(self.waiting_time)  # Wait a bit to prevent checking too often

    def pause(self):
        self.spinning = False

        # Wait on the daemon
        while self.is_spinning:
            time.sleep(self.waiting_time)  # Wait a bit to prevent checking too often

    def resume(self):
        self.spinning = True

        # Wait on the daemon
        while not self.is_spinning:
            time.sleep(self.waiting_time)  # Wait a bit to prevent checking too often

    def stop(self):

        if not self.is_empty:
            self.erase()

        self._stop_event.set()

        # Wait on the daemon
        while not self.is_stopped:
            time.sleep(self.waiting_time)  # Wait a bit to prevent checking too often

        self.spinning = False
        self.started = False

    def erase(self):
        if self.spinning:
            self.pause()

        print("\x1b[2K\r", end='', flush=True)  # Erase current content
        self.is_empty = True

    def msg(self, msg, field_size=0):

        if not self.is_empty:
            self.erase()

        if not self.started:  # No spinner started yet
            formatter = "{:<%d} " % field_size
            print(formatter.format(msg), end='', flush=True)  # Write message
            self.start_()  # Start the spinner
        else:
            formatter = "{:<%d}  " % field_size
            print(formatter.format(msg), end='', flush=True)  # Write message

        if not self.spinning:
            self.resume()

        self.is_empty = False

    def spinner_task(self):
        while not self._stop_event.is_set():

            if not self.spinning:
                self.is_spinning = False
            else:
                self.is_spinning = True
                print(next(self.__seq), end='', flush=True)
                time.sleep(self.delay)
                print("\b", end='', flush=True)

        self.is_stopped = True


if __name__ == "__main__":

    t = 1

    spinner = Spinner()
    print("Hello world", end='', flush=True)
    time.sleep(t)
    spinner.start_()
    time.sleep(t)
    spinner.msg("My msg", 25)
    time.sleep(t)
    spinner.stop()
    time.sleep(t)
    spinner.msg("2nd", 25)
    time.sleep(t)
    spinner.pause()
    time.sleep(t)
    spinner.resume()
    time.sleep(t)
    spinner.msg("This is almost the end !!!", 25)
    time.sleep(t)
    spinner.erase()
    time.sleep(t)
    spinner.msg("This is the end !!!", 25)
    time.sleep(t)
    spinner.stop()
