"""Root library.

This module contains multiple base functions for general applications.

No __main__, just defining classes and functions.

"""


def seconds_to_hms(t):
    """Formats a datetime.timedelta object to a HH:MM:SS string."""

    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    if hours < 10:
        hours = "0%s" % int(hours)
    if minutes < 10:
        minutes = "0%s" % minutes
    if seconds < 10:
        seconds = "0%s" % seconds
    return "%s:%s:%s" % (hours, minutes, seconds)


if __name__ == '__main__':
    pass
