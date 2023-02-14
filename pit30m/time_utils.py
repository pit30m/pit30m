"""time utilities.

GPS - UTC conversion from https://stackoverflow.com/a/35772372
"""
import bisect
from datetime import datetime, timedelta

_LEAP_DATES = ((1981, 6, 30), (1982, 6, 30), (1983, 6, 30),
               (1985, 6, 30), (1987, 12, 31), (1989, 12, 31),
               (1990, 12, 31), (1992, 6, 30), (1993, 6, 30),
               (1994, 6, 30), (1995, 12, 31), (1997, 6, 30),
               (1998, 12, 31), (2005, 12, 31), (2008, 12, 31),
               (2012, 6, 30), (2015, 6, 30), (2016, 12, 31))

LEAP_DATES = tuple(datetime(i[0], i[1], i[2], 23, 59, 59) for i in _LEAP_DATES)

def leap(date):
    """
    Return the number of leap seconds since 1980-01-01

    :param date: datetime instance
    :return: leap seconds for the date (int)
    """
    # bisect.bisect returns the index `date` would have to be
    # inserted to keep `LEAP_DATES` sorted, so is the number of
    # values in `LEAP_DATES` that are less than `date`, or the
    # number of leap seconds.
    return bisect.bisect(LEAP_DATES, date)

def gps_seconds_to_utc(gps_secs):
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    date_before_leaps = gps_epoch + timedelta(seconds=gps_secs)
    return date_before_leaps - timedelta(seconds=leap(date_before_leaps))

def gps2utc(week, secs):
    """
    :param week: GPS week number, i.e. 1866
    :param secs: number of seconds since the beginning of `week`
    :return: datetime instance with UTC time
    """
    secs_in_week = 604800
    gps_secs = week * secs_in_week + secs
    return gps_seconds_to_utc(gps_secs)