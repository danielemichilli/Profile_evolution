import datetime as dt
import math
import time

def convert(mjd):
  # Convert a date from mjd to datetime format
  jd =  mjd + 2400000.5
  jd = jd + 0.5
  F, I = math.modf(jd)
  I = int(I)
  A = math.trunc((I - 1867216.25)/36524.25)
  if I > 2299160:
    B = I + 1 + A - math.trunc(A / 4.)
  else:
    B = I
  C = B + 1524
  D = math.trunc((C - 122.1) / 365.25)
  E = math.trunc(365.25 * D)
  G = math.trunc((C - E) / 30.6001)
  day = C - E + F - math.trunc(30.6001 * G)
  if G < 13.5:
    month = G - 1
  else:
    month = G - 13
  if month > 2.5:
    year = D - 4716
  else:
    year = D - 4715
  
  return dt.date(year,month,int(day))


def year(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt.datetime(year=year, month=1, day=1)
    startOfNextYear = dt.datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction



