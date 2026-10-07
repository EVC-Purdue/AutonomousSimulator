"""Local ENU about (LAT0, LON0), the same equirectangular formula as gps_node."""
import math

LAT0 = 40.4380475
LON0 = -86.9442826
R = 6_371_000


def to_local(lat, lon):
    return (R * math.radians(lon - LON0) * math.cos(math.radians(LAT0)),
            R * math.radians(lat - LAT0))


def to_latlon(x, y):
    return (LAT0 + math.degrees(y / R),
            LON0 + math.degrees(x / (R * math.cos(math.radians(LAT0)))))
