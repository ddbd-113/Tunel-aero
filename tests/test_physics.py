import math

import numpy as np
import pytest

from tunel_aero.atmosphere import Atmosphere, pressure_altitude
from tunel_aero.math3d import (cross3, cross_rows, dcm_to_quat, euler_from_quat, quat_from_euler,
                               quat_integrate, quat_to_dcm, wind_from_to_ned)
from tunel_aero.vehicles.battery import Battery
from tunel_aero.weather import Weather
from tunel_aero.wind import DrydenTurbulence, Microburst, WindProfile


# ---------------------------------------------------------------- atmosfera
def test_isa_sea_level():
    s = Atmosphere().at(0.0)
    assert s.temperature == pytest.approx(288.15)
    assert s.pressure == pytest.approx(101325.0)
    assert s.density == pytest.approx(1.225, rel=1e-3)
    assert s.speed_of_sound == pytest.approx(340.3, rel=1e-3)


def test_isa_tropopause():
    s = Atmosphere().at(11000.0)
    assert s.temperature == pytest.approx(216.65, abs=0.01)
    assert s.pressure == pytest.approx(22632, rel=2e-3)
    s20 = Atmosphere().at(20000.0)
    assert s20.pressure == pytest.approx(5474.9, rel=3e-3)


def test_hot_and_humid_air_is_thinner():
    std = Atmosphere().at(1000)
    hot = Atmosphere(temperature_offset=25).at(1000)
    humid = Atmosphere(relative_humidity=1.0, temperature_offset=15).at(0)
    dry = Atmosphere(relative_humidity=0.0, temperature_offset=15).at(0)
    assert hot.density < std.density
    assert hot.density_altitude > 1600   # ~ +36 m na każdy kelwin powyżej ISA
    assert humid.density < dry.density


def test_ground_conditions_and_baro_error():
    atm = Atmosphere.from_ground_conditions(500, temperature_c=-20)
    assert atm.at(500).temperature_c == pytest.approx(-20, abs=1e-6)
    # w mrozie wysokościomierz barometryczny zawyża przyrost wysokości
    dh = pressure_altitude(atm.at(600).pressure) - pressure_altitude(atm.at(500).pressure)
    assert dh > 100 * 1.08


# ---------------------------------------------------------------- matematyka
def test_quaternion_roundtrips():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r, p, y = rng.uniform(-3, 3), rng.uniform(-1.5, 1.5), rng.uniform(-3, 3)
        q = quat_from_euler(r, p, y)
        assert np.allclose(euler_from_quat(q), (r, p, y), atol=1e-9)
        q2 = dcm_to_quat(quat_to_dcm(q))
        assert min(np.linalg.norm(q2 - q), np.linalg.norm(q2 + q)) < 1e-9


def test_quat_integrate_constant_rate():
    q = quat_from_euler(0, 0, 0)
    for _ in range(1000):
        q = quat_integrate(q, np.array([0, 0, math.pi / 2]), 0.001)
    assert euler_from_quat(q)[2] == pytest.approx(math.pi / 2, abs=1e-9)


def test_cross_helpers():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([-2.0, 0.5, 4.0])
    assert np.allclose(cross3(a, b), np.cross(a, b))
    A = np.random.default_rng(1).normal(size=(6, 3))
    assert np.allclose(cross_rows(A, b), np.cross(A, b))
    assert np.allclose(cross_rows(b, A), np.cross(b, A))


def test_wind_direction_convention():
    vn, ve = wind_from_to_ned(10, 270)   # wiatr z zachodu wieje na wschód
    assert vn == pytest.approx(0, abs=1e-9) and ve == pytest.approx(10)


# ---------------------------------------------------------------- wiatr
def test_eurocode_profile():
    open_ = WindProfile(10, 0, roughness=0.05)
    city = WindProfile(10, 0, roughness=1.0)
    assert np.linalg.norm(open_.mean_wind(10)) == pytest.approx(10, rel=0.02)
    assert np.linalg.norm(city.mean_wind(10)) < 0.65 * 10                       # w mieście słabiej...
    assert city.friction_velocity() > open_.friction_velocity()                 # ...ale bardziej porywiście
    assert np.linalg.norm(open_.mean_wind(100)) > np.linalg.norm(open_.mean_wind(10))


def test_wind_layers_interpolation():
    p = WindProfile(layers=[[10, 5, 270], [100, 15, 270]], roughness=0.05)
    assert np.linalg.norm(p.mean_wind(55)) == pytest.approx(10, rel=1e-6)


@pytest.mark.parametrize("level", ["light", "moderate", "severe"])
def test_dryden_variance(level):
    rng = np.random.default_rng(42)
    turb = DrydenTurbulence(level, WindProfile(), rng)
    sig, _ = turb.params(50.0)
    xs = np.array([turb.update(0.01, 50.0, 20.0, 0.0) for _ in range(120000)])
    std = xs.std(axis=0)
    assert std[0] == pytest.approx(sig[0], rel=0.2)
    assert std[2] == pytest.approx(sig[2], rel=0.2)


def test_microburst_continuity_and_intensity():
    mb = Microburst(radius=500, max_outflow=15, t_start=0, ramp_time=1e-6)
    h = 2.0

    def ur(r, z):
        return mb.value(10, r, 0, z)[0]

    def w(r, z):
        return -mb.value(10, r, 0, z)[2]

    for r, z in ((300, 60), (700, 120), (150, 30)):
        div = (1 / r) * ((r + h) * ur(r + h, z) - (r - h) * ur(r - h, z)) / (2 * h) + (w(r, z + h) - w(r, z - h)) / (2 * h)
        assert abs(div) < 1e-4
    peak = max(ur(r, z) for r in range(100, 1500, 10) for z in range(10, 200, 5))
    assert peak == pytest.approx(15, rel=0.05)
    assert w(0, 100) < -3   # prąd zstępujący w osi


# ---------------------------------------------------------------- pogoda / bateria
def test_icing_accumulates_only_in_supercooled_cloud():
    w = Weather(lwc=0.5, cloud_base=100, cloud_top=500)
    for _ in range(600):
        w.update(1.0, 200, -8.0, 20.0)
    assert w.ice > 0.9
    w2 = Weather(lwc=0.5, cloud_base=100, cloud_top=500)
    for _ in range(600):
        w2.update(1.0, 200, 5.0, 20.0)       # powyżej zera - brak lodu
        w2.update(1.0, 50, -8.0, 20.0)       # pod chmurą - brak lodu
    assert w2.ice == 0.0
    d = w.degradation()
    assert d.cl_factor < 0.8 and d.thrust_factor < 0.5 and d.cd_add > 0.03


def test_cold_battery():
    warm, cold = Battery(temperature_c=25), Battery(temperature_c=-20)
    assert cold.capacity_factor() < 0.7 * warm.capacity_factor()
    assert cold.terminal_voltage(30) < warm.terminal_voltage(30) - 0.5


def test_battery_energy_and_heating():
    b = Battery(cells=4, capacity_ah=5.0, temperature_c=25.0)
    for _ in range(600):
        b.update(1.0, 250.0, 25.0)
    assert 0.4 < b.soc < 0.6
    assert b.energy_used_wh == pytest.approx(250 * 600 / 3600, rel=0.02)
    assert b.temp_c > 26.0   # nagrzewa się od prądu
    cold = Battery(cells=4, capacity_ah=5.0, temperature_c=-10.0)
    for _ in range(600):
        cold.update(1.0, 250.0, -10.0)
    assert cold.soc < b.soc - 0.1   # w mrozie ta sama energia "zjada" więcej pojemności
