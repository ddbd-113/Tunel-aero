"""Zjawiska pogodowe wpływające na osiągi: deszcz i oblodzenie.

Modele są półempiryczne (rzędy wielkości z literatury NASA: Bezos i in. - wpływ
silnego deszczu na profile lotnicze; Bragg i in. - model degradacji osiągów
przy oblodzeniu C_iced = (1 + eta * k) C_clean). Służą do testów odporności
i porównań "co jeśli", a nie do certyfikacji.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Degradation:
    """Mnożniki/dodatki aerodynamiczne wynikające z pogody (dla danej chwili)."""
    cl_factor: float = 1.0          # mnożnik współczynnika siły nośnej / nachylenia CL(alpha)
    cd_add: float = 0.0             # przyrost oporu pasożytniczego
    stall_reduction: float = 0.0    # zmniejszenie krytycznego kąta natarcia [rad]
    cm_factor: float = 1.0          # mnożnik skuteczności sterów / stateczności
    thrust_factor: float = 1.0      # mnożnik ciągu śmigieł
    prop_drag_factor: float = 1.0   # mnożnik strat profilowych łopat (lód zwiększa moment oporowy)
    rain_mass_flux: float = 0.0     # strumień masy wody deszczowej [kg/(m^2 s)]
    ice: float = 0.0                # stopień oblodzenia 0..1 (do logów)


def icing_temperature_factor(t_c: float) -> float:
    """Intensywność narastania lodu w funkcji temperatury (przechłodzone krople wody).

    0 powyżej 0 C, maksimum od -3 do -12 C, zanika poniżej -20 C (głównie kryształki lodu).
    """
    if t_c >= 0.0 or t_c <= -20.0:
        return 0.0
    if t_c > -3.0:
        return -t_c / 3.0
    if t_c > -12.0:
        return 1.0
    return (t_c + 20.0) / 8.0


class Weather:
    """Deszcz + oblodzenie w chmurze/mgle.

    Parametry:
      rain_rate: intensywność opadu [mm/h] (słaby < 2.5, umiarkowany 2.5-10, silny 10-50, ulewa > 50),
      lwc: zawartość wody ciekłej w chmurze/mgle [g/m^3] (typowo 0.1-1.0),
      cloud_base / cloud_top: granice warstwy chmur AGL [m] (oblodzenie tylko w niej),
      accretion_rate: współczynnik narastania lodu,
      deicing: czy pojazd ma instalację przeciwoblodzeniową (ogranicza oblodzenie o 85%).
    """

    def __init__(self, rain_rate: float = 0.0, lwc: float = 0.0, cloud_base: float = 0.0,
                 cloud_top: float = 0.0, accretion_rate: float = 2.0e-4, deicing: bool = False):
        self.rain_rate = rain_rate
        self.lwc = lwc
        self.cloud_base = cloud_base
        self.cloud_top = cloud_top
        self.k_acc = accretion_rate
        self.deicing = deicing
        self.ice = 0.0   # stan: stopień oblodzenia 0..1

    def in_cloud(self, h_agl: float) -> bool:
        return self.lwc > 0 and self.cloud_base <= h_agl <= self.cloud_top

    def update(self, dt: float, h_agl: float, t_c: float, exposure_speed: float) -> None:
        """Aktualizuje akumulację lodu. exposure_speed - prędkość opływu powierzchni nośnych/łopat [m/s]."""
        if self.in_cloud(h_agl):
            rate = self.k_acc * self.lwc * exposure_speed * icing_temperature_factor(t_c)
            if self.deicing:
                rate *= 0.15
            self.ice = min(1.0, self.ice + rate * dt)
        if t_c > 2.0 and self.ice > 0:
            self.ice = max(0.0, self.ice - self.ice * dt / 60.0)  # topnienie / zrzucanie lodu

    def degradation(self) -> Degradation:
        R = self.rain_rate
        eta = self.ice
        return Degradation(
            # deszcz: do ~ -8% CLmax i +40% oporu przy ulewie 100 mm/h; lód: do -30% CL, x2.5 opór
            cl_factor=(1.0 - min(0.08, 0.0008 * R)) * (1.0 - 0.30 * eta),
            cd_add=min(0.012, 0.00012 * R) + 0.05 * eta,
            stall_reduction=0.10 * eta,          # do ~6 stopni wcześniejsze przeciągnięcie
            cm_factor=1.0 - 0.15 * eta,
            thrust_factor=(1.0 - min(0.06, 0.0006 * R)) * (1.0 - 0.6 * eta),
            prop_drag_factor=1.0 + 2.0 * eta,
            rain_mass_flux=R / 3600.0,           # 1 mm/h = 1 kg/m^2/h
            ice=eta,
        )
