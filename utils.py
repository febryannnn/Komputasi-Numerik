from decimal import Decimal, ROUND_HALF_UP
import math


def custom_round(num: float) -> float:
    try:
        num = float(num)
        if math.isnan(num) or math.isinf(num):
            raise ValueError("Invalid number")

        return float(
            Decimal(str(num)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

    except Exception:
        raise ValueError("Invalid number")


def Et(true: float, approx: float) -> float:
    approx = custom_round(approx)
    true = custom_round(true)

    if true == 0.0 and approx == 0.0:
        return 0.0

    if true == 0.0:
        return custom_round(abs(approx) * 100)

    return custom_round(abs((true - approx) / true) * 100)


def Ea(approx: float, approx_old: float) -> float:
    approx = custom_round(approx)
    approx_old = custom_round(approx_old)

    if approx == 0 and approx_old == 0:
        return 0.0

    if approx == 0.0:
        return custom_round(abs(approx_old) * 100)

    if approx == 0:
        return float("inf") if approx_old != 0 else 0
    return custom_round(abs((approx - approx_old) / approx) * 100)


def sign(num: float | int) -> str:
    # untuk string, jika num > 0 kita tambahkan +
    if num > 0:
        return "+"

    return ""
