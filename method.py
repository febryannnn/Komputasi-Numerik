from typing import Literal
import sympy as sp
import pandas as pd
from utils import custom_round, Ea, Et
import numpy as np


class BiSection:
    def __init__(
        self,
        f: str,
        xl: float,
        xu: float,
        x_true: float,
        max_iter: int = 10,
        tol: float = 0.1,
    ) -> None:
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.xl = xl
        self.xu = xu
        self.x_true = x_true
        self.max_iter = max_iter
        self.tol = tol

    def evaluate(self, expr, val: float) -> float:
        return float(expr.subs(self.x, val))

    def solve(self):
        iterasi = 0
        xr_old = self.xl

        rows = []
        steps = []
        err = None
        while True:
            fl = custom_round(self.evaluate(self.f, self.xl))
            fu = custom_round(self.evaluate(self.f, self.xu))
            if fl * fu > 0:
                err = f"**Error**: Pemilihan batas interval tidak valid (f(xl) dan f(xu) memiliki tanda yang sama)."
                break

            xr = custom_round((self.xl + self.xu) / 2.0)
            fxr = custom_round(self.evaluate(self.f, xr))
            iterasi += 1
            try:
                ea = Ea(xr, xr_old)
                et = Et(self.x_true, xr)
            except (ZeroDivisionError, ValueError):
                err = f"**Error**: Angka tidak valid pada perhitungan error pada iterasi {iterasi}"
                break

            fl = custom_round(self.evaluate(self.f, self.xl))
            fu = custom_round(self.evaluate(self.f, self.xu))

            rows.append(
                {
                    "Iterasi": iterasi,
                    "XL": custom_round(self.xl),
                    "XU": custom_round(self.xu),
                    "XR": custom_round(xr),
                    "f(XL)": fl,
                    "f(XU)": fu,
                    "f(XR)": fxr,
                    "Et (%)": et,
                    "Ea (%)": None if iterasi == 1 else ea,
                }
            )

            step = f"**Iterasi {iterasi}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"f(x_l) &= {fl} \\\\\n"
            step += f"f(x_u) &= {fu} \\\\[1em]\n"
            step += f"x_r &= \\frac{{x_l + x_u}}{{2}} = \\frac{{{custom_round(self.xl)} + {custom_round(self.xu)}}}{{2}} = {xr} \\\\[1em]\n"
            step += f"f(x_r) &= {fxr} \n"
            step += "\\end{aligned}\n$$ \n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({xr})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
                if self.x_true != 0
                else f"E_t &= \\left| {self.x_true} - ({xr}) \\right| \\times 100\\% = {et}\\%"
            )
            if iterasi > 1:
                error.append(
                    f"E_a &= \\left| \\frac{{{xr} - ({custom_round(xr_old)})}}{{{xr}}} \\right| \\times 100\\% = {ea}\\%"
                    if xr != 0
                    else f"E_a &= \\left| {xr} - ({custom_round(xr_old)}) \\right| \\times 100\\% = {ea}\\%"
                )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\[0.5em]\n".join(error)
            step += "\n\\end{aligned}\n$$\n\n"

            xr_old = xr

            if fl * fxr < 0.0:
                step += (
                    f"Karena $f(x_l) \\cdot f(x_r) < 0$, maka $x_u$ baru $= {xr}$ \n\n"
                )
                self.xu = xr
            elif fl * fxr > 0.0:
                step += (
                    f"Karena $f(x_l) \\cdot f(x_r) > 0$, maka $x_l$ baru $= {xr}$ \n\n"
                )
                self.xl = xr

            steps.append(step)

            if et < self.tol or ea < self.tol:
                steps.append(f"### **Konvergen** pada iterasi ke-{iterasi}")
                break

            if iterasi >= self.max_iter:
                err = f"**Tidak konvergen** setelah {self.max_iter} iterasi."
                break

        return pd.DataFrame(rows), steps, custom_round(xr_old), err


class FalsePosition:
    def __init__(
        self,
        f: str,
        xl: float,
        xu: float,
        x_true: float,
        max_iter: int = 10,
        tol: float = 0.1,
    ) -> None:
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.xl = xl
        self.xu = xu
        self.x_true = x_true
        self.max_iter = max_iter
        self.tol = tol

    def evaluate(self, expr, val: float) -> float:
        return float(expr.subs(self.x, val))

    def solve(self):
        iterasi = 0
        xr_old = self.xl
        rows = []
        steps = []
        err = None

        while True:
            fl = custom_round(self.evaluate(self.f, self.xl))
            fu = custom_round(self.evaluate(self.f, self.xu))
            if fl * fu > 0:
                err = f"**Error**: Pemilihan batas interval tidak valid (f(xl) dan f(xu) memiliki tanda yang sama)."
                break

            try:
                xr = custom_round(self.xu - (fu * (self.xl - self.xu)) / (fl - fu))
            except (ZeroDivisionError, ValueError):
                err = f"**Error**: Pembagian dengan nol (f(xl) - f(xu) = 0) terjadi pada iterasi ke-{iterasi}."
                break

            fxr = custom_round(self.evaluate(self.f, xr))
            iterasi += 1

            try:
                ea = Ea(xr, xr_old)
                et = Et(self.x_true, xr)
            except (ZeroDivisionError, ValueError):
                err = f"**Error**: Angka tidak valid pada perhitungan error pada iterasi {iterasi}"
                break

            rows.append(
                {
                    "Iterasi": iterasi,
                    "XL": custom_round(self.xl),
                    "XU": custom_round(self.xu),
                    "XR": custom_round(xr),
                    "f(XL)": fl,
                    "f(XU)": fu,
                    "f(XR)": fxr,
                    "Et (%)": et,
                    "Ea (%)": None if iterasi == 1 else ea,
                }
            )

            step = f"**Iterasi {iterasi}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"f(x_l) &= {fl} \\\\\n"
            step += f"f(x_u) &= {fu} \\\\[1em]\n"
            step += f"x_r &= x_u - \\frac{{f(x_u) \\cdot (x_l - x_u)}}{{f(x_l) - f(x_u)}} = {custom_round(self.xu)} - \\frac{{ {fu} \\cdot ({custom_round(self.xl)} - ({custom_round(self.xu)}))}}{{{fl} - ({fu})}} = {xr} \\\\[1em]\n"
            step += f"f(x_r) &= {fxr}\n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({xr})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
                if self.x_true != 0
                else f"E_t &= \\left| {self.x_true} - ({xr}) \\right| \\times 100\\% = {et}\\%"
            )
            if iterasi > 1:
                error.append(
                    f"E_a &= \\left| \\frac{{{xr} - ({custom_round(xr_old)})}}{{{xr}}} \\right| \\times 100\\% = {ea}\\%"
                    if xr != 0
                    else f"E_a &= \\left| {xr} - ({custom_round(xr_old)}) \\right| \\times 100\\% = {ea}\\%"
                )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\[0.5em]\n".join(error)
            step += "\n\\end{aligned}\n$$\n\n"

            xr_old = xr

            if fl * fxr < 0.0:
                step += (
                    f"Karena $f(x_l) \\cdot f(x_r) < 0$, maka $x_u$ baru $= {xr}$ \n\n"
                )
                self.xu = xr
            elif fl * fxr > 0.0:
                step += (
                    f"Karena $f(x_l) \\cdot f(x_r) > 0$, maka $x_l$ baru $= {xr}$ \n\n"
                )
                self.xl = xr

            steps.append(step)

            if et < self.tol or ea < self.tol:
                steps.append(f"### **Konvergen** pada iterasi ke-{iterasi}")
                break

            if iterasi >= self.max_iter:
                err = f"**Tidak konvergen** setelah {self.max_iter} iterasi."
                break

        return pd.DataFrame(rows), steps, custom_round(xr_old), err


class FixedPoint:
    def __init__(
        self, f: str, x0: float, x_true: float, max_iter: int = 10, tol: float = 0.1
    ) -> None:
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.x0 = x0
        self.x_true = x_true
        self.max_iter = max_iter
        self.tol = tol

    def evaluate(self, expr, val: float) -> float:
        return float(expr.subs(self.x, val))

    def solve(self):
        iterasi = 0
        x_old = self.x0

        rows = []
        steps = []
        err = None

        while True:
            x_old = custom_round(x_old)
            try:
                x_new = custom_round(self.evaluate(self.f, x_old))
            except Exception:
                err = f"**Error**: Angka tidak valid pada perhitungan x pada iterasi {iterasi + 1}."
                break

            iterasi += 1

            # check agar tidak overflow
            if x_new > self.x_true + float(1e6) or x_new < self.x_true - float(1e6):
                err = f"**Error**: x_{iterasi} melebihi batas maksimum."
                break

            try:
                ea = Ea(x_new, x_old)
                et = Et(self.x_true, x_new)
            except (ValueError, ZeroDivisionError):
                err = f"**Error**: Angka tidak valid pada perhitungan error pada iterasi {iterasi}"
                break

            rows.append(
                {
                    "Iterasi": iterasi,
                    "x_i": x_old,
                    "x_(i+1)": x_new,
                    "Et (%)": et,
                    "Ea (%)": None if iterasi == 1 else ea,
                }
            )

            step = f"**Iterasi {iterasi}:**\n\n"

            val_str = f"({x_old})" if x_old < 0 else str(x_old)

            substituted_eq = self.f.subs(self.x, sp.Symbol(val_str))
            latex_eq = sp.latex(substituted_eq)

            step += "$$\n\\begin{aligned}\n"
            step += f"x_{{{iterasi}}} &= {latex_eq} = {x_new}\n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({x_new})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
                if self.x_true != 0
                else f"E_t &= \\left| {self.x_true} - ({x_new}) \\right| \\times 100\\% = {et}\\%"
            )
            error.append(
                f"E_a &= \\left| \\frac{{{x_new} - ({custom_round(x_old)})}}{{{x_new}}} \\right| \\times 100\\% = {ea}\\%"
                if x_new != 0
                else f"E_a &= \\left| {x_new} - ({custom_round(x_old)}) \\right| \\times 100\\% = {ea}\\%"
            )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\[0.5em]\n".join(error)
            step += "\n\\end{aligned}\n$$\n\n"

            x_old = x_new

            steps.append(step)

            if et < self.tol or ea < self.tol:
                steps.append(f"### **Konvergen** pada iterasi ke-{iterasi}")
                break

            if iterasi >= self.max_iter:
                err = f"**Tidak konvergen** setelah {self.max_iter} iterasi."
                break

        return pd.DataFrame(rows), steps, custom_round(x_old), err


class NewtonRaphson:
    def __init__(
        self, f: str, x0: float, x_true: float, max_iter: float = 10, tol: float = 0.1
    ) -> None:
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.df = sp.diff(self.f, self.x)
        self.x0 = x0
        self.x_true = x_true
        self.max_iter = max_iter
        self.tol = tol

    def evaluate(self, expr, val: float) -> float:
        return float(expr.subs(self.x, val))

    def solve(self):
        iterasi = 0
        x_old = self.x0

        rows = []
        steps = []
        err = None
        while True:
            iterasi += 1
            fx = custom_round(self.evaluate(self.f, x_old))
            dfx = custom_round(self.evaluate(self.df, x_old))
            try:
                x_new = custom_round(x_old - (fx / dfx))
            except (ZeroDivisionError, ValueError):
                err = f"**Error**: Pembagian dengan nol f'(x_{iterasi}) = 0 terjadi pada iterasi ke-{iterasi}."
                break

            try:
                ea = Ea(x_new, x_old)
                et = Et(self.x_true, x_new)
            except ValueError:
                err = f"**Error**: Angka tidak valid pada perhitungan error pada iterasi {iterasi}"
                break

            rows.append(
                {
                    "Iterasi": iterasi,
                    "x_i": custom_round(x_old),
                    "f(x_i)": fx,
                    "f'(x_i)": dfx,
                    "x_(i+1)": x_new,
                    "Et (%)": custom_round(et),
                    "Ea (%)": custom_round(ea),
                }
            )

            step = f"**Iterasi {iterasi}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"f(x_{{{iterasi - 1}}}) &= {fx} \\\\\n"
            step += f"f'(x_{{{iterasi - 1}}}) &= {dfx} \\\\[1em]\n"
            step += f"x_{{{iterasi}}} &= x_{{{iterasi - 1}}} - \\frac{{f(x_{{{iterasi - 1}}})}}{{f'(x_{{{iterasi - 1}}})}} = {x_new} \n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({x_new})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
                if self.x_true != 0
                else f"E_t &= \\left| {self.x_true} - ({x_new}) \\right| \\times 100\\% = {et}\\%"
            )
            error.append(
                f"E_a &= \\left| \\frac{{{x_new} - ({custom_round(x_old)})}}{{{x_new}}} \\right| \\times 100\\% = {ea}\\%"
                if x_new != 0
                else f"E_a &= \\left| {x_new} - ({custom_round(x_old)}) \\right| \\times 100\\% = {ea}\\%"
            )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\[0.5em]\n".join(error)
            step += "\n\\end{aligned}\n$$\n\n"

            steps.append(step)

            x_old = x_new

            if et < self.tol or ea < self.tol:
                steps.append(f"### **Konvergen** pada iterasi ke-{iterasi}")
                break

            if iterasi >= self.max_iter:
                err = f"**Tidak konvergen** setelah {self.max_iter} iterasi."
                break

        return pd.DataFrame(rows), steps, custom_round(x_old), err


class Secant:
    def __init__(
        self,
        f: str,
        x0: float,
        x1: float,
        x_true: float,
        max_iter: float = 10,
        tol: float = 0.1,
    ) -> None:
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.x0 = x0
        self.x1 = x1
        self.x_true = x_true
        self.max_iter = max_iter
        self.tol = tol

    def evaluate(self, expr, val: float) -> float:
        return float(expr.subs(self.x, val))

    def solve(self):
        iterasi = 0
        x_old_0 = self.x0
        x_old_1 = self.x1

        rows = []
        steps = []
        err = None
        while True:
            iterasi += 1
            fx0_raw = self.evaluate(self.f, x_old_0)
            fx1_raw = self.evaluate(self.f, x_old_1)
            fx0 = custom_round(fx0_raw)
            fx1 = custom_round(fx1_raw)
            try:
                x_new = custom_round(
                    x_old_1 - (fx1_raw * (x_old_0 - x_old_1)) / (fx0_raw - fx1_raw)
                )
            except (ZeroDivisionError, ValueError):
                err = f"**Error**: Pembagian dengan nol terjadi f(x_{iterasi - 2}) sama dengan f(x_{iterasi - 1}) pada iterasi ke-{iterasi}."
                break

            try:
                ea = Ea(x_new, x_old_1)
                et = Et(self.x_true, x_new)
            except ValueError:
                err = f"**Error**: Angka tidak valid pada perhitungan error pada iterasi {iterasi}"
                break

            rows.append(
                {
                    "Iterasi": iterasi,
                    "x_(i-1)": custom_round(x_old_0),
                    "x_i": custom_round(x_old_1),
                    "f(x_(i-1))": fx0,
                    "f(x_i)": fx1,
                    "x_(i+1)": x_new,
                    "Et (%)": custom_round(et),
                    "Ea (%)": custom_round(ea),
                }
            )

            step = f"**Iterasi {iterasi}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"f(x_{{{iterasi - 2}}}) &= {fx0} \\\\\n"
            step += f"f(x_{{{iterasi - 1}}}) &= {fx1} \\\\[1em]\n"
            step += f"x_{{{iterasi}}} &= x_{{{iterasi - 1}}} - \\frac{{f(x_{{{iterasi - 1}}}) \\cdot (x_{{{iterasi - 2}}} - x_{{{iterasi - 1}}})}}{{f(x_{{{iterasi - 2}}}) - f(x_{{{iterasi - 1}}})}} = {x_new} \n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({x_new})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
                if self.x_true != 0
                else f"E_t &= \\left| {self.x_true} - ({x_new}) \\right| \\times 100\\% = {et}\\%"
            )
            error.append(
                f"E_a &= \\left| \\frac{{{x_new} - ({custom_round(x_old_1)})}}{{{x_new}}} \\right| \\times 100\\% = {ea}\\%"
                if x_new != 0
                else f"E_a &= \\left| {x_new} - ({custom_round(x_old_1)}) \\right| \\times 100\\% = {ea}\\%"
            )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\[0.5em]\n".join(error)
            step += "\n\\end{aligned}\n$$\n\n"

            steps.append(step)

            x_old_0 = x_old_1
            x_old_1 = x_new

            if et < self.tol or ea < self.tol:
                steps.append(f"### **Konvergen** pada iterasi ke-{iterasi}")
                break

            if iterasi >= self.max_iter:
                err = f"**Tidak konvergen** setelah {self.max_iter} iterasi."
                break

        return pd.DataFrame(rows), steps, custom_round(x_old_1), err


class MNewtonRaphson:
    def __init__(
        self, f: str, x0: float, x_true: float, max_iter: float = 10, tol: float = 0.1
    ) -> None:
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.df = sp.diff(self.f, self.x)
        self.ddf = sp.diff(self.df, self.x)
        self.x0 = x0
        self.x_true = x_true
        self.max_iter = max_iter
        self.tol = tol

    def evaluate(self, expr, val: float) -> float:
        return float(expr.subs(self.x, val))

    def solve(self):
        iterasi = 0
        x_old = self.x0

        rows = []
        steps = []
        err = None
        while True:
            iterasi += 1
            fx = custom_round(self.evaluate(self.f, x_old))
            dfx = custom_round(self.evaluate(self.df, x_old))
            ddfx = custom_round(self.evaluate(self.ddf, x_old))
            try:
                x_new = custom_round(x_old - (fx * dfx) / ((dfx**2) - (fx * ddfx)))
            except (ZeroDivisionError, ValueError):
                err = f"**Error**: Pembagian dengan nol terjadi pada iterasi ke-{iterasi}."
                break

            try:
                ea = Ea(x_new, x_old)
                et = Et(self.x_true, x_new)
            except ValueError:
                err = f"**Error**: Angka tidak valid pada perhitungan error pada iterasi {iterasi}"
                break

            rows.append(
                {
                    "Iterasi": iterasi,
                    "x_i": custom_round(x_old),
                    "f(x_i)": fx,
                    "f'(x_i)": dfx,
                    "x_(i+1)": x_new,
                    "Et (%)": custom_round(et),
                    "Ea (%)": custom_round(ea),
                }
            )

            step = f"**Iterasi {iterasi}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"f(x_{{{iterasi - 1}}}) &= {fx} \\\\\n"
            step += f"f'(x_{{{iterasi - 1}}}) &= {dfx} \\\\\n"
            step += f"f''(x_{{{iterasi - 1}}}) &= {ddfx} \\\\[1em]\n"
            step += f"x_{{{iterasi}}} &= x_{{{iterasi - 1}}} - \\frac{{f(x_{{{iterasi - 1}}}) \\cdot f'(x_{{{iterasi - 1}}})}}{{[f'(x_{{{iterasi - 1}}})]^2 - (f(x_{{{iterasi - 1}}}) \\cdot f''(x_{{{iterasi - 1}}}))}} = {x_new} \n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({x_new})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
                if self.x_true != 0
                else f"E_t &= \\left| {self.x_true} - ({x_new}) \\right| \\times 100\\% = {et}\\%"
            )
            error.append(
                f"E_a &= \\left| \\frac{{{x_new} - ({custom_round(x_old)})}}{{{x_new}}} \\right| \\times 100\\% = {ea}\\%"
                if x_new != 0
                else f"E_a &= \\left| {x_new} - ({custom_round(x_old)}) \\right| \\times 100\\% = {ea}\\%"
            )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\[0.5em]\n".join(error)
            step += "\n\\end{aligned}\n$$\n\n"

            steps.append(step)

            x_old = x_new

            if et < self.tol or ea < self.tol:
                steps.append(f"### **Konvergen** pada iterasi ke-{iterasi}")
                break

            if iterasi >= self.max_iter:
                err = f"**Tidak konvergen** setelah {self.max_iter} iterasi."
                break

        return pd.DataFrame(rows), steps, custom_round(x_old), err


# class PolynomFactorization:
#     def __init__(self, f: str, max_iter: int = 10) -> None:
#         self.x = sp.symbols("x")
#         self.f = sp.sympify(f)
#         self.max_iter = max_iter

#     @staticmethod
#     def ABC(a: float = 1, b: float = 1, c: float = 1) -> tuple[float, float]:
#         dis = b**2 - 4 * a * c
#         if dis < 0:
#             return np.nan, np.nan

#         sq = np.sqrt(dis)
#         x1 = ((-b) + sq) / (2 * a)
#         x2 = ((-b) - sq) / (2 * a)
#         return custom_round(x1), custom_round(x2)

#     def _solve_deg3(self, coeff: list[int]):
#         A2, A1, A0 = coeff

#         rows = []
#         b0 = 0
#         a1 = A2
#         a0 = A1

#         for i in range(self.max_iter):
#             b0 = custom_round(A0 / a0)
#             a1 = custom_round(A2 - b0)
#             a0 = custom_round(A1 - a1 * b0)
#             rows.append({"Iterasi": i + 1, "b0": b0, "a1": a1, "a0": a0})

#         x1 = -1 * b0
#         x2, x3 = self.ABC(a=1, b=a1, c=a0)

#         return rows, (x1, x2, x3)

#     def _solve_deg4(self, coeff: list[int]):
#         A3, A2, A1, A0 = coeff

#         rows = []
#         b1 = 0
#         b0 = 0
#         a1 = A3
#         a0 = A2
#         for i in range(self.max_iter):
#             b0 = custom_round(A0 / a0)
#             b1 = custom_round((A1 - a1 * b0) / a0)
#             a1 = custom_round(A3 - b1)
#             a0 = custom_round(A2 - b0 - a1 * b1)
#             rows.append({"Iterasi": i + 1, "b0": b0, "b1": b1, "a1": a1, "a0": a0})

#         x1, x2 = self.ABC(b=b1, c=b0)
#         x3, x4 = self.ABC(b=a1, c=a0)
#         return rows, (x1, x2, x3, x4)

#     def _solve_deg5(self, coeff: list[int]):
#         A4, A3, A2, A1, A0 = coeff

#         rows = []
#         a0 = 0
#         b1 = 0
#         b0 = 0
#         c1 = A4
#         c0 = A3
#         for i in range(self.max_iter):
#             b0 = custom_round((A1 - a0 * A2 + a0**2 * A3 - a0**3 * A4 + a0**4) / c0)
#             b1 = custom_round((A2 - a0 * A3 + a0**2 * A4 - a0**3 + c1 * b0) / c0)
#             a0 = custom_round(A0 / (b0 * c0))
#             c1 = custom_round(A4 - a0 - b1)
#             c0 = custom_round(A3 - a0 * A4 + a0**2 - b0 - c1 * b1)
#             rows.append(
#                 {"Iterasi": i + 1, "b0": b0, "b1": b1, "a0": a0, "c1": c1, "c0": c0}
#             )

#         x1 = -1 * a0
#         x2, x3 = self.ABC(b=b1, c=b0)
#         x4, x5 = self.ABC(b=c1, c=c0)
#         return rows, (x1, x2, x3, x4, x5)

#     def solve(self):
#         degree = self.f.as_poly().degree()
#         coeff = [int(self.f.coeff(self.x, i)) for i in range(degree, -1, -1)]
#         err = None
#         match degree:
#             case 3:
#                 rows, roots = self._solve_deg3(coeff)
#             case 4:
#                 rows, roots = self._solve_deg4(coeff)
#             case 5:
#                 rows, roots = self._solve_deg5(coeff)
#             case _:
#                 err = f"Derajat Polinomial {degree} tidak didukung pada program ini"

#         return rows, roots, err


class PolynomFactorization:
    def __init__(self, f: str, max_iter: int = 10) -> None:
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.max_iter = max_iter

    def ABC(self, a=1, b=1, c=1):
        dis = b**2 - 4 * a * c
        if dis < 0:
            return np.nan, np.nan

        sq = np.sqrt(dis)
        x1 = ((-b) + sq) / (2 * a)
        x2 = ((-b) - sq) / (2 * a)
        return custom_round(x1), custom_round(x2)

    def solve(self):
        poly = self.f.as_poly()
        degree = poly.degree()

        coeff = [poly.coeff_monomial(self.x**i) for i in range(degree, -1, -1)]
        coeff = [int(c) for c in coeff]

        if degree == 2:
            A2, A1, A0 = coeff
            x1, x2 = self.ABC(a=A2, b=A1, c=A0)

            rows = [{"Iterasi": 0, "Info": "Rumus ABC langsung"}]
            steps = [
                f"""
**Langkah:**

$$
\\begin{{aligned}}
x &= \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}} \\\\
a &= {A2}, \\quad b = {A1}, \\quad c = {A0}
\\end{{aligned}}
$$
"""
            ]
            return rows, steps, (x1, x2), None

        elif degree == 3:
            A3, A2, A1, A0 = coeff

            rows = []
            steps = []

            b0 = 0
            a1 = A2
            a0 = A1

            rows.append({"Iterasi": 0, "b0": b0, "a1": a1, "a0": a0})
            steps.append(
                f"""
**Iterasi 0:**

$$
\\begin{{aligned}}
a_1 &= {A2} \\\\
a_0 &= {A1}
\\end{{aligned}}
$$
"""
            )

            for i in range(1, self.max_iter):
                a0_old = a0

                b0 = custom_round(A0 / a0_old)
                a1 = custom_round(A2 - b0)
                a0 = custom_round(A1 - a1 * b0)

                rows.append({"Iterasi": i, "b0": b0, "a1": a1, "a0": a0})
                steps.append(
                    f"""
**Iterasi {i}:**

$$
\\begin{{aligned}}
b_0 &= \\frac{{{A0}}}{{{a0_old}}} = {b0} \\\\
a_1 &= {A2} - ({b0}) = {a1} \\\\
a_0 &= {A1} - ({a1} \\cdot {b0}) = {a0}
\\end{{aligned}}
$$
"""
                )

            x1 = -1 * b0
            x2, x3 = self.ABC(b=a1, c=a0)

            return rows, steps, (x1, x2, x3), None

        elif degree == 4:
            A4, A3, A2, A1, A0 = coeff

            rows = []
            steps = []

            b1 = 0
            b0 = 0
            a1 = A3
            a0 = A2

            rows.append({"Iterasi": 0, "b0": b0, "b1": b1, "a1": a1, "a0": a0})
            steps.append(
                f"""
**Iterasi 0:**

$$
\\begin{{aligned}}
a_1 &= {A3} \\\\
a_0 &= {A2}
\\end{{aligned}}
$$
"""
            )

            for i in range(1, self.max_iter):
                a0_old = a0

                b0 = custom_round(A0 / a0_old)
                b1 = custom_round((A1 - a1 * b0) / a0_old)
                a1 = custom_round(A3 - b1)
                a0 = custom_round(A2 - b0 - a1 * b1)

                rows.append({"Iterasi": i, "b0": b0, "b1": b1, "a1": a1, "a0": a0})
                steps.append(
                    f"""
**Iterasi {i}:**

$$
\\begin{{aligned}}
b_0 &= \\frac{{{A0}}}{{{a0_old}}} = {b0} \\\\
b_1 &= \\frac{{{A1} - ({a1} \\cdot {b0})}}{{{a0_old}}} = {b1} \\\\
a_1 &= {A3} - {b1} = {a1} \\\\
a_0 &= {A2} - {b0} - ({a1} \\cdot {b1}) = {a0}
\\end{{aligned}}
$$
"""
                )

            x1, x2 = self.ABC(b=b1, c=b0)
            x3, x4 = self.ABC(b=a1, c=a0)

            return rows, steps, (x1, x2, x3, x4), None

        elif degree == 5:
            A5, A4, A3, A2, A1, A0 = coeff

            rows = []
            steps = []

            a0 = 0
            b1 = 0
            b0 = 0
            c1 = A4
            c0 = A3

            rows.append(
                {"Iterasi": 0, "b0": b0, "b1": b1, "a0": a0, "c1": c1, "c0": c0}
            )
            steps.append(
                f"""
**Iterasi 0:**

$$
\\begin{{aligned}}
c_1 &= {A4} \\\\
c_0 &= {A3}
\\end{{aligned}}
$$
"""
            )

            for i in range(1, self.max_iter):
                c0_old = c0

                b0 = custom_round(
                    (A1 - a0 * A2 + a0**2 * A3 - a0**3 * A4 + a0**4) / c0_old
                )
                b1 = custom_round(
                    (A2 - a0 * A3 + a0**2 * A4 - a0**3 + c1 * b0) / c0_old
                )
                a0 = custom_round(A0 / (b0 * c0_old))
                c1 = custom_round(A4 - a0 - b1)
                c0 = custom_round(A3 - a0 * A4 + a0**2 - b0 - c1 * b1)

                rows.append(
                    {"Iterasi": i, "b0": b0, "b1": b1, "a0": a0, "c1": c1, "c0": c0}
                )
                steps.append(
                    f"""
**Iterasi {i}:**

$$
\\begin{{aligned}}
b_0 &= {b0} \\\\
b_1 &= {b1} \\\\
a_0 &= {a0} \\\\
c_1 &= {c1} \\\\
c_0 &= {c0}
\\end{{aligned}}
$$
"""
                )

            x1 = -1 * a0
            x2, x3 = self.ABC(b=b1, c=b0)
            x4, x5 = self.ABC(b=c1, c=c0)

            return rows, steps, (x1, x2, x3, x4, x5), None

        else:
            return None, None, None, f"Derajat {degree} tidak didukung"


class LinearRegression:
    def __init__(self, data, n=0, mode="std"):
        self.data = data
        self.n = len(self.data) if n == 0 else n
        self.mode = mode

    def solve(self):
        rows = []
        steps = []
        err = None

        match self.mode:
            case "std":
                x = [d[0] for d in self.data]
                y = [d[1] for d in self.data]
            case "log":
                x = [custom_round(np.log10(d[0])) for d in self.data]
                y = [custom_round(np.log10(d[1])) for d in self.data]
            case "exp":
                x = [d[0] for d in self.data]
                y = [custom_round(np.log(d[1])) for d in self.data]
            case _:
                return pd.DataFrame(), [], None, f"Mode '{self.mode}' tidak didukung"

        sum_x = custom_round(sum(x))
        sum_y = custom_round(sum(y))
        xy = [custom_round(x[i] * y[i]) for i in range(self.n)]
        sum_xy = custom_round(sum(xy))
        x2 = [custom_round(x[i] ** 2) for i in range(self.n)]
        sum_x2 = custom_round(sum(x2))
        avg_x = custom_round(sum_x / self.n)
        avg_y = custom_round(sum_y / self.n)

        try:
            a1 = custom_round(
                (self.n * sum_xy - sum_x * sum_y) / (self.n * sum_x2 - sum_x**2)
            )
        except (ZeroDivisionError, ValueError):
            return pd.DataFrame(), [], None, "Pembagian dengan nol saat menghitung a1"
        a0 = custom_round(avg_y - a1 * avg_x)

        for i in range(self.n):
            rows.append(
                {
                    "x": x[i],
                    "y": y[i],
                    "x*y": xy[i],
                    "x^2": x2[i],
                }
            )

        step = "**Perhitungan Regresi Linear"
        match self.mode:
            case "log":
                step += " (Logaritmik)"
            case "exp":
                step += " (Eksponensial)"
        step += ":**\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"\\sum x &= {sum_x} \\\\\n"
        step += f"\\sum y &= {sum_y} \\\\\n"
        step += f"\\sum xy &= {sum_xy} \\\\\n"
        step += f"\\sum x^2 &= {sum_x2} \\\\\n"
        step += f"\\bar{{x}} &= {avg_x} \\\\\n"
        step += f"\\bar{{y}} &= {avg_y} \\\\[1em]\n"
        step += f"a_1 &= \\frac{{n \\sum xy - \\sum x \\sum y}}{{n \\sum x^2 - (\\sum x)^2}} = \\frac{{{self.n} \\cdot {sum_xy} - {sum_x} \\cdot {sum_y}}}{{{self.n} \\cdot {sum_x2} - ({sum_x})^2}} = {a1} \\\\\n"
        step += f"a_0 &= \\bar{{y}} - a_1 \\bar{{x}} = {avg_y} - {a1} \\cdot {avg_x} = {a0}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        match self.mode:
            case "std":
                equation = f"{a0} + {a1}x"
                steps.append(f"### Persamaan: $y = {a0} + {a1}x$")
            case "log":
                a = custom_round(10**a0)
                equation = f"{a} * x^{a1}"
                steps.append(f"### Persamaan: $y = {a} \\cdot x^{{{a1}}}$")
            case "exp":
                a = custom_round(np.exp(a0))
                equation = f"{a} * e^({a1}x)"
                steps.append(f"### Persamaan: $y = {a} \\cdot e^{{{a1}x}}$")

        return pd.DataFrame(rows), steps, (a0, a1), err


class QuadraticRegression:
    def __init__(self, data, n=0):
        self.data = data
        self.n = len(self.data) if n == 0 else n

    def solve(self):
        rows = []
        steps = []
        err = None

        x = [d[0] for d in self.data]
        y = [d[1] for d in self.data]

        sum_x = custom_round(sum(x))
        sum_x2 = custom_round(sum([custom_round(x[i] ** 2) for i in range(self.n)]))
        sum_x3 = custom_round(sum([custom_round(x[i] ** 3) for i in range(self.n)]))
        sum_x4 = custom_round(sum([custom_round(x[i] ** 4) for i in range(self.n)]))
        sum_y = custom_round(sum(y))
        sum_xy = custom_round(sum([custom_round(x[i] * y[i]) for i in range(self.n)]))
        sum_x2y = custom_round(
            sum([custom_round((x[i] ** 2) * y[i]) for i in range(self.n)])
        )

        for i in range(self.n):
            rows.append(
                {
                    "x": x[i],
                    "y": y[i],
                    "x^2": custom_round(x[i] ** 2),
                    "x^3": custom_round(x[i] ** 3),
                    "x^4": custom_round(x[i] ** 4),
                    "x*y": custom_round(x[i] * y[i]),
                    "x^2*y": custom_round((x[i] ** 2) * y[i]),
                }
            )

        A = [[self.n, sum_x, sum_x2], [sum_x, sum_x2, sum_x3], [sum_x2, sum_x3, sum_x4]]
        B = [sum_y, sum_xy, sum_x2y]

        step = "**Perhitungan Regresi Kuadratik:**\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"\\sum x &= {sum_x}, \\quad \\sum x^2 = {sum_x2}, \\quad \\sum x^3 = {sum_x3}, \\quad \\sum x^4 = {sum_x4} \\\\\n"
        step += f"\\sum y &= {sum_y}, \\quad \\sum xy = {sum_xy}, \\quad \\sum x^2 y = {sum_x2y}\n"
        step += "\\end{aligned}\n$$\n\n"
        step += "Sistem persamaan:\n\n"
        step += "$$\n\\begin{bmatrix}\n"
        step += f"{self.n} & {sum_x} & {sum_x2} \\\\\n"
        step += f"{sum_x} & {sum_x2} & {sum_x3} \\\\\n"
        step += f"{sum_x2} & {sum_x3} & {sum_x4}\n"
        step += "\\end{bmatrix}\n"
        step += "\\begin{bmatrix} a_0 \\\\ a_1 \\\\ a_2 \\end{bmatrix} = \n"
        step += f"\\begin{{bmatrix}} {sum_y} \\\\ {sum_xy} \\\\ {sum_x2y} \\end{{bmatrix}}\n$$\n\n"
        steps.append(step)

        GJ = GaussJordan(A, B)
        _, gj_steps, a, gj_err = GJ.solve()
        if gj_err is not None:
            return pd.DataFrame(rows), steps, None, gj_err

        steps.extend(gj_steps)
        a0, a1, a2 = custom_round(a[0]), custom_round(a[1]), custom_round(a[2])
        steps.append(f"### Persamaan: $y = {a0} + {a1}x + {a2}x^2$")

        return pd.DataFrame(rows), steps, (a0, a1, a2), err


class GaussJordan:
    def __init__(self, A, B):
        self.A = [row[:] for row in A]
        self.B = B[:]

    def _get_matrix(self):
        n = len(self.A)
        m = len(self.A[0]) if n > 0 else 0

        col_format = " ".join(["c"] * m) + " | c"

        latex = "$$\n\\left[\n\\begin{array}{" + col_format + "}\n"
        for i in range(n):
            row_str = " & ".join([f"{custom_round(self.A[i][j])}" for j in range(m)])
            row_str += f" & {custom_round(self.B[i])}"

            latex += f"    {row_str}"
            if i < n - 1:
                latex += " \\\\"
            latex += "\n"

        latex += "\\end{array}\n\\right]\n$$\n"
        return latex

    def solve(self):
        steps = []
        err = None

        n, m = len(self.A), len(self.A[0])
        iterasi = 1

        step = "**Matriks Awal:**\n" + self._get_matrix() + "\n"
        step += "**Eliminasi Gauss-Jordan:**\n\n"

        # Gauss method
        for i in range(0, min(n, m)):
            if self.A[i][i] != 0 and self.A[i][i] != 1:
                pivot = self.A[i][i]
                for j in range(i, m):
                    self.A[i][j] /= pivot
                self.B[i] /= pivot

                step += f"**Iterasi {iterasi}:** $B_{{{i}}} \\leftarrow \\frac{{B_{{{i}}}}}{{{custom_round(pivot)}}}$\n"
                step += self._get_matrix()
                iterasi += 1

            for j in range(i + 1, n):
                times = self.A[j][i]
                if times != 0:
                    for k in range(i, m):
                        self.A[j][k] -= self.A[i][k] * times
                    self.B[j] -= self.B[i] * times

                    # TODO: ketika times == 1 ga perlu dikali lagi
                    step += f"**Iterasi {iterasi}:** $B_{{{j}}} \\leftarrow B_{{{j}}} - ({custom_round(times)}) B_{{{i}}}$\n"
                    step += self._get_matrix()
                    iterasi += 1

        # Jordan method
        for i in range(min(n, m) - 1, -1, -1):
            if self.A[i][i] == 1:
                for j in range(i - 1, -1, -1):
                    times = self.A[j][i]
                    if times != 0:
                        for k in range(0, m):
                            self.A[j][k] -= self.A[i][k] * times
                        self.B[j] -= self.B[i] * times

                        step += f"**Iterasi {iterasi}:** $B_{{{j}}} \\leftarrow B_{{{j}}} - ({custom_round(times)}) B_{{{i}}}$\n"
                        step += self._get_matrix()
                        iterasi += 1

        for i in range(0, n):
            for j in range(0, m):
                self.A[i][j] = custom_round(self.A[i][j])
            self.B[i] = custom_round(self.B[i])

        step += "**Hasil Akhir Eliminasi:**\n"
        step += self._get_matrix()

        step += "**Solusi:**\n" + ", ".join(
            [f"$x_{{{i}}} = {self.B[i]}$" for i in range(n)]
        )

        steps.append(step)

        return None, steps, self.B, err


class Jacobi:
    def __init__(self, A, B, tol=0.1, max_iter=10):
        self.A = A
        self.B = B
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        rows = []
        steps = []
        err = None

        n, m = len(self.A), len(self.B)
        x = np.zeros((m,))
        x_new = x.copy()

        for iter_num in range(self.max_iter):
            for i in range(m):
                sum_v = 0
                for j in range(m):
                    if i != j:
                        sum_v += self.A[i][j] * x[j]
                x_new[i] = (self.B[i] - sum_v) / self.A[i][i]

            row = {"Iterasi": iter_num + 1}
            for i in range(m):
                row[f"x{i+1}"] = custom_round(x_new[i])
            rows.append(row)

            step = f"**Iterasi {iter_num + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            for i in range(m):
                sum_parts = " - ".join(
                    [
                        f"{self.A[i][j]} \\cdot {custom_round(x[j])}"
                        for j in range(m)
                        if i != j
                    ]
                )
                step += f"x_{{{i+1}}} &= \\frac{{{self.B[i]} - ({sum_parts})}}{{{self.A[i][i]}}} = {custom_round(x_new[i])}"
                if i < m - 1:
                    step += " \\\\\n"
            step += "\n\\end{aligned}\n$$\n\n"
            steps.append(step)

            # Check convergence
            converged = True
            i = 0
            while i < m:
                if x_new[i] - x[i] > self.tol:
                    converged = False
                    break
                i += 1

            if converged:
                x = x_new.copy()
                steps.append(f"### **Konvergen** pada iterasi ke-{iter_num + 1}")
                break
            x = x_new.copy()

        for i in range(m):
            x[i] = custom_round(x[i])

        result = x.tolist()
        return pd.DataFrame(rows), steps, result, err


class GaussSeidel:
    def __init__(self, A, B, tol=0.1, max_iter=10):
        self.A = A
        self.B = B
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        rows = []
        steps = []
        err = None

        n, m = len(self.A), len(self.B)
        x = np.zeros((m,))
        x_new = x.copy()

        for iter_num in range(self.max_iter):
            for i in range(m):
                sum_v = 0
                for j in range(m):
                    if i != j:
                        sum_v += self.A[i][j] * x_new[j]
                x_new[i] = (self.B[i] - sum_v) / self.A[i][i]

            row = {"Iterasi": iter_num + 1}
            for i in range(m):
                row[f"x{i+1}"] = custom_round(x_new[i])
            rows.append(row)

            step = f"**Iterasi {iter_num + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            for i in range(m):
                sum_parts = " - ".join(
                    [
                        f"{self.A[i][j]} \\cdot {custom_round(x_new[j])}"
                        for j in range(m)
                        if i != j
                    ]
                )
                step += f"x_{{{i+1}}} &= \\frac{{{self.B[i]} - ({sum_parts})}}{{{self.A[i][i]}}} = {custom_round(x_new[i])}"
                if i < m - 1:
                    step += " \\\\\n"
            step += "\n\\end{aligned}\n$$\n\n"
            steps.append(step)

            # Check convergence
            converged = True
            i = 0
            while i < m:
                if x_new[i] - x[i] > self.tol:
                    converged = False
                    break
                i += 1

            if converged:
                x = x_new.copy()
                steps.append(f"### **Konvergen** pada iterasi ke-{iter_num + 1}")
                break
            x = x_new.copy()

        for i in range(m):
            x[i] = custom_round(x[i])

        result = x.tolist()
        return pd.DataFrame(rows), steps, result, err


class NewtonInterpolation:
    def __init__(self, data, n=0, x=0):
        self.data = data
        self.n = n if n != 0 else len(self.data)
        self.x = x

    def solve(self):
        rows = []
        steps = []
        err = None

        diff_table = [[0 for _ in range(self.n - 1)] for _ in range(self.n)]

        for i in range(self.n - 1):
            diff_table[i][0] = (self.data[i + 1][1] - self.data[i][1]) / (
                self.data[i + 1][0] - self.data[i][0]
            )

        for i in range(1, self.n - 1):
            for j in range(self.n - i - 1):
                diff_table[j][i] = (diff_table[j + 1][i - 1] - diff_table[j][i - 1]) / (
                    self.data[j + i + 1][0] - self.data[j][0]
                )

        # Round the table
        for i in range(self.n):
            for j in range(self.n - 1):
                if diff_table[i][j] != 0:
                    diff_table[i][j] = custom_round(diff_table[i][j])

        # Build rows
        for i in range(self.n):
            row = {"x": self.data[i][0], "y": self.data[i][1]}
            for j in range(self.n - 1):
                row[f"f[{j+1}]"] = diff_table[i][j] if diff_table[i][j] != 0 else ""
            rows.append(row)

        step = "**Tabel Divided Difference:**\n\n"
        step += "Koefisien: "
        coeffs = [custom_round(self.data[0][1])]
        for i in range(self.n - 1):
            coeffs.append(custom_round(diff_table[0][i]))
        step += ", ".join([str(c) for c in coeffs])
        step += "\n\n"
        steps.append(step)

        res = self.data[0][1]
        step2 = f"**Evaluasi polinomial di $x = {self.x}$:**\n\n"
        step2 += "$$\n\\begin{aligned}\n"
        step2 += f"P({self.x}) &= {custom_round(self.data[0][1])}"
        for i in range(0, self.n - 1):
            terms = " \\cdot ".join(
                [f"({self.x} - {self.data[j][0]})" for j in range(i + 1)]
            )
            step2 += f" + {custom_round(diff_table[0][i])} \\cdot {terms}"
        step2 += " \\\\\n"

        # Compute result
        res = self.data[0][1]
        for i in range(0, self.n - 1):
            term = diff_table[0][i]
            for j in range(0, i + 1):
                term *= self.x - self.data[j][0]
            res += term
        res = custom_round(res)

        step2 += f"&= {res}\n"
        step2 += "\\end{aligned}\n$$\n\n"
        steps.append(step2)

        return pd.DataFrame(rows), steps, res, err


class LagrangeInterpolation:
    def __init__(self, data, n=0, x=0):
        self.data = data
        self.n = n if n != 0 else len(self.data)
        self.x = x

    def solve(self):
        rows = []
        steps = []
        err = None

        res = 0.0
        step = f"**Interpolasi Lagrange di $x = {self.x}$:**\n\n"
        step += "$$\n\\begin{aligned}\n"

        for i in range(self.n):
            li = self.data[i][1]
            num_parts = []
            den_parts = []
            for j in range(self.n):
                if i != j:
                    li *= (self.x - self.data[j][0]) / (
                        self.data[i][0] - self.data[j][0]
                    )
                    num_parts.append(f"({self.x} - {self.data[j][0]})")
                    den_parts.append(f"({self.data[i][0]} - {self.data[j][0]})")
            li = custom_round(li)
            res += li

            rows.append(
                {
                    "i": i,
                    "x_i": self.data[i][0],
                    "y_i": self.data[i][1],
                    "L_i * y_i": li,
                }
            )

            num_str = " \\cdot ".join(num_parts)
            den_str = " \\cdot ".join(den_parts)
            step += f"L_{{{i}}} \\cdot y_{{{i}}} &= {self.data[i][1]} \\cdot \\frac{{{num_str}}}{{{den_str}}} = {li}"
            if i < self.n - 1:
                step += " \\\\\n"

        res = custom_round(res)
        step += f" \\\\[1em]\nP({self.x}) &= {res}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        return pd.DataFrame(rows), steps, res, err


class NewtonGregoryInterpolation:
    def __init__(self, data, n=0, x=0, x0=0, orde=-1, mode="forward"):
        self.data = data
        self.n = n if n != 0 else len(self.data)
        self.x = x
        self.x0 = x0 if x0 != 0 else self.data[0][0]
        self.orde = orde
        self.mode = mode

    def factorial(self, n):
        res = 1
        for i in range(2, n + 1):
            res *= i
        return res

    def search_x0(self):
        l, r = 0, self.n - 1
        while l <= r:
            mid = l + (r - l) // 2
            if self.data[mid][0] == self.x0:
                return mid
            elif self.data[mid][0] < self.x0:
                l = mid + 1
            else:
                r = mid - 1
        return -1

    def solve(self):
        rows = []
        steps = []
        err = None

        table = [[0 for _ in range(self.n - 1)] for _ in range(self.n)]
        idx = self.search_x0()
        if idx == -1:
            return pd.DataFrame(), [], None, "x0 tidak ada dalam data"

        h = self.data[1][0] - self.data[0][0]
        s = (self.x - self.data[idx][0]) / h

        for i in range(self.n - 1):
            table[i][0] = self.data[i + 1][1] - self.data[i][1]

        for i in range(1, self.n - 1):
            for j in range(self.n - i - 1):
                table[j][i] = table[j + 1][i - 1] - table[j][i - 1]

        # Round table
        for i in range(self.n):
            for j in range(self.n - 1):
                if table[i][j] != 0:
                    table[i][j] = custom_round(table[i][j])

        # Build rows
        for i in range(self.n):
            row = {"x": self.data[i][0], "y": self.data[i][1]}
            for j in range(self.n - 1):
                col_label = f"Δ{j+1}y"
                row[col_label] = table[i][j] if table[i][j] != 0 else ""
            rows.append(row)

        step = f"**Newton-Gregory ({self.mode}):**\n\n"
        step += f"$h = {custom_round(h)}$, $s = \\frac{{{self.x} - {self.data[idx][0]}}}{{{custom_round(h)}}} = {custom_round(s)}$\n\n"
        steps.append(step)

        match self.mode:
            case "forward":
                res = self.data[idx][1]
                max_terms = self.n - 1 - idx
                num_terms = (
                    self.orde
                    if self.orde != -1 and self.orde < max_terms
                    else max_terms
                )
                step2 = "$$\n\\begin{aligned}\n"
                step2 += f"P({self.x}) &= {custom_round(self.data[idx][1])}"
                for i in range(num_terms):
                    k = i + 1
                    temp = 1
                    for j in range(k):
                        temp *= s - j
                    term_val = custom_round(temp * table[idx][i] / self.factorial(k))
                    res = custom_round(res + term_val)
                    s_parts = " \\cdot ".join(
                        [
                            (
                                f"({custom_round(s)} - {j})"
                                if j > 0
                                else f"{custom_round(s)}"
                            )
                            for j in range(k)
                        ]
                    )
                    step2 += f" + \\frac{{{s_parts} \\cdot {custom_round(table[idx][i])}}}{{{k}!}}"
                step2 += f" \\\\\n&= {custom_round(res)}\n"
                step2 += "\\end{aligned}\n$$\n\n"
                steps.append(step2)

            case "backward":
                res = self.data[idx][1]
                step2 = "$$\n\\begin{aligned}\n"
                step2 += f"P({self.x}) &= {custom_round(self.data[idx][1])}"
                j = 0
                for i in range(idx - 1, -1, -1):
                    if self.orde != -1 and j >= self.orde:
                        break
                    k = j + 1
                    temp = 1
                    for mm in range(k):
                        temp *= s + mm
                    term_val = custom_round(temp * table[i][j] / self.factorial(k))
                    res = custom_round(res + term_val)
                    s_parts = " \\cdot ".join(
                        [
                            (
                                f"({custom_round(s)} + {mm})"
                                if mm > 0
                                else f"{custom_round(s)}"
                            )
                            for mm in range(k)
                        ]
                    )
                    step2 += f" + \\frac{{{s_parts} \\cdot {custom_round(table[i][j])}}}{{{k}!}}"
                    j += 1
                step2 += f" \\\\\n&= {custom_round(res)}\n"
                step2 += "\\end{aligned}\n$$\n\n"
                steps.append(step2)

        return pd.DataFrame(rows), steps, custom_round(res), err


class StirlingInterpolation:
    def __init__(self, data, n=0, x=0, x0=0, orde=-1):
        self.data = data
        self.n = n if n != 0 else len(self.data)
        self.x = x
        self.x0 = x0 if x0 != 0 else self.data[self.n // 2][0]
        self.orde = orde

    def factorial(self, n):
        res = 1
        for i in range(2, n + 1):
            res *= i
        return res

    def search_x0(self):
        for i in range(self.n):
            if self.data[i][0] == self.x0:
                return i
        return -1

    def solve(self):
        rows = []
        steps = []
        err = None

        table = [[0 for _ in range(self.n - 1)] for _ in range(self.n)]
        idx = self.search_x0()
        if idx == -1:
            return pd.DataFrame(), [], None, "x0 tidak ada dalam data"

        h = self.data[1][0] - self.data[0][0]
        s = (self.x - self.data[idx][0]) / h

        for i in range(self.n - 1):
            table[i][0] = self.data[i + 1][1] - self.data[i][1]

        for i in range(1, self.n - 1):
            for j in range(self.n - i - 1):
                table[j][i] = table[j + 1][i - 1] - table[j][i - 1]

        for i in range(self.n):
            for j in range(self.n - 1):
                if table[i][j] != 0:
                    table[i][j] = custom_round(table[i][j])

        for i in range(self.n):
            row = {"x": self.data[i][0], "y": self.data[i][1]}
            for j in range(self.n - 1):
                col_label = f"Δ{j+1}y"
                row[col_label] = table[i][j] if table[i][j] != 0 else ""
            rows.append(row)

        step = f"**Interpolasi Stirling:**\n\n"
        step += f"$h = {custom_round(h)}$, $s = \\frac{{{self.x} - {self.data[idx][0]}}}{{{custom_round(h)}}} = {custom_round(s)}$\n\n"
        steps.append(step)

        res = self.data[idx][1]
        step2 = "$$\n\\begin{aligned}\n"
        step2 += f"P({self.x}) &= {custom_round(self.data[idx][1])}"

        max_orde = self.orde if self.orde != -1 else self.n - 1
        k = 1
        while k <= max_orde:
            if k % 2 == 1:
                # Odd order: average of forward and backward differences
                fwd_idx = idx - (k + 1) // 2
                bwd_idx = idx - (k - 1) // 2
                if fwd_idx < 0 or bwd_idx < 0 or k - 1 >= self.n - 1:
                    break
                if fwd_idx >= self.n or bwd_idx >= self.n:
                    break
                avg_diff = custom_round(
                    (table[fwd_idx][k - 1] + table[bwd_idx][k - 1]) / 2
                )
                # Product: s * (s^2 - 1^2) * (s^2 - 2^2) * ... for odd k
                temp = s
                for j in range(1, (k + 1) // 2):
                    temp *= s**2 - j**2
                temp = custom_round(temp)
                term_val = custom_round(temp * avg_diff / self.factorial(k))
                res = custom_round(res + term_val)
                step2 += f" + \\frac{{{temp} \\cdot {avg_diff}}}{{{k}!}}"
            else:
                # Even order: central difference
                c_idx = idx - k // 2
                if c_idx < 0 or k - 1 >= self.n - 1:
                    break
                diff_val = table[c_idx][k - 1]
                # Product: s^2 * (s^2 - 1^2) * ... * (s^2 - ((k/2)-1)^2)
                temp = s**2
                for j in range(1, k // 2):
                    temp *= s**2 - j**2
                temp = custom_round(temp)
                term_val = custom_round(temp * diff_val / self.factorial(k))
                res = custom_round(res + term_val)
                step2 += f" + \\frac{{{temp} \\cdot {custom_round(diff_val)}}}{{{k}!}}"
            k += 1

        res = custom_round(res)
        step2 += f" \\\\\n&= {res}\n"
        step2 += "\\end{aligned}\n$$\n\n"
        steps.append(step2)

        return pd.DataFrame(rows), steps, res, err


class BesselInterpolation:
    def __init__(self, data, n=0, x=0, x0=0, orde=-1):
        self.data = data
        self.n = n if n != 0 else len(self.data)
        self.x = x
        self.x0 = x0 if x0 != 0 else self.data[self.n // 2][0]
        self.orde = orde

    def factorial(self, n):
        res = 1
        for i in range(2, n + 1):
            res *= i
        return res

    def search_x0(self):
        for i in range(self.n):
            if self.data[i][0] == self.x0:
                return i
        return -1

    def solve(self):
        rows = []
        steps = []
        err = None

        table = [[0 for _ in range(self.n - 1)] for _ in range(self.n)]
        idx = self.search_x0()
        if idx == -1:
            return pd.DataFrame(), [], None, "x0 tidak ada dalam data"
        if idx + 1 >= self.n:
            return (
                pd.DataFrame(),
                [],
                None,
                "x0 harus bukan titik terakhir untuk Bessel",
            )

        h = self.data[1][0] - self.data[0][0]
        s = (self.x - self.data[idx][0]) / h
        u = s - 0.5

        for i in range(self.n - 1):
            table[i][0] = self.data[i + 1][1] - self.data[i][1]

        for i in range(1, self.n - 1):
            for j in range(self.n - i - 1):
                table[j][i] = table[j + 1][i - 1] - table[j][i - 1]

        for i in range(self.n):
            for j in range(self.n - 1):
                if table[i][j] != 0:
                    table[i][j] = custom_round(table[i][j])

        for i in range(self.n):
            row = {"x": self.data[i][0], "y": self.data[i][1]}
            for j in range(self.n - 1):
                col_label = f"Δ{j+1}y"
                row[col_label] = table[i][j] if table[i][j] != 0 else ""
            rows.append(row)

        step = f"**Interpolasi Bessel:**\n\n"
        step += f"$h = {custom_round(h)}$, $s = {custom_round(s)}$, $u = s - 0.5 = {custom_round(u)}$\n\n"
        steps.append(step)

        # Bessel starts with average of f[idx] and f[idx+1]
        res = custom_round((self.data[idx][1] + self.data[idx + 1][1]) / 2)
        step2 = "$$\n\\begin{aligned}\n"
        step2 += f"P({self.x}) &= \\frac{{f_0 + f_1}}{{2}}"

        max_orde = self.orde if self.orde != -1 else self.n - 1
        k = 1
        while k <= max_orde:
            if k % 2 == 1:
                # Odd order: u * product * diff / k!
                if k - 1 >= self.n - 1:
                    break
                if k == 1:
                    diff_val = table[idx][0]
                else:
                    c1 = idx - k // 2
                    c2 = c1 - 1
                    if c1 < 0 or c2 < 0 or k - 1 >= self.n - 1:
                        break
                    diff_val = custom_round((table[c2][k - 1] + table[c1][k - 1]) / 2)
                if k == 1:
                    temp = u
                else:
                    temp = u
                    for j in range(1, (k + 1) // 2):
                        temp *= u**2 - (j - 0.5) ** 2
                temp = custom_round(temp)
                term_val = custom_round(temp * diff_val / self.factorial(k))
                res = custom_round(res + term_val)
                step2 += f" + \\frac{{{temp} \\cdot {custom_round(diff_val)}}}{{{k}!}}"
            else:
                # Even order: average of two central differences * product / k!
                c_idx = idx - k // 2
                c_idx2 = c_idx + 1
                if c_idx < 0 or c_idx2 >= self.n or k - 1 >= self.n - 1:
                    break
                avg_diff = custom_round(
                    (table[c_idx][k - 1] + table[c_idx2][k - 1]) / 2
                )
                temp = 1.0
                for j in range(k // 2):
                    temp *= u**2 - j**2
                temp = custom_round(temp)
                term_val = custom_round(temp * avg_diff / self.factorial(k))
                res = custom_round(res + term_val)
                step2 += f" + \\frac{{{temp} \\cdot {custom_round(avg_diff)}}}{{{k}!}}"
            k += 1

        res = custom_round(res)
        step2 += f" \\\\\n&= {res}\n"
        step2 += "\\end{aligned}\n$$\n\n"
        steps.append(step2)

        return pd.DataFrame(rows), steps, res, err


class NewtonGregoryDifferentiation:
    def __init__(self, data, n=0, x=0, x0=0, orde=-1, mode="forward"):
        self.data = data
        self.n = n if n != 0 else len(self.data)
        self.x = x
        self.x0 = x0 if x0 != 0 else self.data[0][0]
        self.orde = orde
        self.mode = mode

    def factorial(self, n):
        res = 1
        for i in range(2, n + 1):
            res *= i
        return res

    def search_x0(self):
        l, r = 0, self.n - 1
        while l <= r:
            mid = l + (r - l) // 2
            if self.data[mid][0] == self.x0:
                return mid
            elif self.data[mid][0] < self.x0:
                l = mid + 1
            else:
                r = mid - 1
        return -1

    def _product_derivative(self, s, k):
        """Derivative of product (s)(s-1)(s-2)...(s-(k-1)) w.r.t. s."""
        if k == 0:
            return 0
        if k == 1:
            return 1
        total = 0
        for i in range(k):
            prod = 1
            for j in range(k):
                if j != i:
                    prod *= s - j
            total += prod
        return total

    def _product_derivative_backward(self, s, k):
        """Derivative of product (s)(s+1)(s+2)...(s+(k-1)) w.r.t. s."""
        if k == 0:
            return 0
        if k == 1:
            return 1
        total = 0
        for i in range(k):
            prod = 1
            for j in range(k):
                if j != i:
                    prod *= s + j
            total += prod
        return total

    def solve(self):
        rows = []
        steps = []
        err = None

        table = [[0 for _ in range(self.n - 1)] for _ in range(self.n)]
        idx = self.search_x0()
        if idx == -1:
            return pd.DataFrame(), [], None, "x0 tidak ada dalam data"

        h = self.data[1][0] - self.data[0][0]
        s = (self.x - self.data[idx][0]) / h

        for i in range(self.n - 1):
            table[i][0] = self.data[i + 1][1] - self.data[i][1]

        for i in range(1, self.n - 1):
            for j in range(self.n - i - 1):
                table[j][i] = table[j + 1][i - 1] - table[j][i - 1]

        for i in range(self.n):
            for j in range(self.n - 1):
                if table[i][j] != 0:
                    table[i][j] = custom_round(table[i][j])

        for i in range(self.n):
            row = {"x": self.data[i][0], "y": self.data[i][1]}
            for j in range(self.n - 1):
                col_label = f"Δ{j+1}y"
                row[col_label] = table[i][j] if table[i][j] != 0 else ""
            rows.append(row)

        step = f"**Diferensiasi Newton-Gregory ({self.mode}):**\n\n"
        step += f"$h = {custom_round(h)}$, $s = {custom_round(s)}$\n\n"
        steps.append(step)

        res_deriv_x = 0

        match self.mode:
            case "forward":
                res_deriv_s = 0
                max_terms = self.n - 1 - idx
                num_terms = (
                    self.orde
                    if self.orde != -1 and self.orde < max_terms
                    else max_terms
                )

                step2 = "$$\n\\begin{aligned}\n"
                step2 += f"P'({self.x}) &= \\frac{{1}}{{h}} \\sum \\frac{{\\Delta^k y_0}}{{k!}} \\cdot \\frac{{d}}{{ds}} \\prod_{{j=0}}^{{k-1}} (s - j) \\\\\n"

                for i in range(num_terms):
                    k = i + 1
                    deriv_prod = self._product_derivative(s, k)
                    term_val = custom_round(
                        deriv_prod * table[idx][i] / self.factorial(k)
                    )
                    res_deriv_s = custom_round(res_deriv_s + term_val)

                res_deriv_x = custom_round(res_deriv_s / h)
                step2 += f"&= \\frac{{1}}{{{custom_round(h)}}} \\cdot {custom_round(res_deriv_s)} = {res_deriv_x}\n"
                step2 += "\\end{aligned}\n$$\n\n"
                steps.append(step2)

            case "backward":
                res_deriv_s = 0
                step2 = "$$\n\\begin{aligned}\n"
                step2 += f"P'({self.x}) &= \\frac{{1}}{{h}} \\sum \\frac{{\\Delta^k y}}{{k!}} \\cdot \\frac{{d}}{{ds}} \\prod_{{j=0}}^{{k-1}} (s + j) \\\\\n"

                j = 0
                for i in range(idx - 1, -1, -1):
                    if self.orde != -1 and j >= self.orde:
                        break
                    k = j + 1
                    deriv_prod = self._product_derivative_backward(s, k)
                    term_val = custom_round(
                        deriv_prod * table[i][j] / self.factorial(k)
                    )
                    res_deriv_s = custom_round(res_deriv_s + term_val)
                    j += 1

                res_deriv_x = custom_round(res_deriv_s / h)
                step2 += f"&= \\frac{{1}}{{{custom_round(h)}}} \\cdot {custom_round(res_deriv_s)} = {res_deriv_x}\n"
                step2 += "\\end{aligned}\n$$\n\n"
                steps.append(step2)

        return pd.DataFrame(rows), steps, custom_round(res_deriv_x), err


class LagrangeDifferentiation:
    def __init__(self, data, n=0, x=0):
        self.data = data
        self.n = n if n != 0 else len(self.data)
        self.x = x

    def _div(self, i, x_val):
        """Derivative of L_i(x) at x_val."""
        result = 0.0
        for k in range(self.n):
            if k == i:
                continue
            prod = 1.0
            for j in range(self.n):
                if j == i or j == k:
                    continue
                prod *= (x_val - self.data[j][0]) / (self.data[i][0] - self.data[j][0])
            prod /= self.data[i][0] - self.data[k][0]
            result += prod
        return result

    def solve(self):
        rows = []
        steps = []
        err = None

        res = 0.0
        step = f"**Diferensiasi Lagrange di $x = {self.x}$:**\n\n"
        step += "$$\n\\begin{aligned}\n"

        for i in range(self.n):
            dl_i = custom_round(self._div(i, self.x))
            term = custom_round(self.data[i][1] * dl_i)
            res = custom_round(res + term)

            rows.append(
                {
                    "i": i,
                    "x_i": self.data[i][0],
                    "y_i": self.data[i][1],
                    "L'_i(x)": dl_i,
                    "y_i * L'_i(x)": term,
                }
            )

            step += f"L'_{{{i}}}({self.x}) &= {dl_i}, \\quad y_{{{i}}} \\cdot L'_{{{i}}} = {term}"
            if i < self.n - 1:
                step += " \\\\\n"

        res = custom_round(res)
        step += f" \\\\[1em]\nP'({self.x}) &= {res}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        return pd.DataFrame(rows), steps, res, err


class Integration:
    def __init__(self, f, a, b):
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.a = a
        self.b = b

    def solve(self):
        rows = []
        steps = []
        err = None

        if self.a > self.b:
            self.a, self.b = self.b, self.a

        try:
            L = float(sp.integrate(self.f, (self.x, self.a, self.b)))
            L = custom_round(L)
        except Exception as e:
            return pd.DataFrame(), [], None, f"Error saat integrasi: {str(e)}"

        rows.append({"f(x)": str(self.f), "a": self.a, "b": self.b, "Hasil": L})

        step = f"**Integrasi Analitik:**\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"\\int_{{{self.a}}}^{{{self.b}}} {sp.latex(self.f)} \\, dx &= {L}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        return pd.DataFrame(rows), steps, L, err


class TrapezoidalIntegration:
    def __init__(self, f, a, b, n=0, true_val=-1):
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.a = a
        self.b = b
        self.n = n
        self.true_val = true_val

    def evaluate(self, val):
        return float(self.f.subs(self.x, val))

    def solve(self):
        rows = []
        steps = []
        err = None

        if self.a > self.b:
            self.a, self.b = self.b, self.a

        if self.n > 0:
            # Multi-segment trapezoidal
            h = custom_round((self.b - self.a) / self.n)
            total = 0
            step = f"**Trapezoidal Multi-Segmen (n={self.n}):**\n\n"
            step += f"$h = \\frac{{{self.b} - {self.a}}}{{{self.n}}} = {h}$\n\n"

            step += "$$\n\\begin{aligned}\n"
            step += f"I &= \\frac{{h}}{{2}} \\left[ f(x_0) + 2\\sum_{{i=1}}^{{n-1}} f(x_i) + f(x_n) \\right] \\\\\n"

            for i in range(self.n + 1):
                xi = custom_round(self.a + i * h)
                fi = custom_round(self.evaluate(xi))
                if i == 0 or i == self.n:
                    total += fi
                else:
                    total += 2 * fi
                rows.append({"i": i, "x_i": xi, "f(x_i)": fi})

            L = custom_round(h / 2 * total)
            step += f"&= \\frac{{{h}}}{{2}} \\cdot {custom_round(total)} = {L}\n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)
        else:
            # Single segment trapezoidal
            fa = custom_round(self.evaluate(self.a))
            fb = custom_round(self.evaluate(self.b))
            L = custom_round((self.b - self.a) * (fa + fb) / 2)

            rows.append({"x": self.a, "f(x)": fa})
            rows.append({"x": self.b, "f(x)": fb})

            step = "**Trapezoidal (Segmen Tunggal):**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"I &= \\frac{{b - a}}{{2}} \\left[ f(a) + f(b) \\right] \\\\\n"
            step += f"&= \\frac{{{self.b} - {self.a}}}{{2}} \\left[ {fa} + {fb} \\right] = {L}\n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

        if self.true_val != -1:
            et = (
                custom_round(abs((self.true_val - L) / self.true_val) * 100)
                if self.true_val != 0
                else custom_round(abs(L) * 100)
            )
            steps.append(f"$E_t = {et}\\%$")

        return pd.DataFrame(rows), steps, L, err


class Simpson13Integration:
    def __init__(self, f, a, b, n=0, true_val=-1):
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.a = a
        self.b = b
        self.n = n
        self.true_val = true_val

    def evaluate(self, val):
        return float(self.f.subs(self.x, val))

    def solve(self):
        rows = []
        steps = []
        err = None

        if self.a > self.b:
            self.a, self.b = self.b, self.a

        if self.n > 0:
            # Multi-segment Simpson 1/3
            h = custom_round((self.b - self.a) / self.n)
            total = 0

            step = f"**Simpson 1/3 Multi-Segmen (n={self.n}):**\n\n"
            step += f"$h = \\frac{{{self.b} - {self.a}}}{{{self.n}}} = {h}$\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"I &= \\frac{{h}}{{3}} \\left[ f(x_0) + 4\\sum_{{\\text{{ganjil}}}} f(x_i) + 2\\sum_{{\\text{{genap}}}} f(x_i) + f(x_n) \\right] \\\\\n"

            for i in range(self.n + 1):
                xi = custom_round(self.a + i * h)
                fi = custom_round(self.evaluate(xi))
                if i == 0 or i == self.n:
                    total += fi
                elif i % 2 == 1:
                    total += 4 * fi
                else:
                    total += 2 * fi
                rows.append({"i": i, "x_i": xi, "f(x_i)": fi})

            L = custom_round(h / 3 * total)
            step += f"&= \\frac{{{h}}}{{3}} \\cdot {custom_round(total)} = {L}\n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)
        else:
            # Single Simpson 1/3
            h = custom_round((self.b - self.a) / 2)
            x0 = self.a
            x1 = custom_round(self.a + h)
            x2 = self.b

            f0 = custom_round(self.evaluate(x0))
            f1 = custom_round(self.evaluate(x1))
            f2 = custom_round(self.evaluate(x2))

            L = custom_round(h / 3 * (f0 + 4 * f1 + f2))

            rows.append({"x": x0, "f(x)": f0})
            rows.append({"x": x1, "f(x)": f1})
            rows.append({"x": x2, "f(x)": f2})

            step = "**Simpson 1/3 (Segmen Tunggal):**\n\n"
            step += f"$h = \\frac{{{self.b} - {self.a}}}{{2}} = {h}$\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"I &= \\frac{{h}}{{3}} \\left[ f(x_0) + 4f(x_1) + f(x_2) \\right] \\\\\n"
            step += f"&= \\frac{{{h}}}{{3}} \\left[ {f0} + 4 \\cdot {f1} + {f2} \\right] = {L}\n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

        if self.true_val != -1:
            et = (
                custom_round(abs((self.true_val - L) / self.true_val) * 100)
                if self.true_val != 0
                else custom_round(abs(L) * 100)
            )
            steps.append(f"$E_t = {et}\\%$")

        return pd.DataFrame(rows), steps, L, err


class Simpson38Integration:
    def __init__(self, f, a, b, true_val=-1):
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.a = a
        self.b = b
        self.true_val = true_val

    def evaluate(self, val):
        return float(self.f.subs(self.x, val))

    def solve(self):
        rows = []
        steps = []
        err = None

        if self.a > self.b:
            self.a, self.b = self.b, self.a

        h = custom_round((self.b - self.a) / 3)
        x0 = self.a
        x1 = custom_round(self.a + h)
        x2 = custom_round(self.a + 2 * h)
        x3 = self.b

        f0 = custom_round(self.evaluate(x0))
        f1 = custom_round(self.evaluate(x1))
        f2 = custom_round(self.evaluate(x2))
        f3 = custom_round(self.evaluate(x3))

        L = custom_round(3 * h / 8 * (f0 + 3 * f1 + 3 * f2 + f3))

        rows.append({"x": x0, "f(x)": f0})
        rows.append({"x": x1, "f(x)": f1})
        rows.append({"x": x2, "f(x)": f2})
        rows.append({"x": x3, "f(x)": f3})

        step = "**Simpson 3/8:**\n\n"
        step += f"$h = \\frac{{{self.b} - {self.a}}}{{3}} = {h}$\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"I &= \\frac{{3h}}{{8}} \\left[ f(x_0) + 3f(x_1) + 3f(x_2) + f(x_3) \\right] \\\\\n"
        step += f"&= \\frac{{3 \\cdot {h}}}{{8}} \\left[ {f0} + 3 \\cdot {f1} + 3 \\cdot {f2} + {f3} \\right] = {L}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        if self.true_val != -1:
            et = (
                custom_round(abs((self.true_val - L) / self.true_val) * 100)
                if self.true_val != 0
                else custom_round(abs(L) * 100)
            )
            steps.append(f"$E_t = {et}\\%$")

        return pd.DataFrame(rows), steps, L, err


class RiemannIntegration:
    def __init__(self, f, a, b, n, true_val=-1):
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.a = a
        self.b = b
        self.n = n
        self.true_val = true_val

    def evaluate(self, val):
        return float(self.f.subs(self.x, val))

    def solve(self):
        rows = []
        steps = []
        err = None

        if self.a > self.b:
            self.a, self.b = self.b, self.a

        h = custom_round((self.b - self.a) / self.n)
        total = 0

        step = f"**Riemann (Left Sum, n={self.n}):**\n\n"
        step += f"$h = \\frac{{{self.b} - {self.a}}}{{{self.n}}} = {h}$\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"I &= h \\sum_{{i=0}}^{{n-1}} f(x_i) \\\\\n"

        for i in range(self.n):
            xi = custom_round(self.a + i * h)
            fi = custom_round(self.evaluate(xi))
            total = custom_round(total + fi)
            rows.append({"i": i, "x_i": xi, "f(x_i)": fi})

        L = custom_round(h * total)
        step += f"&= {h} \\cdot {total} = {L}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        if self.true_val != -1:
            et = (
                custom_round(abs((self.true_val - L) / self.true_val) * 100)
                if self.true_val != 0
                else custom_round(abs(L) * 100)
            )
            steps.append(f"$E_t = {et}\\%$")

        return pd.DataFrame(rows), steps, L, err


class GaussIntegration:
    def __init__(self, f, a, b, true_val=-1):
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.a = a
        self.b = b
        self.true_val = true_val

    def evaluate(self, val):
        return float(self.f.subs(self.x, val))

    def solve(self):
        rows = []
        steps = []
        err = None

        if self.a > self.b:
            self.a, self.b = self.b, self.a

        # 2-point Gauss quadrature on [-1, 1]
        t1 = custom_round(-1 / np.sqrt(3))
        t2 = custom_round(1 / np.sqrt(3))
        w1 = 1
        w2 = 1

        # Transform to [a, b]
        c1 = custom_round((self.b - self.a) / 2)
        c2 = custom_round((self.a + self.b) / 2)

        x1 = custom_round(c1 * t1 + c2)
        x2 = custom_round(c1 * t2 + c2)

        f1 = custom_round(self.evaluate(x1))
        f2 = custom_round(self.evaluate(x2))

        L = custom_round(c1 * (w1 * f1 + w2 * f2))

        rows.append({"t": t1, "x": x1, "f(x)": f1, "w": w1})
        rows.append({"t": t2, "x": x2, "f(x)": f2, "w": w2})

        step = "**Gauss Quadrature (2-point):**\n\n"
        step += f"Transformasi: $x = \\frac{{{self.b} - {self.a}}}{{2}} t + \\frac{{{self.a} + {self.b}}}{{2}} = {c1} t + {c2}$\n\n"
        step += f"Titik: $t_1 = {t1}$, $t_2 = {t2}$\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"x_1 &= {c1} \\cdot ({t1}) + {c2} = {x1}, \\quad f(x_1) = {f1} \\\\\n"
        step += (
            f"x_2 &= {c1} \\cdot ({t2}) + {c2} = {x2}, \\quad f(x_2) = {f2} \\\\[1em]\n"
        )
        step += f"I &= \\frac{{b-a}}{{2}} [w_1 f(x_1) + w_2 f(x_2)] \\\\\n"
        step += f"&= {c1} \\cdot [{w1} \\cdot {f1} + {w2} \\cdot {f2}] = {L}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        if self.true_val != -1:
            et = (
                custom_round(abs((self.true_val - L) / self.true_val) * 100)
                if self.true_val != 0
                else custom_round(abs(L) * 100)
            )
            steps.append(f"$E_t = {et}\\%$")

        return pd.DataFrame(rows), steps, L, err


class Euler:
    def __init__(self, df, a, b, h):
        self.x = sp.symbols("x")
        self.df_expr = sp.sympify(df)
        self.a = a
        self.b = b
        self.h = h

    def evaluate(self, val):
        return float(self.df_expr.subs(self.x, val))

    def solve(self):
        rows = []
        steps = []
        err = None

        # True value from integration
        try:
            true_integral = float(sp.integrate(self.df_expr, (self.x, self.a, self.b)))
        except Exception:
            true_integral = None

        n_steps = int((self.b - self.a) / self.h)
        y = 0.0
        xi = self.a

        step_header = f"**Metode Euler:**\n\n$h = {self.h}$\n\n"
        steps.append(step_header)

        for i in range(n_steps):
            fi = custom_round(self.evaluate(xi))
            y_old = custom_round(y)
            y = custom_round(y + fi * self.h)

            et_val = ""
            if true_integral is not None and true_integral != 0:
                et_val = custom_round(abs((true_integral - y) / true_integral) * 100)
            elif true_integral is not None:
                et_val = custom_round(abs(y) * 100)

            rows.append(
                {
                    "i": i,
                    "x_i": custom_round(xi),
                    "f(x_i)": fi,
                    "y_i": y,
                    "Et (%)": et_val,
                }
            )

            step = f"**Langkah {i + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"f({custom_round(xi)}) &= {fi} \\\\\n"
            step += f"y_{{{i+1}}} &= {y_old} + {fi} \\cdot {self.h} = {y}\n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

            xi = custom_round(xi + self.h)

        if true_integral is not None:
            steps.append(f"Nilai sejati (integrasi): ${custom_round(true_integral)}$")

        return pd.DataFrame(rows), steps, custom_round(y), err


class Heunn:
    def __init__(self, df, a, b, h):
        self.x = sp.symbols("x")
        self.df_expr = sp.sympify(df)
        self.a = a
        self.b = b
        self.h = h

    def evaluate(self, val):
        return float(self.df_expr.subs(self.x, val))

    def solve(self):
        rows = []
        steps = []
        err = None

        # True value from integration
        try:
            true_integral = float(sp.integrate(self.df_expr, (self.x, self.a, self.b)))
        except Exception:
            true_integral = None

        n_steps = int((self.b - self.a) / self.h)
        y = 0.0
        xi = self.a

        step_header = f"**Metode Heun:**\n\n$h = {self.h}$\n\n"
        steps.append(step_header)

        for i in range(n_steps):
            fi = custom_round(self.evaluate(xi))
            fi_next = custom_round(self.evaluate(xi + self.h))
            y_old = custom_round(y)
            y = custom_round(y + (fi + fi_next) / 2 * self.h)

            et_val = ""
            if true_integral is not None and true_integral != 0:
                et_val = custom_round(abs((true_integral - y) / true_integral) * 100)
            elif true_integral is not None:
                et_val = custom_round(abs(y) * 100)

            rows.append(
                {
                    "i": i,
                    "x_i": custom_round(xi),
                    "f(x_i)": fi,
                    "f(x_i+h)": fi_next,
                    "y_i": y,
                    "Et (%)": et_val,
                }
            )

            step = f"**Langkah {i + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"f({custom_round(xi)}) &= {fi} \\\\\n"
            step += f"f({custom_round(xi + self.h)}) &= {fi_next} \\\\\n"
            step += f"y_{{{i+1}}} &= {y_old} + \\frac{{{fi} + {fi_next}}}{{2}} \\cdot {self.h} = {y}\n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

            xi = custom_round(xi + self.h)

        if true_integral is not None:
            steps.append(f"Nilai sejati (integrasi): ${custom_round(true_integral)}$")

        return pd.DataFrame(rows), steps, custom_round(y), err


class RungeKutta:
    def __init__(self, df, a, b, h, a2):
        self.x = sp.symbols("x")
        self.df_expr = sp.sympify(df)
        self.a = a
        self.b = b
        self.h = h
        self.a2 = a2

    def evaluate(self, val):
        return float(self.df_expr.subs(self.x, val))

    def solve(self):
        rows = []
        steps = []
        err = None

        a1 = custom_round(1 - self.a2)
        p = custom_round(1 / (2 * self.a2))
        q = p

        # True value from integration
        try:
            true_integral = float(sp.integrate(self.df_expr, (self.x, self.a, self.b)))
        except Exception:
            true_integral = None

        n_steps = int((self.b - self.a) / self.h)
        y = 0.0
        xi = self.a

        step_header = f"**Metode Runge-Kutta:**\n\n"
        step_header += f"$a_2 = {self.a2}$, $a_1 = 1 - a_2 = {a1}$, $p = q = \\frac{{1}}{{2a_2}} = {p}$\n\n"
        steps.append(step_header)

        for i in range(n_steps):
            k1 = custom_round(self.evaluate(xi))
            k2 = custom_round(self.evaluate(xi + p * self.h))
            y_old = custom_round(y)
            y = custom_round(y + (a1 * k1 + self.a2 * k2) * self.h)

            et_val = ""
            if true_integral is not None and true_integral != 0:
                et_val = custom_round(abs((true_integral - y) / true_integral) * 100)
            elif true_integral is not None:
                et_val = custom_round(abs(y) * 100)

            rows.append(
                {
                    "i": i,
                    "x_i": custom_round(xi),
                    "k1": k1,
                    "k2": k2,
                    "y_i": y,
                    "Et (%)": et_val,
                }
            )

            step = f"**Langkah {i + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"k_1 &= f({custom_round(xi)}) = {k1} \\\\\n"
            step += f"k_2 &= f({custom_round(xi)} + {p} \\cdot {self.h}) = f({custom_round(xi + p * self.h)}) = {k2} \\\\\n"
            step += f"y_{{{i+1}}} &= {y_old} + ({a1} \\cdot {k1} + {self.a2} \\cdot {k2}) \\cdot {self.h} = {y}\n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

            xi = custom_round(xi + self.h)

        if true_integral is not None:
            steps.append(f"Nilai sejati (integrasi): ${custom_round(true_integral)}$")

        return pd.DataFrame(rows), steps, custom_round(y), err
