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


class PolynomFactorization:
    def __init__(self, f: str, max_iter: int = 10) -> None:
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.max_iter = max_iter

    def ABC(self, a: float = 1, b: float = 1, c: float = 1):
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

        match degree:
            case 2:
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
            case 3:
                A3, A2, A1, A0 = coeff
                if A3 != 1:
                    return None, None, None, f"Koefisien $x^3$ harus 1"

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
a_0 &= {A1} - ({a1} \\cdot ({b0})) = {a0}
\\end{{aligned}}
$$
"""
                    )

                x1 = -1 * b0
                x2, x3 = self.ABC(b=a1, c=a0)

                roots = (x1, x2, x3)
                steps.append(
                    f"### Akar-akar polinomial: {', '.join(str(f'$x_{i+1} = {root}$') for i, root in enumerate(roots) if not np.isnan(root))}"
                )

                return rows, steps, roots, None
            case 4:
                A4, A3, A2, A1, A0 = coeff
                if A4 != 1:
                    return None, None, None, f"Koefisien $x^4$ harus 1"

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
b_1 &= \\frac{{{A1} - ({a1} \\cdot ({b0}))}}{{{a0_old}}} = {b1} \\\\
a_1 &= {A3} - {b1} = {a1} \\\\
a_0 &= {A2} - {b0} - ({a1} \\cdot ({b1})) = {a0}
\\end{{aligned}}
$$
"""
                    )

                x1, x2 = self.ABC(b=b1, c=b0)
                x3, x4 = self.ABC(b=a1, c=a0)
                roots = (x1, x2, x3, x4)
                steps.append(
                    f"### Akar-akar polinomial: {', '.join(str(f'$x_{i+1} = {root}$') for i, root in enumerate(roots) if not np.isnan(root))}"
                )
                return rows, steps, roots, None
            case 5:
                A5, A4, A3, A2, A1, A0 = coeff
                if A5 != 1:
                    return None, None, None, f"Koefisien $x^5$ harus 1"

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

                roots = (x1, x2, x3, x4, x5)
                steps.append(
                    f"### Akar-akar polinomial: {', '.join(str(f'$x_{i+1} = {root}$') for i, root in enumerate(roots) if not np.isnan(root))}"
                )

                return rows, steps, roots, None
            case _:
                return None, None, None, f"Derajat {degree} tidak didukung"