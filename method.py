import numpy as np
import pandas as pd
import sympy as sp

from utils import Ea, Et, custom_round, sign


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
        coeff = [c for c in coeff]

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

            if A3 != 1:
                return (
                    None,
                    None,
                    None,
                    f"**Error**: Koefisien pada derajat {degree} harus bernilai **1**",
                )

            rows = []
            steps = []

            b0 = 0
            a1 = A2
            a0 = A1

            rows.append({"Iterasi": 0, "b0": b0, "a1": a1, "a0": a0})
            steps.append(f"""
**Iterasi 0:**

$$
\\begin{{aligned}}
b_0 &= 0 \\\\
a_1 &= {A2} \\\\
a_0 &= {A1}
\\end{{aligned}}
$$
""")

            for i in range(1, self.max_iter):
                a0_old = a0

                b0 = custom_round(A0 / a0_old)
                a1 = custom_round(A2 - b0)
                a0 = custom_round(A1 - a1 * b0)

                rows.append({"Iterasi": i, "b0": b0, "a1": a1, "a0": a0})
                steps.append(f"""
**Iterasi {i}:**

$$
\\begin{{aligned}}
b_0 &= \\frac{{{A0}}}{{{a0_old}}} = {b0} \\\\
a_1 &= {A2} - ({b0}) = {a1} \\\\
a_0 &= {A1} - ({a1} \\cdot ({b0})) = {a0}
\\end{{aligned}}
$$
""")

            x1 = -1 * b0
            x2, x3 = self.ABC(b=a1, c=a0)

            return rows, steps, (x1, x2, x3), None

        elif degree == 4:
            A4, A3, A2, A1, A0 = coeff

            if A4 != 1:
                return (
                    None,
                    None,
                    None,
                    f"**Error**: Koefisien pada derajat {degree} harus bernilai **1**",
                )

            rows = []
            steps = []

            b1 = 0
            b0 = 0
            a1 = A3
            a0 = A2

            rows.append({"Iterasi": 0, "b0": b0, "b1": b1, "a1": a1, "a0": a0})
            steps.append(f"""
**Iterasi 0:**

$$
\\begin{{aligned}}
b_1 &= 0 \\\\
b_0 &= 0 \\\\
a_1 &= {A3} \\\\
a_0 &= {A2}
\\end{{aligned}}
$$
""")

            for i in range(1, self.max_iter):
                a0_old = a0

                b0 = custom_round(A0 / a0_old)
                b1 = custom_round((A1 - a1 * b0) / a0_old)
                a1 = custom_round(A3 - b1)
                a0 = custom_round(A2 - b0 - a1 * b1)

                rows.append({"Iterasi": i, "b0": b0, "b1": b1, "a1": a1, "a0": a0})
                steps.append(f"""
**Iterasi {i}:**

$$
\\begin{{aligned}}
b_0 &= \\frac{{{A0}}}{{{a0_old}}} = {b0} \\\\
b_1 &= \\frac{{{A1} - ({a1} \\cdot ({b0}))}}{{{a0_old}}} = {b1} \\\\
a_1 &= {A3} - ({b1}) = {a1} \\\\
a_0 &= {A2} - ({b0}) - ({a1} \\cdot ({b1})) = {a0}
\\end{{aligned}}
$$
""")

            x1, x2 = self.ABC(b=b1, c=b0)
            x3, x4 = self.ABC(b=a1, c=a0)

            return rows, steps, (x1, x2, x3, x4), None

        elif degree == 5:
            A5, A4, A3, A2, A1, A0 = coeff

            if A5 != 1:
                return (
                    None,
                    None,
                    None,
                    f"**Error**: Koefisien pada derajat {degree} harus bernilai **1**",
                )

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
            steps.append(f"""
**Iterasi 0:**

$$
\\begin{{aligned}}
b_1 &= 0 \\\\
b_0 &= 0 \\\\
a_0 &= 0 \\\\
c_1 &= {A4} \\\\
c_0 &= {A3}
\\end{{aligned}}
$$
""")

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
                steps.append(f"""
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
""")

            x1 = -1 * a0
            x2, x3 = self.ABC(b=b1, c=b0)
            x4, x5 = self.ABC(b=c1, c=c0)

            return rows, steps, (x1, x2, x3, x4, x5), None

        else:
            return None, None, None, f"Derajat {degree} tidak didukung"


class LinearRegression:
    def __init__(self, data, n=0, mode="std"):
        self.data = data
        self.n = len(data) if n == 0 else n
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

        print(x)
        print(y)

        sum_x = custom_round(sum(x))
        sum_y = custom_round(sum(y))

        xy = [custom_round(x[i] * y[i]) for i in range(self.n)]
        sum_xy = custom_round(sum(xy))

        x2 = [custom_round(x[i] ** 2) for i in range(self.n)]
        sum_x2 = custom_round(sum(x2))

        avg_x = custom_round(sum_x / self.n)
        avg_y = custom_round(sum_y / self.n)

        print(sum_x, sum_y, sum_xy, sum_x2, avg_x, avg_y)

        try:
            a1 = (self.n * sum_xy - sum_x * sum_y) / (self.n * sum_x2 - sum_x**2)
        except (ZeroDivisionError, ValueError):
            return pd.DataFrame(), [], None, "Pembagian dengan nol saat menghitung a1"
        a0 = custom_round(avg_y - a1 * avg_x)

        print(a1, a0)

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
        step += f"a_1 &= \\frac{{n \\sum xy - \\sum x \\sum y}}{{n \\sum x^2 - (\\sum x)^2}} = \\frac{{({self.n}) \\cdot ({sum_xy}) - ({sum_x}) \\cdot ({sum_y})}}{{({self.n}) \\cdot ({sum_x2}) - ({sum_x})^2}} = {a1} \\\\\n"
        step += f"a_0 &= \\bar{{y}} - a_1 \\bar{{x}} = {avg_y} - {a1} \\cdot {avg_x} = {a0}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        match self.mode:
            case "std":
                steps.append(f"### Persamaan: $y = {a0} {sign(a1)} {a1}x$")
            case "log":
                a = custom_round(10**a0)
                steps.append(f"### Persamaan: $y = {a} \\cdot x^{{{a1}}}$")
            case "exp":
                a = custom_round(np.exp(a0))
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
        steps.append(f"### Persamaan: $y = {a0} {sign(a1)} {a1}x {sign(a2)} {a2}x^2$")

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

                    step += f"**Iterasi {iterasi}:** $B_{{{j}}} \\leftarrow B_{{{j}}} - {f'({custom_round(times)})' if times != 1 else ''}B_{{{i}}}$\n"
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

                        step += f"**Iterasi {iterasi}:** $B_{{{j}}} \\leftarrow B_{{{j}}} - {f'({custom_round(times)})' if times != 1 else ''}B_{{{i}}}$\n"
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

        # initial value
        rows.append(
            {"Iterasi": 0, **{f"x{i + 1}": custom_round(x[i]) for i in range(m)}}
        )
        step = f"**Iterasi 0:**\n\n"
        step += "$$\n\\begin{aligned}\n"
        for i in range(m):
            step += f"x_{{{i + 1}}} &= 0"
            if i < m - 1:
                step += " \\\\\n"
        step += "\n\\end{aligned}\n$$\n\n"
        steps.append(step)

        for iter_num in range(self.max_iter):
            for i in range(m):
                sum_v = 0
                for j in range(m):
                    if i != j:
                        sum_v += self.A[i][j] * x[j]
                x_new[i] = (self.B[i] - sum_v) / self.A[i][i]

            row = {"Iterasi": iter_num + 1}
            for i in range(m):
                row[f"x{i + 1}"] = custom_round(x_new[i])
            rows.append(row)

            step = f"**Iterasi {iter_num + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            for i in range(m):
                sum_parts = " - ".join(
                    [
                        f"\\left({self.A[i][j]}\\right) \\cdot \\left({custom_round(x[j])}\\right)"
                        for j in range(m)
                        if i != j
                    ]
                )
                step += f"x_{{{i + 1}}} &= \\frac{{{self.B[i]} - {sum_parts}}}{{{self.A[i][i]}}} = {custom_round(x_new[i])}"
                if i < m - 1:
                    step += " \\\\[1em]\n"
            step += "\n\\end{aligned}\n$$\n\n"
            steps.append(step)

            # Check convergence
            converged = True
            i = 0
            while i < m:
                if abs(x_new[i] - x[i]) > self.tol:
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

        # initial value
        rows.append(
            {"Iterasi": 0, **{f"x{i + 1}": custom_round(x[i]) for i in range(m)}}
        )
        step = f"**Iterasi 0:**\n\n"
        step += "$$\n\\begin{aligned}\n"
        for i in range(m):
            step += f"x_{{{i + 1}}} &= 0"
            if i < m - 1:
                step += " \\\\\n"
        step += "\n\\end{aligned}\n$$\n\n"
        steps.append(step)

        for iter_num in range(self.max_iter):
            step = f"**Iterasi {iter_num + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"

            for i in range(m):
                sum_v = 0
                sum_parts_list = []

                for j in range(m):
                    if i != j:
                        sum_v += self.A[i][j] * x_new[j]
                        sum_parts_list.append(
                            f"\\left({self.A[i][j]}\\right) \\cdot \\left({custom_round(x_new[j])}\\right)"
                        )

                x_new[i] = (self.B[i] - sum_v) / self.A[i][i]

                sum_parts = " - ".join(sum_parts_list)
                step += f"x_{{{i + 1}}} &= \\frac{{{self.B[i]} - {sum_parts}}}{{{self.A[i][i]}}} = {custom_round(x_new[i])}"
                if i < m - 1:
                    step += " \\\\[1em]\n"

            step += "\n\\end{aligned}\n$$\n\n"
            steps.append(step)

            row = {"Iterasi": iter_num + 1}
            for i in range(m):
                row[f"x{i + 1}"] = custom_round(x_new[i])
            rows.append(row)

            converged = True
            for i in range(m):
                if abs(x_new[i] - x[i]) > self.tol:
                    converged = False
                    break

            if converged:
                x = x_new.copy()
                steps.append(f"### **Konvergen** pada iterasi ke-{iter_num + 1}")
                break

            x = x_new.copy()

        # Finalisasi output
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
            diff_table[i][0] = custom_round(
                (self.data[i + 1][1] - self.data[i][1])
                / (self.data[i + 1][0] - self.data[i][0])
            )

            diff_table[i][0] = custom_round(diff_table[i][0])

        for i in range(1, self.n - 1):
            for j in range(self.n - i - 1):
                diff_table[j][i] = custom_round(
                    (diff_table[j + 1][i - 1] - diff_table[j][i - 1])
                    / (self.data[j + i + 1][0] - self.data[j][0])
                )

                diff_table[j][i] = custom_round(diff_table[j][i])

        # Build rows
        for i in range(self.n):
            row = {"x": self.data[i][0], "f(x)": self.data[i][1]}
            for j in range(self.n - 1):
                row[f"Δ{f'{j + 1}' if j + 1 > 1 else ''}f"] = (
                    diff_table[i][j] if diff_table[i][j] != 0 else np.nan
                )
            rows.append(row)

        step = "**Koefisien:**\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"b_0 = {self.data[0][1]} \\\\\n"
        for i in range(self.n - 1):
            step += f"b_{{{i + 1}}} = {diff_table[0][i]} \\\\\n"
        step += "\n\\end{aligned}\n$$\n\n"
        step += "\n\n"
        steps.append(step)

        res = self.data[0][1]
        step2 = f"**Evaluasi polinomial di $x = {self.x}$:**\n\n"
        step2 += "$$\n\\begin{alignat}{2}\n"
        step2 += (
            f"P({self.x}) & = && \\; {custom_round(self.data[0][1])} \\nonumber \\\\\n"
        )
        for i in range(0, self.n - 1):
            terms = " \\cdot ".join(
                [f"({self.x} - {self.data[j][0]})" for j in range(i + 1)]
            )
            step2 += f"& && + \\; {custom_round(diff_table[0][i])} \\cdot {terms}  \\nonumber \\\\[0.5em]\n"

        # Compute result
        res = self.data[0][1]
        for i in range(0, self.n - 1):
            term = diff_table[0][i]
            for j in range(0, i + 1):
                term *= self.x - self.data[j][0]
            res += term
        res = custom_round(res)

        step2 += f"& = && \\; {res} \\nonumber\n"
        step2 += "\\end{alignat}\n$$\n\n"
        steps.append(step2)

        return pd.DataFrame(rows), steps, res, err


class LagrangeInterpolation:
    def __init__(self, data, n=0, x=0):
        self.data = data
        self.n = n if n != 0 else len(self.data)
        self.x = x

    def solve(self):
        steps = []
        err = None

        res = 0.0
        terms = []
        step = f"**Interpolasi Lagrange di $x = {self.x}$:**\n\n"
        step += "$$\n\\begin{alignat}{2}\n"
        step += f"P({self.x}) & = && \\; "
        for i in range(self.n):
            li = self.data[i][1]
            num_parts = []
            den_parts = []
            for j in range(self.n):
                if i != j:
                    li *= (self.x - self.data[j][0]) / (
                        self.data[i][0] - self.data[j][0]
                    )
                    num_parts.append(
                        f"({self.x} {'-' if self.data[j][0] >= 0 else '+'} {abs(self.data[j][0])})"
                    )
                    den_parts.append(
                        f"({self.data[i][0]} {'-' if self.data[j][0] >= 0 else '+'} {abs(self.data[j][0])})"
                    )
            li = custom_round(li)
            terms.append(li)
            res += li

            num_str = " \\cdot ".join(num_parts)
            den_str = " \\cdot ".join(den_parts)
            if i == 0:
                step += f"({self.data[i][1]}) \\cdot \\frac{{{num_str}}}{{{den_str}}}"
            else:
                step += f"& && \\; + ({self.data[i][1]}) \\cdot \\frac{{{num_str}}}{{{den_str}}}"

            step += "\\nonumber \\\\[1.5em]\n"

        res = custom_round(res)
        step += f"& = && \\; {res} \\nonumber"
        step += "\\end{alignat}\n$$\n\n"

        step += "$$\n\\begin{aligned}\n"
        step += f"P({self.x}) &= " + " ".join(
            f"{sign(terms[i])} {terms[i]}" for i in range(self.n)
        ).lstrip("+ ")
        step += f" \\\\\n&= {res}\n"
        step += "\n\\end{aligned}\n$$\n\n"

        steps.append(step)

        return None, steps, res, err


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
                col_label = f"Δ{j + 1}y"
                row[col_label] = table[i][j] if table[i][j] != 0 else np.nan
            rows.append(row)

        step = f"**Newton-Gregory ({self.mode}):**\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"h &= {custom_round(h)} \\\\[1em]\n"
        step += f"s &= \\frac{{{self.x} - {self.data[idx][0]}}}{{{custom_round(h)}}} = {custom_round(s)}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        match self.mode:
            case "forward":
                term_temp = []
                res = self.data[idx][1]
                max_terms = self.n - 1 - idx
                num_terms = (
                    self.orde
                    if self.orde != -1 and self.orde < max_terms
                    else max_terms
                )
                step2 = "$$\n\\begin{alignat}{2}\n"
                step2 += f"P({self.x}) & = && \\; {custom_round(self.data[idx][1])} \\nonumber \\\\[0.5em]\n"
                for i in range(num_terms):
                    k = i + 1
                    temp = 1
                    for j in range(k):
                        temp *= s - j
                    term_val = custom_round(temp * table[idx][i] / self.factorial(k))
                    term_temp.append(term_val)
                    res = custom_round(res + term_val)
                    s_parts = " \\cdot ".join(
                        [
                            (
                                f"({custom_round(s)} - {j})"
                                if j > 0
                                else f"({custom_round(s)})"
                            )
                            for j in range(k)
                        ]
                    )
                    step2 += f"& && \\; + \\frac{{{s_parts}}}{{{k}!}} \\; ({custom_round(table[idx][i])}) \\nonumber {f'\\\\[0.5em]' if i < num_terms - 1 else ''}\n"
                step2 += "\\end{alignat}\n$$\n\n"

                step2 += "$$\n\\begin{aligned}\n"
                step2 += (
                    f"P({self.x}) &= {self.data[idx][1]} {sign(term_temp[0]) if len(term_temp) > 0 else ''}"
                    + " ".join(
                        f"{sign(term_temp[i])} {term_temp[i]}" for i in range(num_terms)
                    ).lstrip("+ ")
                )
                step2 += f" \\\\\n&= {res}\n"
                step2 += "\n\\end{aligned}\n$$\n\n"
                steps.append(step2)

            case "backward":
                term_temp = []
                res = self.data[idx][1]
                step2 = "$$\n\\begin{alignat}{2}\n"
                step2 += f"P({self.x}) & = && \\; {custom_round(self.data[idx][1])} \\nonumber \\\\[1em]\n"
                j = 0
                for i in range(idx - 1, -1, -1):
                    if self.orde != -1 and j >= self.orde:
                        break
                    k = j + 1
                    temp = 1
                    for mm in range(k):
                        temp *= s + mm
                    term_val = custom_round(temp * table[i][j] / self.factorial(k))
                    term_temp.append(term_val)
                    res = custom_round(res + term_val)
                    s_parts = " \\cdot ".join(
                        [
                            (
                                f"({custom_round(s)} + {mm})"
                                if mm > 0
                                else f"({custom_round(s)})"
                            )
                            for mm in range(k)
                        ]
                    )
                    step2 += f"& && + \\; \\frac{{{s_parts}}}{{{k}!}} \\; ({custom_round(table[i][j])}) \\nonumber {f'\\\\[0.5em]' if i > 0 else ''}\n"
                    j += 1
                step2 += "\\end{alignat}\n$$\n\n"

                step2 += "$$\n\\begin{aligned}\n"
                step2 += (
                    f"P({self.x}) &= {self.data[idx][1]} {sign(term_temp[0]) if len(term_temp) > 0 else ''}"
                    + " ".join(
                        f"{sign(term_temp[i])} {term_temp[i]}"
                        for i in range(len(term_temp))
                    ).lstrip("+ ")
                )
                step2 += f" \\\\\n&= {res}\n"
                step2 += "\n\\end{aligned}\n$$\n\n"
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

        h = custom_round(self.data[1][0] - self.data[0][0])
        s = custom_round(((self.x - self.data[idx][0]) / h))

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
                col_label = f"Δ{j + 1}y"
                row[col_label] = table[i][j] if table[i][j] != 0 else np.nan
            rows.append(row)

        step = f"**Interpolasi Stirling:**\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"h &= {custom_round(h)} \\\\[1em]\n"
        step += f"s &= \\frac{{{self.x} - {self.data[idx][0]}}}{{{h}}} = {s}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        res = self.data[idx][1]
        step2 = "$$\n\\begin{alignat}{2}\n"
        step2 += f"P({self.x}) & = && \\; {custom_round(self.data[idx][1])} \\nonumber \\\\[0.5em]\n"

        max_orde = self.orde if self.orde != -1 else self.n - 1
        term_temp = [self.data[idx][1]]
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
                num_parts = [f"({s})"]
                temp = s
                for j in range(1, (k + 1) // 2):
                    temp *= s**2 - j**2
                    num_parts.append(f"(({s})^2 - {j**2})")
                temp = custom_round(temp)
                term_val = custom_round(temp * avg_diff / self.factorial(k))
                term_temp.append(term_val)
                res = custom_round(res + term_val)
                num_str = " ".join(num_parts)
                step2 += f"& && \\; + \\frac{{{num_str}}}{{{k}!}} \\; ({avg_diff}) \\nonumber \\\\[0.5em]\n"
            else:
                # Even order: central difference
                c_idx = idx - k // 2
                if c_idx < 0 or k - 1 >= self.n - 1:
                    break
                diff_val = table[c_idx][k - 1]
                # Product: s^2 * (s^2 - 1^2) * ... * (s^2 - ((k/2)-1)^2)
                num_parts = [f"({s})^2"]
                temp = s**2
                for j in range(1, k // 2):
                    temp *= s**2 - j**2
                    num_parts.append(f"(({s})^2 - {j**2})")
                temp = custom_round(temp)
                term_val = custom_round(temp * diff_val / self.factorial(k))
                term_temp.append(term_val)
                res = custom_round(res + term_val)
                num_str = " ".join(num_parts)
                step2 += f"& && \\; + \\frac{{{num_str}}}{{{k}!}} \\; ({custom_round(diff_val)}) \\nonumber \\\\[0.5em]\n"
            k += 1

        step2 += "\\end{alignat}\n$$\n\n"

        step2 += "$$\n\\begin{aligned}\n"
        step2 += f"P({self.x}) &= " + " ".join(
            f"{sign(term_temp[i])} {term_temp[i]}" for i in range(len(term_temp))
        ).lstrip("+ ")
        step2 += f" \\\\\n&= {custom_round(res)}\n"
        step2 += "\n\\end{aligned}\n$$\n\n"
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

        table = [[0.0 for _ in range(self.n)] for _ in range(self.n)]

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

        h = custom_round(self.data[1][0] - self.data[0][0])
        s = custom_round(((self.x - self.data[idx][0]) / h))
        u = custom_round(s - 0.5)

        for i in range(self.n):
            table[i][0] = self.data[i][1]

        for j in range(1, self.n):
            for i in range(self.n - j):
                table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

        for i in range(self.n):
            for j in range(self.n):
                if table[i][j] != 0:
                    table[i][j] = custom_round(table[i][j])

        for i in range(self.n):
            row = {"x": self.data[i][0], "y": self.data[i][1]}
            for j in range(1, self.n - i):
                col_label = f"Δ{j + 1}y"
                row[col_label] = table[i][j] if table[i][j] != 0 else np.nan
            rows.append(row)

        step = f"**Interpolasi Bessel:**\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"h &= {custom_round(h)} \\\\[1em]\n"
        step += f"s &= \\frac{{{self.x} - {self.data[idx][0]}}}{{{h}}} = {s}\\\\[1em]\n"
        step += f"u &= s - \\frac{{1}}{{2}} = {u}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        res = custom_round((table[idx][0] + table[idx + 1][0]) / 2)
        step2 = "$$\n\\begin{alignat}{2}\n"
        step2 += f"P({self.x}) & = && \\; {res} \\nonumber \\\\[0.5em]\n"

        curr_idx = idx
        term_temp = [res]

        for term in range(1, self.orde + 1 if self.orde != -1 else self.n):
            num_parts = []
            if term % 2 == 1:
                if curr_idx < 0 or curr_idx >= self.n - term:
                    break

                diff_val = table[curr_idx][term]
                coef = u
                num_parts.append(f"({u})")
                for k in range(1, (term // 2) + 1):
                    coef *= u**2 - ((2 * k - 1) ** 2) / 4
                    num_parts.append(f"(({u})^2 - \\frac{{{(2 * k - 1) ** 2}}}{{4}})")

                curr_idx -= 1

            else:
                if curr_idx + 1 < 0 or curr_idx + 1 >= self.n - term:
                    break

                diff_val = (table[curr_idx][term] + table[curr_idx + 1][term]) / 2
                coef = 1
                for k in range(1, (term // 2) + 1):
                    coef *= u**2 - ((2 * k - 1) ** 2) / 4
                    num_parts.append(f"(({u})^2 - \\frac{{{(2 * k - 1) ** 2}}}{{4}})")

            term_value = custom_round((coef * diff_val) / self.factorial(term))
            term_temp.append(term_value)
            res += term_value
            num_str = " ".join(num_parts)
            step2 += f"& && \\; + \\frac{{{num_str}}}{{{term}!}} \\; ({diff_val}) \\nonumber \\\\[0.5em]\n"

        step2 += "\\end{alignat}\n$$\n\n"

        step2 += "$$\n\\begin{aligned}\n"
        step2 += f"P({self.x}) &= " + " ".join(
            f"{sign(term_temp[i])} {term_temp[i]}" for i in range(len(term_temp))
        ).lstrip("+ ")
        step2 += f" \\\\\n&= {custom_round(res)}\n"
        step2 += "\n\\end{aligned}\n$$\n\n"
        steps.append(step2)

        return pd.DataFrame(rows), steps, custom_round(res), err


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

    def solve(self):
        rows = []
        steps = []
        err = None

        table = [[0 for _ in range(self.n - 1)] for _ in range(self.n)]
        idx = self.search_x0()
        if idx == -1:
            return pd.DataFrame(), [], None, "x0 tidak ada dalam data"

        h = custom_round(self.data[1][0] - self.data[0][0])
        s = custom_round((self.x - self.data[idx][0]) / h)

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
                col_label = f"Δ{j + 1}y"
                row[col_label] = table[i][j] if table[i][j] != 0 else np.nan
            rows.append(row)

        step = f"**Diferensiasi Newton-Gregory ({self.mode}):**\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"h &= {h} \\\\[1em]\n"
        step += f"s &= \\frac{{{self.x} - {self.data[idx][0]}}}{{{h}}} = {s}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        res_deriv_x = 0
        s_sym = sp.symbols("s")
        match self.mode:
            case "forward":
                max_terms = self.n - 1 - idx
                num_terms = (
                    self.orde
                    if self.orde != -1 and self.orde < max_terms
                    else max_terms
                )

                latex_formula_terms = []
                latex_numeric_terms = []
                term_values = []

                for k in range(1, num_terms + 1):
                    diff_val = table[idx][k - 1]

                    prod = 1
                    for j in range(k):
                        prod *= s_sym - j

                    poly_deriv = sp.diff(prod, s_sym)
                    poly_deriv_expanded = sp.expand(poly_deriv)

                    # 3. Evaluate the derivative numerically at s
                    poly_deriv_val = float(poly_deriv_expanded.subs(s_sym, s))

                    # 4. Calculate total value for this term
                    fact_k = self.factorial(k)
                    term_val = custom_round((poly_deriv_val / fact_k) * diff_val)
                    term_values.append(term_val)

                    formula_poly_latex = sp.latex(poly_deriv_expanded)
                    numeric_poly_latex = sp.latex(
                        poly_deriv_expanded.subs(s_sym, sp.Symbol(f"({s})"))
                    )

                    if k == 1:
                        latex_formula_terms.append(f"{custom_round(diff_val)}")
                        latex_numeric_terms.append(f"{custom_round(diff_val)}")
                    else:
                        latex_formula_terms.append(
                            f"\\frac{{{formula_poly_latex}}}{{{k}!}} ({custom_round(diff_val)})"
                        )
                        latex_numeric_terms.append(
                            f"\\frac{{{numeric_poly_latex}}}{{{k}!}} ({custom_round(diff_val)})"
                        )

                step2 = "$$\n\\begin{alignat*}{2}\n"
                if len(latex_numeric_terms) == 1:
                    step2 += f"P'({self.x}) &= \\frac{{1}}{{{h}}} \\; \\Biggl[ \\; && {latex_numeric_terms[0]} \\Biggr] \\\\[1em]\n"
                elif len(latex_numeric_terms) == 2:
                    step2 += f"P'({self.x}) &= \\frac{{1}}{{{h}}} \\; \\Biggl[ \\; && {latex_numeric_terms[0]} + {latex_numeric_terms[1]} \\Biggr] \\\\[1em]\n"
                else:
                    step2 += f"P'({self.x}) &= \\frac{{1}}{{{h}}} \\; \\Biggl[ \\; && {latex_numeric_terms[0]} + {latex_numeric_terms[1]} \\; + \\\\\n"
                    for i in range(2, len(latex_numeric_terms) - 1):
                        step2 += f"& && {latex_numeric_terms[i]} \\; + \\\\\n"
                    step2 += f"& && {latex_numeric_terms[-1]} \\; \\Biggr] \\\\[1em]\n"

                term_str_list = []
                for val in term_values:
                    if val >= 0:
                        term_str_list.append(f"+ {val}")
                    else:
                        term_str_list.append(f"- {abs(val)}")
                terms_joined = " ".join(term_str_list).lstrip("+ ")

                res_deriv_s = sum(term_values)
                res_deriv_x = custom_round(res_deriv_s / h)
                step2 += "\\end{alignat*}\n$$\n"

                step2 += "$$\n\\begin{aligned}\n"
                step2 += f"P'({self.x}) &= \\frac{{1}}{{{h}}} \\; \\left[ {terms_joined} \\right] = {res_deriv_x} \\\\[1em]\n"
                step2 += "\n\\end{aligned}\n$$\n\n"

                steps.append(step2)

            case "backward":
                max_terms = idx
                num_terms = (
                    self.orde
                    if self.orde != -1 and self.orde < max_terms
                    else max_terms
                )

                if num_terms <= 0:
                    return (
                        pd.DataFrame(),
                        [],
                        None,
                        "Data tidak cukup untuk mode backward pada x0 tersebut",
                    )

                latex_formula_terms = []
                latex_numeric_terms = []
                term_values = []

                for k in range(1, num_terms + 1):
                    diff_val = table[idx - k][k - 1]

                    prod = 1
                    for j in range(k):
                        prod *= s_sym + j

                    poly_deriv = sp.diff(prod, s_sym)
                    poly_deriv_expanded = sp.expand(poly_deriv)

                    poly_deriv_val = float(poly_deriv_expanded.subs(s_sym, s))

                    fact_k = self.factorial(k)
                    term_val = custom_round((poly_deriv_val / fact_k) * diff_val)
                    term_values.append(term_val)

                    formula_poly_latex = sp.latex(poly_deriv_expanded)
                    numeric_poly_latex = sp.latex(
                        poly_deriv_expanded.subs(s_sym, sp.Symbol(f"({s})"))
                    )

                    if k == 1:
                        latex_formula_terms.append(f"{custom_round(diff_val)}")
                        latex_numeric_terms.append(f"{custom_round(diff_val)}")
                    else:
                        latex_formula_terms.append(
                            f"\\frac{{{formula_poly_latex}}}{{{k}!}} ({custom_round(diff_val)})"
                        )
                        latex_numeric_terms.append(
                            f"\\frac{{{numeric_poly_latex}}}{{{k}!}} ({custom_round(diff_val)})"
                        )

                step2 = "$$\n\\begin{alignat*}{2}\n"

                if len(latex_numeric_terms) == 1:
                    step2 += f"P'({self.x}) &= \\frac{{1}}{{{h}}} \\; \\Biggl[ \\; && {latex_numeric_terms[0]} \\; \\Biggr] \\\\[1em]\n"
                elif len(latex_numeric_terms) == 2:
                    step2 += f"P'({self.x}) &= \\frac{{1}}{{{h}}} \\; \\Biggl[ \\; && {latex_numeric_terms[0]} + {latex_numeric_terms[1]} \\; \\Biggr] \\\\[1em]\n"
                else:
                    # Matches your target layout exactly
                    step2 += f"P'({self.x}) &= \\frac{{1}}{{{h}}} \\; \\Biggl[ \\; && {latex_numeric_terms[0]} + {latex_numeric_terms[1]} + \\\\\n"
                    for i in range(2, len(latex_numeric_terms) - 1):
                        step2 += f"& && {latex_numeric_terms[i]} + \\\\\n"
                    step2 += f"& && {latex_numeric_terms[-1]} \\;  \\Biggr] \\\\[1em]\n"

                # --- Step 2c & 2d: Final Simplification and Resolution ---
                term_str_list = []
                for val in term_values:
                    if val >= 0:
                        term_str_list.append(f"+ {val}")
                    else:
                        term_str_list.append(f"- {abs(val)}")
                terms_joined = " ".join(term_str_list).lstrip("+ ")

                res_deriv_s = sum(term_values)
                res_deriv_x = custom_round(res_deriv_s / h)
                step2 += "\\end{alignat*}\n$$\n"

                step2 += "$$\n\\begin{aligned}\n"
                step2 += f"P'({self.x}) &= \\frac{{1}}{{{h}}} \\; \\left[ {terms_joined} \\right] = {res_deriv_x} \\\\[1em]\n"
                step2 += "\n\\end{aligned}\n$$\n\n"

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
        steps = []
        err = None

        if self.a > self.b:
            self.a, self.b = self.b, self.a

        try:
            L = float(sp.integrate(self.f, (self.x, self.a, self.b)))
            L = custom_round(L)
        except Exception as e:
            return None, [], None, f"Error saat integrasi: {str(e)}"

        step = "**Integrasi:**\n\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"\\int_{{{self.a}}}^{{{self.b}}} {sp.latex(self.f)} \\, dx &= {L}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        return None, steps, L, err


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

        if self.n > 1:
            # Multi-segment trapezoidal
            h = custom_round((self.b - self.a) / self.n)
            total = 0
            step = f"**Trapezoidal Multi-Segmen ($n = {self.n}$):**\n\n"
            step += f"$h = \\frac{{{self.b} - {self.a}}}{{{self.n}}} = {h}$\n\n"

            step += "$$\n\\begin{alignat*}{2}\n"
            step += "L & = \\frac{{h}}{{2}} && \\; \\left[ f(x_0) + 2\\sum_{{i=1}}^{{n-1}} f(x_i) + f(x_n) \\right] \\\\\n"

            for i in range(self.n + 1):
                xi = custom_round(self.a + i * h)
                fi = custom_round(self.evaluate(xi))
                if i == 0 or i == self.n:
                    total += fi
                else:
                    total += 2 * fi
                rows.append({"i": i, "x_i": xi, "f(x_i)": fi})

            L = custom_round(0.5 * h * total)
            step += f"& = \\frac{{{h}}}{{2}} && \\; {custom_round(total)} = {L}\n"
            step += "\\end{alignat*}\n$$\n\n"
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
            step += "L &= \\frac{{b - a}}{{2}} \\, \\left( f(a) + f(b) \\right) \\\\\n"
            step += f"&= \\frac{{{self.b} - {self.a}}}{{2}} \\, \\left( {fa} + {fb} \\right) = {L}\n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

        if self.true_val != -1:
            et = Et(self.true_val, L)
            step = "$$\n\\begin{aligned}\n"
            step += f"E_t &= \\left| \\frac{{{self.true_val} - ({L})}}{{{self.true_val}}} \\right| \\times 100\\% \\\\[1em]\n"
            step += f"&= {et} \\% \n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

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

        if self.n != 0 and self.n % 2 != 0:
            return None, None, None, "Jumlah segmen harus kelipatan 2"

        if self.a > self.b:
            self.a, self.b = self.b, self.a

        if self.n > 2:
            # Multi-segment Simpson 1/3
            h = custom_round((self.b - self.a) / self.n)
            total = 0

            step = f"**Simpson 1/3 Multi-Segmen ($n = {self.n}$):**\n\n"
            step += f"$h = \\frac{{{self.b} - {self.a}}}{{{self.n}}} = {h}$\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += "L &= \\frac{{h}}{{3}} \\left[ f(x_0) + 4\\sum_{{\\text{{ganjil}}}} f(x_i) + 2\\sum_{{\\text{{genap}}}} f(x_i) + f(x_n) \\right] \\\\\n"

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
            step += f"&= \\frac{{{h}}}{{3}} \\, {custom_round(total)} = {L}\n"
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
            step += "L &= \\frac{{h}}{{3}} \\left[ f(x_0) + 4f(x_1) + f(x_2) \\right] \\\\\n"
            step += f"&= \\frac{{{h}}}{{3}} \\left[ {f0} + 4 \\cdot {f1} + {f2} \\right] = {L}\n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

        if self.true_val != -1:
            et = Et(self.true_val, L)
            step = "$$\n\\begin{aligned}\n"
            step += f"E_t &= \\left| \\frac{{{self.true_val} - ({L})}}{{{self.true_val}}} \\right| \\times 100\\% \\\\[1em]\n"
            step += f"&= {et} \\% \n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

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
        step += "L &= \\frac{{3h}}{{8}} \\left[ f(x_0) + 3f(x_1) + 3f(x_2) + f(x_3) \\right] \\\\\n"
        step += f"&= \\frac{{3 \\times {h}}}{{8}} \\left[ {f0} + 3 \\times {f1} + 3 \\times {f2} + {f3} \\right] = {L}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        if self.true_val != -1:
            et = Et(self.true_val, L)
            step = "$$\n\\begin{aligned}\n"
            step += f"E_t &= \\left| \\frac{{{self.true_val} - ({L})}}{{{self.true_val}}} \\right| \\times 100\\% \\\\[1em]\n"
            step += f"&= {et} \\% \n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

        return pd.DataFrame(rows), steps, L, err


class RiemannIntegration:
    def __init__(self, f, a, b, n, true_val=-1):
        self.x = sp.symbols("x")
        self.f = sp.sympify(f)
        self.a = a
        self.b = b
        self.n = np.maximum(1, n)
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
        step += "L &= h \\sum_{{i=0}}^{{n-1}} f(x_i) \\\\\n"

        for i in range(self.n):
            xi = custom_round(self.a + i * h)
            fi = custom_round(self.evaluate(xi))
            total = custom_round(total + fi)
            rows.append({"i": i, "x_i": xi, "f(x_i)": fi})

        L = custom_round(h * total)
        step += f"&= {h} \\times {total} = {L}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        if self.true_val != -1:
            et = Et(self.true_val, L)
            step = "$$\n\\begin{aligned}\n"
            step += f"E_t &= \\left| \\frac{{{self.true_val} - ({L})}}{{{self.true_val}}} \\right| \\times 100\\% \\\\[1em]\n"
            step += f"&= {et} \\% \n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

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
        steps = []
        err = None

        if self.a > self.b:
            self.a, self.b = self.b, self.a

        u = sp.symbols("u")
        x_u = (self.b - self.a) * (u / 2) + (self.b + self.a) / 2
        g_u = ((self.b - self.a) / 2) * self.f.subs(self.x, x_u)

        u1 = custom_round(-1 / np.sqrt(3))
        u2 = custom_round(1 / np.sqrt(3))

        g1 = custom_round(g_u.subs(u, u1))
        g2 = custom_round(g_u.subs(u, u2))

        L = custom_round(g1 + g2)

        step = "**Gauss Quadrature (2-point):**\n\n"
        step += f"Transformasi: $\\; x = \\frac{{{self.b} - {self.a}}}{{2}} u + \\frac{{{self.a} + {self.b}}}{{2}} = {sp.latex(x_u)}$\n\n"
        step += f"Fungsi $g(u)$: $\\; g(u) = \\frac{{{self.b} - {self.a}}}{{2}} f(x) = {sp.latex(g_u)}$\n\n"

        step += "$$\n\\begin{aligned}\n"
        step += "L &= g\\left( \\frac{{1}}{{\\sqrt{{3}}}} \\right) + g\\left( - \\frac{{1}}{{\\sqrt{{3}}}} \\right) \\\\[1em]\n"
        step += f"&= {g1} + {g2} = {L}\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        if self.true_val != -1:
            et = Et(self.true_val, L)
            step = "$$\n\\begin{aligned}\n"
            step += f"E_t &= \\left| \\frac{{{self.true_val} - ({L})}}{{{self.true_val}}} \\right| \\times 100\\% \\\\[1em]\n"
            step += f"&= {et} \\% \n"
            step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

        return None, steps, L, err


class Euler:
    def __init__(self, df, a, b, h):
        self.x = sp.symbols("x")
        self.df_expr = sp.sympify(df)
        self.a = a
        self.b = b
        self.h = h

    def evaluate(self, expr, val):
        return custom_round(expr.subs(self.x, val))

    def round_coeff(self, expr):
        return expr.xreplace({n: custom_round(n) for n in expr.atoms(sp.Number)})

    def solve(self):
        rows1 = []
        rows2 = []
        steps = []
        err = None

        true_f = sp.integrate(self.df_expr)
        true_f = self.round_coeff(true_f)

        n_steps = int((self.b - self.a) / self.h)
        xi = self.a
        y = self.evaluate(true_f, xi)

        step = "**Metode Euler:**\n\n"
        step += "Asumsikan $\\int \\left( \\frac{{dy}}{{dx}} \\right) \\,dx = \\int F(x, y) \\,dx = f(x) + C, \\; C = 0$\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"h &= {self.h} \\\\[0.5em]\n"
        step += f"(x_0, y_0) &= ({self.a}, {y}) \\\\[0.5em]\n"
        step += (
            f"\\frac{{dy}}{{dx}} = F(x_i, y_i) &= {sp.latex(self.df_expr)} \\\\[1em]\n"
        )
        step += f"f(x) &= {sp.latex(true_f)} \\\\[0.5em]\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        rows1.append(
            {
                "i": 0,
                "x_i": custom_round(xi),
                "F(x_i, y_i)": custom_round(self.evaluate(self.df_expr, xi)),
                "f(x_i)": custom_round(y),
            }
        )

        # table f and df
        idx = 1
        for i in np.arange(self.a + self.h, self.b + self.h, self.h):
            temp_f = self.evaluate(true_f, i)
            temp_df = self.evaluate(self.df_expr, i)
            rows1.append(
                {
                    "i": idx,
                    "x_i": custom_round(i),
                    "F(x_i, y_i)": custom_round(temp_df),
                    "f(x_i)": custom_round(temp_f),
                }
            )
            idx += 1

        rows2.append({"i": 0, "x_i": self.a, "y_i": y, "Et (%)": np.nan})

        for i in range(n_steps):
            df = custom_round(self.evaluate(self.df_expr, xi))
            y_old = custom_round(y)
            y = custom_round(y + df * self.h)
            xi_next = custom_round(xi + self.h)
            et_val = np.nan
            if true_f is not None:
                true_val = custom_round(self.evaluate(true_f, xi_next))
                et_val = Et(true_val, y)

            rows2.append(
                {
                    "i": i + 1,
                    "x_i": xi_next,
                    "y_i": y,
                    "Et (%)": et_val,
                }
            )

            step = f"**Iterasi {i + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"F({custom_round(xi)}) &= {df} \\\\\n"
            step += f"y_{{{i + 1}}} &= {y_old} + ({df})({self.h}) \\\\\n"
            step += f"&= {y}\n"
            step += "\\end{aligned}\n$$\n\n"
            if true_f is not None:
                step += "$$\n\\begin{aligned}\n"
                step += f"f(x_{i + 1}) &= f({xi_next}) = {true_val} \\\\[1em]\n"
                step += f"E_t &= \\left| \\frac{{{true_val} - ({y})}}{{{true_val}}} \\right| \\times 100\\% \\\\[1em]\n"
                step += f"&= {et_val} \\% \\\\\n"
                step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

            xi = xi_next

        return (pd.DataFrame(rows1), pd.DataFrame(rows2), steps, custom_round(y), err)


class Heunn:
    def __init__(self, df, a, b, h):
        self.x = sp.symbols("x")
        self.df_expr = sp.sympify(df)
        self.a = a
        self.b = b
        self.h = h

    def evaluate(self, expr, val):
        return custom_round(expr.subs(self.x, val))

    def round_coeff(self, expr):
        return expr.xreplace({n: custom_round(n) for n in expr.atoms(sp.Number)})

    def solve(self):
        rows1 = []
        rows2 = []
        steps = []
        err = None

        true_f = sp.integrate(self.df_expr)
        true_f = self.round_coeff(true_f)

        n_steps = int((self.b - self.a) / self.h)
        xi = self.a
        y = self.evaluate(true_f, xi)

        step = "**Metode Heunn:**\n\n"
        step += "Asumsikan $\\int \\left( \\frac{{dy}}{{dx}} \\right) \\,dx = \\int F(x, y) \\,dx = f(x) + C, \\; C = 0$\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"h &= {self.h} \\\\[0.5em]\n"
        step += f"(x_0, y_0) &= ({self.a}, {y}) \\\\[0.5em]\n"
        step += (
            f"\\frac{{dy}}{{dx}} = F(x_i, y_i) &= {sp.latex(self.df_expr)} \\\\[1em]\n"
        )
        step += f"f(x) &= {sp.latex(true_f)} \\\\[0.5em]\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        rows1.append(
            {
                "i": 0,
                "x_i": custom_round(xi),
                "F(x_i, y_i)": custom_round(self.evaluate(self.df_expr, xi)),
                "f(x_i)": custom_round(y),
            }
        )

        # table f and df
        idx = 1
        for i in np.arange(self.a + self.h, self.b + self.h, self.h):
            temp_f = self.evaluate(true_f, i)
            temp_df = self.evaluate(self.df_expr, i)
            rows1.append(
                {
                    "i": idx,
                    "x_i": custom_round(i),
                    "F(x_i, y_i)": custom_round(temp_df),
                    "f(x_i)": custom_round(temp_f),
                }
            )
            idx += 1

        rows2.append(
            {
                "i": 0,
                "x_i": self.a,
                "y_i": y,
                "Et (%)": np.nan,
            }
        )

        for i in range(n_steps):
            df_1 = custom_round(self.evaluate(self.df_expr, xi))
            df_2 = custom_round(self.evaluate(self.df_expr, xi + self.h))
            y_old = custom_round(y)
            y = custom_round(y + ((df_1 + df_2) / 2) * self.h)
            xi_next = custom_round(xi + self.h)
            et_val = np.nan
            if true_f is not None:
                true_val = custom_round(self.evaluate(true_f, xi_next))
                et_val = Et(true_val, y)

            rows2.append(
                {
                    "i": i + 1,
                    "x_i": xi_next,
                    "y_i": y,
                    "Et (%)": et_val,
                }
            )

            step = f"**Iterasi {i + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"\\frac{{dy}}{{dx}} = F({custom_round(xi)}) &= {df_1} \\\\\n"
            step += f"F({custom_round(xi + self.h)}) &= {df_2} \\\\\n"
            step += (
                f"y_{{{i + 1}}} &= "
                f"{y_old} + \\frac{{{df_1} {sign(df_2)} {df_2}}}{2} \\times {self.h} \\\\\n"
            )
            step += f"&= {y}\n"
            step += "\\end{aligned}\n$$\n\n"
            if true_f is not None:
                step += "$$\n\\begin{aligned}\n"
                step += f"f(x_{i + 1}) &= f({xi_next}) = {true_val} \\\\[1em]\n"
                step += f"E_t &= \\left| \\frac{{{true_val} - ({y})}}{{{true_val}}} \\right| \\times 100\\% \\\\[1em]\n"
                step += f"&= {et_val} \\% \\\\\n"
                step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

            xi = xi_next

        return (pd.DataFrame(rows1), pd.DataFrame(rows2), steps, custom_round(y), err)


class RungeKutta:
    def __init__(self, df, a, b, h, a2):
        self.x = sp.symbols("x")
        self.df_expr = sp.sympify(df)
        self.a = a
        self.b = b
        self.h = h
        self.a2 = a2

    def evaluate(self, expr, val):
        return custom_round(expr.subs(self.x, val))

    def round_coeff(self, expr):
        return expr.xreplace({n: custom_round(n) for n in expr.atoms(sp.Number)})

    def solve(self):
        rows1 = []
        rows2 = []
        steps = []
        err = None

        a1 = custom_round(1 - self.a2)
        p = custom_round(1 / (2 * self.a2))

        # true function from integral and y0
        true_f = sp.integrate(self.df_expr)
        true_f = self.round_coeff(true_f)

        n_steps = int((self.b - self.a) / self.h)
        xi = self.a
        y = self.evaluate(true_f, xi)

        step = "**Metode Runge-Kutta:**\n\n"
        step += "Asumsikan $\\int \\left( \\frac{{dy}}{{dx}} \\right) \\,dx = \\int F(x, y) \\,dx = f(x) + C, \\; C = 0$\n"
        step += "$$\n\\begin{aligned}\n"
        step += f"h &= {self.h} \\\\[0.5em]\n"
        step += f"a_2 &= {self.a2} \\\\[0.5em]\n"
        step += f"a_1 &= 1 - a_2 = {a1} \\\\[0.5em]\n"
        step += f"p_1 = q_{{11}} &= \\frac{{1}}{{2a_2}} = {p} \\\\[1em]\n"
        step += f"(x_0, y_0) &= ({self.a}, {y}) \\\\[0.5em]\n"
        step += (
            f"\\frac{{dy}}{{dx}} = F(x_i, y_i) &= {sp.latex(self.df_expr)} \\\\[1em]\n"
        )
        step += f"f(x) &= {sp.latex(true_f)} \\\\[0.5em]\n"
        step += "\\end{aligned}\n$$\n\n"
        steps.append(step)

        rows1.append(
            {
                "i": 0,
                "x_i": custom_round(xi),
                "F(x_i, y_i)": custom_round(self.evaluate(self.df_expr, xi)),
                "f(x_i)": custom_round(y),
            }
        )

        # table f and df
        idx = 1
        for i in np.arange(self.a + self.h, self.b + self.h, self.h):
            temp_f = self.evaluate(true_f, i)
            temp_df = self.evaluate(self.df_expr, i)
            rows1.append(
                {
                    "i": idx,
                    "x_i": custom_round(i),
                    "F(x_i, y_i)": custom_round(temp_df),
                    "f(x_i)": custom_round(temp_f),
                }
            )
            idx += 1

        rows2.append(
            {
                "i": 0,
                "x_i": self.a,
                "y_i": y,
                "Et (%)": np.nan,
            }
        )

        for i in range(n_steps):
            k1 = custom_round(self.evaluate(self.df_expr, xi))
            k2 = custom_round(self.evaluate(self.df_expr, xi + p * self.h))
            y_old = custom_round(y)
            y = custom_round(y + (a1 * k1 + self.a2 * k2) * self.h)
            xi_next = custom_round(xi + self.h)
            et_val = np.nan
            if true_f is not None:
                true_val = custom_round(self.evaluate(true_f, xi_next))
                et_val = Et(true_val, y)

            rows2.append(
                {
                    "i": i + 1,
                    "x_i": xi_next,
                    "y_i": y,
                    "Et (%)": et_val,
                }
            )

            step = f"**Iterasi {i + 1}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"k_1 &= F({xi}) = {k1} \\\\\n"
            step += f"k_2 &= F({xi + p * self.h}) = {k2} \\\\\n"
            step += (
                f"y_{{{i + 1}}} &= "
                f"{y_old} + (({a1})({k1}) + ({self.a2})({k2})) \\times {self.h} \\\\\n"
            )
            step += f"&= {y}\n"
            step += "\\end{aligned}\n$$\n\n"
            if true_f is not None:
                step += "$$\n\\begin{aligned}\n"
                step += f"f(x_{i + 1}) &= f({xi_next}) = {true_val} \\\\[1em]\n"
                step += f"E_t &= \\left| \\frac{{{true_val} - ({y})}}{{{true_val}}} \\right| \\times 100\\% \\\\[1em]\n"
                step += f"&= {et_val} \\% \\\\\n"
                step += "\\end{aligned}\n$$\n\n"
            steps.append(step)

            xi = xi_next

        return (pd.DataFrame(rows1), pd.DataFrame(rows2), steps, custom_round(y), err)
