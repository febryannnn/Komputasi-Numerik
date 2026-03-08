from typing import Literal
import sympy as sp
import pandas as pd
from utils import custom_round, Ea, Et


class BiSection:
    def __init__(
        self,
        f: str,
        xl: float,
        xu: float,
        x_true: float,
        max_iter: float = 10,
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
            xr = custom_round((self.xl + self.xu) / 2.0)
            fxr = custom_round(self.evaluate(self.f, xr))
            iterasi += 1

            ea = Ea(xr, xr_old)
            et = Et(self.x_true, xr)

            fl = self.evaluate(self.f, self.xl)
            fu = self.evaluate(self.f, self.xu)

            rows.append(
                {
                    "Iterasi": iterasi,
                    "XL": custom_round(self.xl),
                    "XU": custom_round(self.xu),
                    "XR": custom_round(xr),
                    "f(XL)": custom_round(fl),
                    "f(XU)": custom_round(fu),
                    "f(XR)": custom_round(fxr),
                    "Et (%)": et,
                    "Ea (%)": None if iterasi == 1 else ea,
                }
            )

            step = f"**Iterasi {iterasi}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"x_r &= \\frac{{x_l + x_u}}{{2}} = \\frac{{{custom_round(self.xl)} + {custom_round(self.xu)}}}{{2}} = {xr} \\\\\n"
            step += f"f(x_l) &= {fl} \\\\\n"
            step += f"f(x_r) &= {fxr}\n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({xr})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
            )
            if iterasi > 1:
                error.append(
                    f"E_a &= \\left| \\frac{{{xr} - ({custom_round(xr_old)})}}{{{xr}}} \\right| \\times 100\\% = {ea}\\%"
                )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\\n".join(error)
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

            if iterasi != 1:
                if 0 <= et < self.tol:
                    steps.append(
                        f"**Konvergen** pada iterasi ke-{iterasi} (Et < Toleransi)."
                    )
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

            try:
                xr = custom_round(self.xu - (fu * (self.xl - self.xu)) / (fl - fu))
            except ZeroDivisionError:
                err = f"**Error**: Pembagian dengan nol (fl - fu = 0) terjadi pada iterasi ke-{iterasi}."
                break

            fxr = custom_round(self.evaluate(self.f, xr))
            iterasi += 1

            ea = Ea(xr, xr_old)
            et = Et(self.x_true, xr)

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

            step += f"x_r &= x_u - \\frac{{f(x_u) \\cdot (x_l - x_u)}}{{f(x_l) - f(x_u)}} = {custom_round(self.xu)} - \\frac{{ {fu} \\cdot ({custom_round(self.xl)} - ({custom_round(self.xu)}))}}{{{fl} - ({fu})}} = {xr} \\\\\n"
            step += f"f(x_l) &= {fl} \\\\\n"
            step += f"f(x_r) &= {fxr}\n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({xr})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
            )
            if iterasi > 1:
                error.append(
                    f"E_a &= \\left| \\frac{{{xr} - ({custom_round(xr_old)})}}{{{xr}}} \\right| \\times 100\\% = {ea}\\%"
                )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\\n".join(error)
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

            if iterasi != 1:
                if 0 <= et < self.tol:
                    steps.append(
                        f"**Konvergen** pada iterasi ke-{iterasi} (Et < Toleransi)."
                    )
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
            x_new = self.evaluate(self.f, x_old)
            iterasi += 1

            x_old_rounded = custom_round(x_old)
            x_new_rounded = custom_round(x_new)

            ea = Ea(x_new, x_old)
            et = Et(self.x_true, x_new)

            rows.append(
                {
                    "Iterasi": iterasi,
                    "x_i": x_old_rounded,
                    "x_(i+1)": x_new_rounded,
                    "Et (%)": et,
                    "Ea (%)": None if iterasi == 1 else ea,
                }
            )

            step = f"**Iterasi {iterasi}:**\n\n"

            val_str = f"({x_old_rounded})" if x_old_rounded < 0 else str(x_old_rounded)

            substituted_expr = self.f.subs(self.x, sp.Symbol(val_str))
            expr_latex = sp.latex(substituted_expr)

            step += "$$\n\\begin{aligned}\n"
            step += f"x_{{{iterasi}}} &= {expr_latex} = {x_new_rounded}\n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({x_new_rounded})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
            )

            if iterasi > 1:
                error.append(
                    f"E_a &= \\left| \\frac{{{x_new_rounded} - ({x_old_rounded})}}{{{x_new_rounded}}} \\right| \\times 100\\% = {ea}\\%"
                )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\\n".join(error)
            step += "\n\\end{aligned}\n$$\n\n"

            x_old = x_new

            steps.append(step)

            if iterasi != 1:
                if 0 <= et < self.tol:
                    steps.append(
                        f"**Konvergen** pada iterasi ke-{iterasi} (Et < Toleransi)."
                    )
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
        i = 0
        x_old = self.x0

        rows = []
        steps = []
        err = None
        while True:
            i += 1
            fx = custom_round(self.evaluate(self.f, x_old))
            dfx = custom_round(self.evaluate(self.df, x_old))
            try:
                x_new = custom_round(x_old - fx / dfx)
            except ZeroDivisionError:
                err = f"**Error**: Pembagian dengan nol f'(x_{i}) = 0 terjadi pada iterasi ke-{i}."
                break

            et = Et(self.x_true, x_new)
            ea = Ea(x_new, x_old)

            rows.append(
                {
                    "Iterasi": i,
                    "x_i": custom_round(x_old),
                    "f(x_i)": fx,
                    "f'(x_i)": dfx,
                    "x_(i+1)": x_new,
                    "Et (%)": custom_round(et),
                    "Ea (%)": custom_round(ea),
                }
            )

            step = f"**Iterasi {i}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"f(x_{{{i - 1}}}) &= {fx} \\\\\n"
            step += f"f'(x_{{{i - 1}}}) &= {dfx} \\\\\n"
            step += f"x_{{{i}}} &= x_{{{i - 1}}} - \\frac{{f(x_{{{i - 1}}})}}{{f'(x_{{{i - 1}}})}} = {x_new} \n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({x_new})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
            )
            error.append(
                f"E_a &= \\left| \\frac{{{x_new} - ({x_old})}}{{{x_new}}} \\right| \\times 100\\% = {ea}\\%"
            )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\\n".join(error)
            step += "\n\\end{aligned}\n$$\n\n"

            steps.append(step)

            x_old = x_new

            if i != 1:
                if 0 <= et < self.tol:
                    steps.append(f"**Konvergen** pada iterasi ke-{i} (Et < Toleransi).")
                    break

            if i >= self.max_iter:
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
        i = 0
        x_old_0 = self.x0
        x_old_1 = self.x1

        rows = []
        steps = []
        err = None
        while True:
            i += 1
            fx0_raw = self.evaluate(self.f, x_old_0)
            fx1_raw = self.evaluate(self.f, x_old_1)
            fx0 = custom_round(fx0_raw)
            fx1 = custom_round(fx1_raw)
            try:
                x_new = custom_round(
                    x_old_1 - (fx1_raw * (x_old_0 - x_old_1)) / (fx0_raw - fx1_raw)
                )
            except ZeroDivisionError:
                err = f"**Error**: Pembagian dengan nol terjadi f(x_{i - 2}) sama dengan f(x_{i - 1}) pada iterasi ke-{i}."
                break

            et = Et(self.x_true, x_new)
            ea = Ea(x_new, x_old_1)

            rows.append(
                {
                    "Iterasi": i,
                    "x_(i-1)": custom_round(x_old_0),
                    "x_i": custom_round(x_old_1),
                    "f(x_(i-1))": fx0,
                    "f(x_i)": fx1,
                    "x_(i+1)": x_new,
                    "Et (%)": custom_round(et),
                    "Ea (%)": custom_round(ea),
                }
            )

            step = f"**Iterasi {i}:**\n\n"
            step += "$$\n\\begin{aligned}\n"
            step += f"f(x_{{{i - 2}}}) &= {fx0} \\\\\n"
            step += f"f(x_{{{i - 1}}}) &= {fx1} \\\\\n"
            step += f"x_{{{i}}} &= x_{{{i - 1}}} - \\frac{{f(x_{{{i - 1}}}) \\cdot (x_{{{i - 2}}} - x_{{{i - 1}}})}}{{f(x_{{{i - 2}}}) - f(x_{{{i - 1}}})}} = {x_new} \n"
            step += "\\end{aligned}\n$$\n\n"

            error = []
            error.append(
                f"E_t &= \\left| \\frac{{{self.x_true} - ({x_new})}}{{{self.x_true}}} \\right| \\times 100\\% = {et}\\%"
            )
            error.append(
                f"E_a &= \\left| \\frac{{{x_new} - ({x_old_1})}}{{{x_new}}} \\right| \\times 100\\% = {ea}\\%"
            )

            step += "$$\n\\begin{aligned}\n"
            step += " \\\\\n".join(error)
            step += "\n\\end{aligned}\n$$\n\n"

            steps.append(step)

            x_old_0 = x_old_1
            x_old_1 = x_new

            if i != 1:
                if 0 <= et < self.tol:
                    steps.append(f"**Konvergen** pada iterasi ke-{i} (Et < Toleransi).")
                    break

            if i >= self.max_iter:
                err = f"**Tidak konvergen** setelah {self.max_iter} iterasi."
                break

        return pd.DataFrame(rows), steps, custom_round(x_old_1), err
