import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal
import math

def custom_round(num: float) -> float:
    try:
        num = float(num)
        if math.isnan(num) or math.isinf(num):
            return num
        temp = float(Decimal(str(num)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        return temp
    except Exception:
        return num

def Et(true: float, approx: float) -> float:
    approx = custom_round(approx)
    true = custom_round(true)
    return custom_round(abs((true - approx) / true) * 100)

def Ea(approx: float, approx_old: float) -> float:
    approx = custom_round(approx)
    approx_old = custom_round(approx_old)
    return custom_round(abs((approx - approx_old) / approx) * 100)

class BiSection:
    def __init__(self, f: str, xl: float, xu: float, x_true: float, max_iter: float = 10, tol: float = 0.1) -> None:
        self.x = sp.symbols('x')
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

        while True:
            xr = custom_round((self.xl + self.xu) / 2.0)
            fxr = custom_round(self.evaluate(self.f, xr))
            iterasi += 1

            ea = Ea(xr, xr_old)
            et = Et(self.x_true, xr)

            fl = self.evaluate(self.f, self.xl)
            fu = self.evaluate(self.f, self.xu)

            rows.append({
                "Iter": iterasi,
                "XL": custom_round(self.xl),
                "XU": custom_round(self.xu),
                "XR": custom_round(xr),
                "f(XL)": custom_round(fl),
                "f(XU)": custom_round(fu),
                "f(XR)": custom_round(fxr),
                "Et (%)": et,
                "Ea (%)": "" if iterasi == 1 else ea
            })

            if fl * fxr < 0:
                self.xu = xr
            else:
                self.xl = xr

            if iterasi != 1:
                if 0 <= et < self.tol:
                    break

            if iterasi >= self.max_iter:
                break

            xr_old = xr

        return pd.DataFrame(rows), custom_round(xr)


class FalsePosition:
    def __init__(self, f: str, xl: float, xu: float, x_true: float, max_iter: float = 10, tol: float = 0.1) -> None:
        self.x = sp.symbols('x')
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

        while True:
            fl = custom_round(self.evaluate(self.f, self.xl))
            fu = custom_round(self.evaluate(self.f, self.xu))
            xr = custom_round(self.xu - (fu * (self.xl - self.xu)) / (fl - fu))
            fxr = custom_round(self.evaluate(self.f, xr))
            iterasi += 1

            ea = Ea(xr, xr_old)
            et = Et(self.x_true, xr)

            rows.append({
                "Iter": iterasi,
                "XL": custom_round(self.xl),
                "XU": custom_round(self.xu),
                "XR": custom_round(xr),
                "f(XL)": fl,
                "f(XU)": fu,
                "Et (%)": et,
                "Ea (%)": "" if iterasi == 1 else ea
            })

            if fl * fxr < 0:
                self.xu = xr
            else:
                self.xl = xr

            if iterasi != 1:
                if 0 <= et < self.tol:
                    break

            if iterasi >= self.max_iter:
                break

            xr_old = xr

        return pd.DataFrame(rows), custom_round(xr)


class FixedPoint:
    def __init__(self, f: str, x0: float, x_true: float, max_iter: float = 10, tol: float = 0.1) -> None:
        self.x = sp.symbols('x')
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

        while True:
            x_new = self.evaluate(self.f, x_old)
            iterasi += 1

            ea = Ea(x_new, x_old)
            et = Et(self.x_true, x_new)

            rows.append({
                "Iter": iterasi,
                "x_i": custom_round(x_old),
                "x_(i+1)": custom_round(x_new),
                "Et (%)": et,
                "Ea (%)": ea
            })

            if iterasi != 1:
                if 0 <= et < self.tol:
                    break

            if iterasi >= self.max_iter:
                break

            x_old = x_new

        return pd.DataFrame(rows), custom_round(x_old)


class NewtonRaphson:
    def __init__(self, f: str, x0: float, x_true: float, max_iter: float = 10, tol: float = 0.1) -> None:
        self.x = sp.symbols('x')
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

        while True:
            fx = custom_round(self.evaluate(self.f, x_old))
            dfx = custom_round(self.evaluate(self.df, x_old))
            x_new = custom_round(x_old - fx / dfx)

            et = Et(self.x_true, x_new)
            ea = Ea(x_new, x_old)

            i += 1
            rows.append({
                "Iter": i,
                "x_i": custom_round(x_old),
                "f(x_i)": fx,
                "f'(x_i)": dfx,
                "x_(i+1)": x_new,
                "Et (%)": custom_round(et),
                "Ea (%)": custom_round(ea)
            })

            if i != 1 and 0 <= et < self.tol:
                break

            if i >= self.max_iter:
                break

            x_old = x_new

        return pd.DataFrame(rows), custom_round(x_new)


class Secant:
    def __init__(self, f: str, x0: float, x1: float, x_true: float, max_iter: float = 10, tol: float = 0.1) -> None:
        self.x = sp.symbols('x')
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

        while True:
            fx0 = custom_round(self.evaluate(self.f, x_old_0))
            fx1 = custom_round(self.evaluate(self.f, x_old_1))
            x_new = custom_round(x_old_1 - (fx1 * (x_old_0 - x_old_1)) / (fx0 - fx1))

            et = Et(self.x_true, x_new)
            ea = Ea(x_new, x_old_1)

            i += 1
            rows.append({
                "Iter": i,
                "x_(i-1)": custom_round(x_old_0),
                "x_i": custom_round(x_old_1),
                "f(x_(i-1))": fx0,
                "f(x_i)": fx1,
                "x_(i+1)": x_new,
                "Et (%)": custom_round(et),
                "Ea (%)": custom_round(ea)
            })

            if i != 1 and 0 <= et < self.tol:
                break

            if i >= self.max_iter:
                break

            x_old_0 = x_old_1
            x_old_1 = x_new

        return pd.DataFrame(rows), custom_round(x_new)


st.title("Komputasi Numerik")

metode = st.selectbox(
    "Pilih Metode",
    ["BiSection", "FalsePosition", "FixedPoint", "NewtonRaphson", "Secant"]
)

st.subheader("Bangun Fungsi")

mode_input = st.radio(
    "Mode Input",
    ["Manual", "Builder Fleksibel"]
)

if mode_input == "Manual":
    fungsi = st.text_input(
        "Masukkan fungsi f(x)",
        "10*x**3 - 220*x**2 - 630*x + 3600"
    )
else:
    jumlah_suku = st.number_input(
        "Jumlah suku",
        min_value=1,
        max_value=10,
        value=3
    )

    terms = []

    for i in range(jumlah_suku):
        st.markdown(f"### Suku {i+1}")

        col1, col2 = st.columns(2)

        with col1:
            coef = st.number_input(
                f"Koefisien",
                value=1.0,
                key=f"coef_{i}"
            )

        with col2:
            pangkat = st.number_input(
                f"Pangkat x",
                min_value=0,
                max_value=20,
                value=1,
                key=f"pow_{i}"
            )

        if coef != 0:
            if pangkat == 0:
                terms.append(f"{coef}")
            elif pangkat == 1:
                terms.append(f"{coef}*x")
            else:
                terms.append(f"{coef}*x**{pangkat}")

    fungsi = " + ".join(terms)

    if fungsi == "":
        fungsi = "0"

    st.info(f"Fungsi terbentuk: {fungsi}")

x_true = st.number_input("Nilai true", value=24.0)
max_iter = st.number_input("Maksimum Iterasi", value=10)
tol = st.number_input("Toleransi (%)", value=0.1)

if metode in ["BiSection", "FalsePosition"]:
    xl = st.number_input("Batas bawah (xl)", value=18.0)
    xu = st.number_input("Batas atas (xu)", value=37.0)

elif metode == "FixedPoint":
    x0 = st.number_input("Nilai awal (x0)", value=20.0)
    lambda_val = 0.0001

elif metode == "NewtonRaphson":
    x0 = st.number_input("Nilai awal (x0)", value=20.0)

elif metode == "Secant":
    x0 = st.number_input("x0", value=18.0)
    x1 = st.number_input("x1", value=37.0)

if st.button("Hitung"):
    if metode == "BiSection":
        solver = BiSection(fungsi, xl, xu, x_true, max_iter, tol)
        df, akar = solver.solve()

    elif metode == "FalsePosition":
        solver = FalsePosition(fungsi, xl, xu, x_true, max_iter, tol)
        df, akar = solver.solve()

    elif metode == "FixedPoint":
        g_fungsi = f"x - ({fungsi})*{lambda_val}"
        solver = FixedPoint(g_fungsi, x0, x_true, max_iter, tol)
        df, akar = solver.solve()

    elif metode == "NewtonRaphson":
        solver = NewtonRaphson(fungsi, x0, x_true, max_iter, tol)
        df, akar = solver.solve()

    elif metode == "Secant":
        solver = Secant(fungsi, x0, x1, x_true, max_iter, tol)
        df, akar = solver.solve()

    st.subheader("Hasil Iterasi")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.success(f"Akar pendekatan: {akar}")