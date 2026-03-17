import streamlit as st
import numpy as np
from method import (
    BiSection,
    FalsePosition,
    FixedPoint,
    NewtonRaphson,
    Secant,
    MNewtonRaphson,
    PolynomFactorization,
)


st.title("Komputasi Numerik Informatika ITS 📐")

metode = st.selectbox(
    "Pilih Metode",
    [
        "Bi Section",
        "False Position",
        "Fixed Point",
        "Newton Raphson",
        "Secant",
        "Modified Newton Raphson",
        "Polynomial Factorization",
    ],
)

st.subheader("Buat Fungsi")

mode_input = st.radio("Mode Input", ["Manual", "Builder Fleksibel"])

if mode_input == "Manual":
    fungsi = st.text_input(
        (
            "Masukkan fungsi f(x)"
            if metode != "Fixed Point"
            else "Masukkan fungsi x_(i+1)"
        ),
        "10*x**3 - 220*x**2 - 630*x + 3600",
    )
else:
    jumlah_suku = st.number_input("Jumlah suku", min_value=1, max_value=10, value=3)

    terms = []

    for i in range(jumlah_suku):
        st.markdown(f"### Suku {i+1}")

        col1, col2 = st.columns(2)

        with col1:
            coef = st.number_input(f"Koefisien", value=1.0, key=f"coef_{i}")

        with col2:
            pangkat = st.number_input(
                f"Pangkat x", min_value=0, max_value=20, value=1, key=f"pow_{i}"
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

max_iter = st.number_input("Maksimum Iterasi", value=10)

if metode in ["Bi Section", "False Position"]:
    x_true = st.number_input("Nilai true", value=24.0)
    tol = st.number_input("Toleransi (%)", value=0.1)
    xl = st.number_input("Batas bawah (xl)", value=18.0)
    xu = st.number_input("Batas atas (xu)", value=37.0)

elif metode == "Fixed Point":
    x_true = st.number_input("Nilai true", value=24.0)
    tol = st.number_input("Toleransi (%)", value=0.1)
    x0 = st.number_input("Nilai awal (x0)", value=20.0)
    lambda_val = 0.0001

elif metode == "Newton Raphson":
    x_true = st.number_input("Nilai true", value=24.0)
    tol = st.number_input("Toleransi (%)", value=0.1)
    x0 = st.number_input("Nilai awal (x0)", value=20.0)

elif metode == "Secant":
    x_true = st.number_input("Nilai true", value=24.0)
    tol = st.number_input("Toleransi (%)", value=0.1)
    x0 = st.number_input("x0", value=18.0)
    x1 = st.number_input("x1", value=37.0)

elif metode == "Modified Newton Raphson":
    x_true = st.number_input("Nilai true", value=24.0)
    tol = st.number_input("Toleransi (%)", value=0.1)
    x0 = st.number_input("Nilai awal (x0)", value=20.0)

elif metode == "Polynomial Factorization":
    pass

if st.button("Hitung"):
    steps = None
    akar = None
    if metode == "Bi Section":
        solver = BiSection(fungsi, xl, xu, x_true, max_iter, tol)
        df, steps, akar, err = solver.solve()

    elif metode == "False Position":
        solver = FalsePosition(fungsi, xl, xu, x_true, max_iter, tol)
        df, steps, akar, err = solver.solve()

    elif metode == "Fixed Point":
        solver = FixedPoint(fungsi, x0, x_true, max_iter, tol)
        df, steps, akar, err = solver.solve()

    elif metode == "Newton Raphson":
        solver = NewtonRaphson(fungsi, x0, x_true, max_iter, tol)
        df, steps, akar, err = solver.solve()

    elif metode == "Secant":
        solver = Secant(fungsi, x0, x1, x_true, max_iter, tol)
        df, steps, akar, err = solver.solve()

    elif metode == "Modified Newton Raphson":
        solver = MNewtonRaphson(fungsi, x0, x_true, max_iter, tol)
        df, steps, akar, err = solver.solve()

    elif metode == "Polynomial Factorization":
        solver = PolynomFactorization(fungsi, max_iter)
        df, steps, roots, err = solver.solve()
        akar = roots

    st.subheader("Hasil Iterasi 📋")
    if df is not None:
        st.dataframe(df, width="stretch", hide_index=True)

    if err is not None:
        st.warning(err)

    # TODO: untuk polynomial factorization, log terakhir harus dibenerin
    if steps is not None:
        st.divider()
        st.subheader("Langkah-Langkah Perhitungan 🕵️‍♂️")
        for i in range(len(steps)):
            st.markdown(steps[i])
            if i != len(steps) - 1:
                st.space("medium")

    st.divider()
    if err is None:
        st.success(f"Akar pendekatan: {akar}")
