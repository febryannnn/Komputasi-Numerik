import streamlit as st
import numpy as np
import pandas as pd
from method import (
    BiSection,
    FalsePosition,
    FixedPoint,
    NewtonRaphson,
    Secant,
    MNewtonRaphson,
    PolynomFactorization,
)

# [yg buat claude code] Impor komponen UI (styling, chart Plotly, dsb.)
from ui import (
    inject_css,
    render_header,
    render_method_info,
    render_section_header,
    render_gradient_divider,
    render_success_result,
    render_metrics,
    plot_convergence,
    plot_error,
    plot_function_with_root,
    plot_polynomial_roots,
    plot_iteration_comparison,
)

# [yg buat claude code] Page config & styling
st.set_page_config(
    page_title="Komputasi Numerik — ITS",
    page_icon="📐",
    layout="wide",
)
inject_css()
render_header()

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

# [yg buat claude code] Kartu info metode
render_method_info(metode)

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
    roots = None

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

    # [yg buat claude code] Error-only: tampilkan pesan lalu stop
    if err is not None and df is None:
        st.warning(err)
        st.stop()

    # [yg buat claude code] Tabel iterasi
    render_section_header("📋", "Hasil Iterasi")
    if isinstance(df, list):
        df = pd.DataFrame(df)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if err is not None:
        st.warning(err)

    # [yg buat claude code] Visualisasi grafik Plotly
    render_gradient_divider()
    render_section_header("📊", "Visualisasi")

    if metode == "Polynomial Factorization" and roots is not None:
        plot_polynomial_roots(roots, fungsi)
    elif isinstance(df, pd.DataFrame) and not df.empty and akar is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["Grafik f(x)", "Konvergensi", "Error", "Gabungan"])
        with tab1:
            plot_function_with_root(
                fungsi, akar, metode,
                xl=xl if metode in ["Bi Section", "False Position"] else None,
                xu=xu if metode in ["Bi Section", "False Position"] else None,
            )
        with tab2:
            plot_convergence(df, metode)
        with tab3:
            plot_error(df, metode)
        with tab4:
            plot_iteration_comparison(df)

    # Langkah-langkah perhitungan (LaTeX)
    # TODO: untuk polynomial factorization, log terakhir harus dibenerin
    if steps is not None:
        st.divider()
        render_section_header("🕵️‍♂️", "Langkah-Langkah Perhitungan")
        for i in range(len(steps)):
            st.markdown(steps[i])
            if i != len(steps) - 1:
                st.divider()

    # [yg buat claude code] Hasil akhir (styled box)
    render_gradient_divider()

    if metode == "Polynomial Factorization" and roots is not None:
        render_success_result(
            "Akar-akar polinomial",
            ", ".join(f"x_{i+1} = {root}" for i, root in enumerate(roots) if not np.isnan(root)),
        )

    st.divider()
    if err is None:
        render_success_result("Akar pendekatan", akar)
