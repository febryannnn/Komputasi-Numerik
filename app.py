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
    LinearRegression,
    QuadraticRegression,
    GaussJordan,
    Jacobi,
    GaussSeidel,
    NewtonInterpolation,
    LagrangeInterpolation,
    NewtonGregoryInterpolation,
    StirlingInterpolation,
    BesselInterpolation,
    NewtonGregoryDifferentiation,
    LagrangeDifferentiation,
    Integration,
    TrapezoidalIntegration,
    Simpson13Integration,
    Simpson38Integration,
    RiemannIntegration,
    GaussIntegration,
    Euler,
    Heunn,
    RungeKutta,
)

# [yg buat claude code] Impor komponen UI
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

# ── Kategori & Metode ──
KATEGORI = {
    "Pencarian Akar": [
        "Bi Section",
        "False Position",
        "Fixed Point",
        "Newton Raphson",
        "Secant",
        "Modified Newton Raphson",
        "Polynomial Factorization",
    ],
    "Sistem Persamaan Linear": [
        "Gauss-Jordan",
        "Jacobi",
        "Gauss-Seidel",
    ],
    "Regresi": [
        "Regresi Linear",
        "Regresi Kuadratik",
    ],
    "Interpolasi": [
        "Interpolasi Newton",
        "Interpolasi Lagrange",
        "Interpolasi Newton-Gregory",
        "Interpolasi Stirling",
        "Interpolasi Bessel",
    ],
    "Diferensiasi": [
        "Diferensiasi Newton-Gregory",
        "Diferensiasi Lagrange",
    ],
    "Integrasi": [
        "Integrasi (Exact)",
        "Trapesium",
        "Simpson 1/3",
        "Simpson 3/8",
        "Riemann",
        "Gauss",
    ],
    "ODE (Persamaan Diferensial)": [
        "Euler",
        "Heunn",
        "Runge-Kutta",
    ],
}

kategori = st.selectbox("Kategori", list(KATEGORI.keys()))
metode = st.selectbox("Pilih Metode", KATEGORI[kategori])

# [yg buat claude code] Kartu info metode
render_method_info(metode)


# ── Helper: parse data points dari text area ──
def parse_data(text):
    data = []
    for line in text.strip().split("\n"):
        parts = line.strip().split(",")
        if len(parts) >= 2:
            data.append((float(parts[0].strip()), float(parts[1].strip())))
    return data


# ── Helper: parse matrix dari text area ──
def parse_matrix(text):
    matrix = []
    for line in text.strip().split("\n"):
        row = [float(v.strip()) for v in line.strip().split(",")]
        matrix.append(row)
    return matrix


def parse_vector(text):
    return [float(v.strip()) for v in text.strip().split(",")]


# ══════════════════════════════════════════════
# INPUT PARAMETER (sesuai kategori/metode)
# ══════════════════════════════════════════════

# -- Pencarian Akar --
if kategori == "Pencarian Akar":
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
                coef = st.number_input("Koefisien", value=1.0, key=f"coef_{i}")
            with col2:
                pangkat = st.number_input(
                    "Pangkat x", min_value=0, max_value=20, value=1, key=f"pow_{i}"
                )
            if coef != 0:
                if pangkat == 0:
                    terms.append(f"{coef}")
                elif pangkat == 1:
                    terms.append(f"{coef}*x")
                else:
                    terms.append(f"{coef}*x**{pangkat}")
        fungsi = " + ".join(terms) if terms else "0"
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

# -- SPL --
elif kategori == "Sistem Persamaan Linear":
    st.subheader("Input Matriks")
    matrix_text = st.text_area(
        "Matriks A (per baris, pisah koma)", "2, 1, -1\n1, 3, 2\n1, -1, 2"
    )
    vector_text = st.text_input("Vektor B (pisah koma)", "8, 13, 7")
    if metode in ["Jacobi", "Gauss-Seidel"]:
        tol = st.number_input("Toleransi", value=0.1)
        max_iter = st.number_input("Maksimum Iterasi", value=10)

# -- Regresi --
elif kategori == "Regresi":
    st.subheader("Input Data")
    data_text = st.text_area(
        "Data (x, y per baris)",
        "1, 0.5\n2, 2.5\n3, 2.0\n4, 4.0\n5, 3.5\n6, 6.0\n7, 5.5",
    )
    if metode == "Regresi Linear":
        reg_mode = st.selectbox("Mode Regresi", ["std", "log", "exp"])

# -- Interpolasi --
elif kategori == "Interpolasi":
    st.subheader("Input Data")
    data_text = st.text_area("Data (x, y per baris)", "1, 1\n2, 8\n3, 27\n4, 64")
    x_val = st.number_input("Nilai x yang dicari", value=2.5)
    if metode in [
        "Interpolasi Newton-Gregory",
        "Interpolasi Stirling",
        "Interpolasi Bessel",
    ]:
        x0_val = st.number_input("x0 (titik acuan)", value=1.0)
        orde = st.number_input("Orde (-1 = max)", value=-1)
    if metode == "Interpolasi Newton-Gregory":
        ng_mode = st.selectbox("Mode", ["forward", "backward"])

# -- Diferensiasi --
elif kategori == "Diferensiasi":
    st.subheader("Input Data")
    data_text = st.text_area("Data (x, y per baris)", "1, 1\n2, 8\n3, 27\n4, 64")
    x_val = st.number_input("Nilai x yang dicari", value=2.5)
    if metode == "Diferensiasi Newton-Gregory":
        x0_val = st.number_input("x0 (titik acuan)", value=1.0)
        orde = st.number_input("Orde (-1 = max)", value=-1)
        diff_mode = st.selectbox("Mode", ["forward", "backward"])

# -- Integrasi --
elif kategori == "Integrasi":
    st.subheader("Input Fungsi & Batas")
    fungsi = st.text_input("Fungsi f(x)", "x**2")
    a_val = st.number_input("Batas bawah (a)", value=0.0)
    b_val = st.number_input("Batas atas (b)", value=1.0)
    if metode in ["Trapesium", "Simpson 1/3", "Riemann"]:
        n_seg = st.number_input("Jumlah segmen (0 = single)", value=0, min_value=0)
    true_val = st.number_input("Nilai true (kosongkan -1)", value=-1.0)

# -- ODE --
# -- ODE --
elif kategori == "ODE (Persamaan Diferensial)":
    st.subheader("Input ODE")
    fungsi = st.text_input("dy/dx = f(x)", "-2*x**3 + 12*x**2 - 20*x + 8.5")
    a_val = st.number_input("x awal (a)", value=0.0)
    b_val = st.number_input("x akhir (b)", value=4.0)
    h_val = st.number_input("Step size (h)", value=0.5)
    y0_val = st.number_input("Nilai awal y0", value=1.0)
    a2_val = None 
    if metode == "Runge-Kutta":
        a2_val = st.number_input("a2", value=0.5)


# ══════════════════════════════════════════════
# HITUNG
# ══════════════════════════════════════════════
if st.button("Hitung"):
    steps = None
    akar = None
    roots = None
    result = None
    df = None
    err = None

    # -- Pencarian Akar --
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

    # -- SPL --
    elif metode == "Gauss-Jordan":
        A = parse_matrix(matrix_text)
        B = parse_vector(vector_text)
        solver = GaussJordan(A, B)
        df, steps, result, err = solver.solve()
    elif metode == "Jacobi":
        A = parse_matrix(matrix_text)
        B = parse_vector(vector_text)
        solver = Jacobi(A, B, tol, max_iter)
        df, steps, result, err = solver.solve()
    elif metode == "Gauss-Seidel":
        A = parse_matrix(matrix_text)
        B = parse_vector(vector_text)
        solver = GaussSeidel(A, B, tol, max_iter)
        df, steps, result, err = solver.solve()

    # -- Regresi --
    elif metode == "Regresi Linear":
        data = parse_data(data_text)
        solver = LinearRegression(data, mode=reg_mode)
        df, steps, result, err = solver.solve()
    elif metode == "Regresi Kuadratik":
        data = parse_data(data_text)
        solver = QuadraticRegression(data)
        df, steps, result, err = solver.solve()

    # -- Interpolasi --
    elif metode == "Interpolasi Newton":
        data = parse_data(data_text)
        solver = NewtonInterpolation(data, x=x_val)
        df, steps, result, err = solver.solve()
    elif metode == "Interpolasi Lagrange":
        data = parse_data(data_text)
        solver = LagrangeInterpolation(data, x=x_val)
        df, steps, result, err = solver.solve()
    elif metode == "Interpolasi Newton-Gregory":
        data = parse_data(data_text)
        solver = NewtonGregoryInterpolation(
            data, x=x_val, x0=x0_val, orde=orde, mode=ng_mode
        )
        df, steps, result, err = solver.solve()
    elif metode == "Interpolasi Stirling":
        data = parse_data(data_text)
        solver = StirlingInterpolation(data, x=x_val, x0=x0_val, orde=orde)
        df, steps, result, err = solver.solve()
    elif metode == "Interpolasi Bessel":
        data = parse_data(data_text)
        solver = BesselInterpolation(data, x=x_val, x0=x0_val, orde=orde)
        df, steps, result, err = solver.solve()

    # -- Diferensiasi --
    elif metode == "Diferensiasi Newton-Gregory":
        data = parse_data(data_text)
        solver = NewtonGregoryDifferentiation(
            data, x=x_val, x0=x0_val, orde=orde, mode=diff_mode
        )
        df, steps, result, err = solver.solve()
    elif metode == "Diferensiasi Lagrange":
        data = parse_data(data_text)
        solver = LagrangeDifferentiation(data, x=x_val)
        df, steps, result, err = solver.solve()

    # -- Integrasi --
    elif metode == "Integrasi (Exact)":
        solver = Integration(fungsi, a_val, b_val)
        df, steps, result, err = solver.solve()
    elif metode == "Trapesium":
        solver = TrapezoidalIntegration(
            fungsi, a_val, b_val, n=n_seg, true_val=true_val
        )
        df, steps, result, err = solver.solve()
    elif metode == "Simpson 1/3":
        solver = Simpson13Integration(fungsi, a_val, b_val, n=n_seg, true_val=true_val)
        df, steps, result, err = solver.solve()
    elif metode == "Simpson 3/8":
        solver = Simpson38Integration(fungsi, a_val, b_val, true_val=true_val)
        df, steps, result, err = solver.solve()
    elif metode == "Riemann":
        solver = RiemannIntegration(fungsi, a_val, b_val, n=n_seg, true_val=true_val)
        df, steps, result, err = solver.solve()
    elif metode == "Gauss":
        solver = GaussIntegration(fungsi, a_val, b_val, true_val=true_val)
        df, steps, result, err = solver.solve()

    # -- ODE --

    elif metode == "Euler":
        solver = Euler(str(fungsi), a_val, b_val, h_val, y0_val)
        df, steps, result, err = solver.solve()
    elif metode == "Heunn":
        solver = Heunn(str(fungsi), a_val, b_val, h_val, y0_val)
        df, steps, result, err = solver.solve()
    elif metode == "Runge-Kutta":
        if a2_val is None:
            a2_val = 0.5
        solver = RungeKutta(str(fungsi), a_val, b_val, h_val, a2_val, y0_val)
        df, steps, result, err = solver.solve()

    # ══════════════════════════════════════════════
    # TAMPILKAN HASIL
    # ══════════════════════════════════════════════

    # Error-only
    if err is not None and df is None:
        st.warning(err)
        st.stop()

    # Tabel
    if df is not None:
        render_section_header("📋", "Hasil Iterasi")
        if isinstance(df, list):
            df = pd.DataFrame(df)

        st.dataframe(df, width="stretch", hide_index=True)

    if err is not None:
        st.warning(err)

    # [yg buat claude code] Visualisasi grafik Plotly (hanya untuk pencarian akar)
    if kategori == "Pencarian Akar":
        render_gradient_divider()
        render_section_header("📊", "Visualisasi")

        if metode == "Polynomial Factorization" and roots is not None:
            plot_polynomial_roots(roots, fungsi)
        elif isinstance(df, pd.DataFrame) and not df.empty and akar is not None:
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Grafik f(x)", "Konvergensi", "Error", "Gabungan"]
            )
            with tab1:
                plot_function_with_root(
                    fungsi,
                    akar,
                    metode,
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
    if steps is not None and len(steps) > 0:
        st.divider()
        render_section_header("🕵️‍♂️", "Langkah-Langkah Perhitungan")
        for i in range(len(steps)):
            st.markdown(steps[i])
            if i != len(steps) - 1:
                st.divider()

    # [yg buat claude code] Hasil akhir
    render_gradient_divider()

    if metode == "Polynomial Factorization" and roots is not None:
        render_success_result(
            "Akar-akar polinomial",
            ", ".join(
                f"x_{i+1} = {root}"
                for i, root in enumerate(roots)
                if not np.isnan(root)
            ),
        )
    elif result is not None and err is None:
        render_success_result("Hasil", result)
    elif akar is not None and err is None:
        render_success_result("Akar pendekatan", akar)
