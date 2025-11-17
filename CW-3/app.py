import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from problem_definition import objective_F1, objective_F2, space_properties
from weights import compute_pareto_points_weights
from weights_3d import compute_pareto_weights_3d
from eps_cons import compute_pareto_epsilon
from eps_cons_3d import compute_pareto_epsilon_3d
from ideal_point import compute_pareto_ideal_point
from ideal_point_3d import compute_pareto_ideal_point_3d
from visualizations import plot_pareto, plot_pareto_3d


st.title("Metody skalaryzacji")

# --- SKALARYZACJA PRZEZ FUNKCJĘ LINIOWĄ ---
dimension = st.selectbox("Wybierz wymiar:", ("2D", "3D"), key='linear_box')

n = st.slider("Liczba wag (dokładność frontu Pareto)", 10, 200, 50)

if st.button("Oblicz", key="linear"):
    with st.spinner("Obliczam..."):
        if dimension == "2D":
            pareto, cx, cy, tx, ty = compute_pareto_points_weights(n)
            fig = plot_pareto(pareto, cx, cy, tx, ty)
            st.session_state["figure_linear"] = fig
        else:  # 3D
            pareto_dec, pareto_obj, sx, sy, sz, sG1, sG2, sG3 = compute_pareto_weights_3d(n_weights=n)
            fig = plot_pareto_3d(pareto_dec, pareto_obj, sx, sy, sz, sG1, sG2, sG3)
            st.session_state["figure_linear_3d"] = fig

# --- Wyświetlanie wykresów ---
if dimension == "2D" and "figure_linear" in st.session_state:
    st.pyplot(st.session_state["figure_linear"])
elif dimension == "3D" and "figure_linear_3d" in st.session_state:
    st.pyplot(st.session_state["figure_linear_3d"])


# -- SKALARYZCJA METODĄ EPS-OGRANICZEŃ ---
st.header("Skalaryzacja metodą ε-ograniczeń")

dimension = st.selectbox("Wybierz wymiar:", ("2D", "3D"), key="eps_box")
n = st.slider("Liczba punktów ε:", 5, 100, 20)
eps_min = st.number_input("Minimalne ε:", value=20.0)
eps_max = st.number_input("Maksymalne ε:", value=40.0)

if st.button("Oblicz", key="epsilon"):
    with st.spinner("Obliczam..."):
        if dimension == "2D":
            pareto, cx, cy, tx, ty = compute_pareto_epsilon(n, eps_min, eps_max)
            fig = plot_pareto(pareto, cx, cy, tx, ty)
            st.session_state["figure_eps"] = fig
        else:
            pareto_dec, pareto_obj, sx, sy, sz, sG1, sG2, sG3 = compute_pareto_epsilon_3d(n=n, eps_min=eps_min,
                                                                                          eps_max=eps_max)
            fig = plot_pareto_3d(pareto_dec, pareto_obj, sx, sy, sz, sG1, sG2, sG3)
            st.session_state["figure_eps_3d"] = fig

if dimension == "2D" and "figure_eps" in st.session_state:
    st.pyplot(st.session_state["figure_eps"])
elif dimension == "3D" and "figure_eps_3d" in st.session_state:
    st.pyplot(st.session_state["figure_eps_3d"])


# --- SKALARYZCJA PRZEZ ODLEGŁOŚĆ OD WYBRANEGO PKT DOMINUJĄCEGO ---
st.header("Skalaryzacja przez odległość")

dimension = st.selectbox("Wybierz wymiar:", ("2D", "3D"), key="dist_box")
n = st.slider("Liczba wag:", 5, 200, 100)

if dimension == "2D":
    w_min = st.number_input("Minimalne w:", value=0.1)
    w_max = st.number_input("Maksymalne w:", value=1.0)

if st.button("Oblicz", key="distance"):
    with st.spinner("Obliczam..."):
        if dimension == "2D":
            pareto, cx, cy, tx, ty, x_star = compute_pareto_ideal_point(n, w_min, w_max)
            fig = plot_pareto(pareto, cx, cy, tx, ty, x_star)
            st.session_state["figure_dist"] = fig
        else:
            pareto_dec, pareto_obj, sx, sy, sz, sG1, sG2, sG3, ideal_point, res_g1, res_g2, res_g3 = compute_pareto_ideal_point_3d(n_weights=n)
            fig = plot_pareto_3d(pareto_dec, pareto_obj, sx, sy, sz, sG1, sG2, sG3, ideal_point, res_g1, res_g2, res_g3)
            st.session_state["figure_dist_3d"] = fig

if dimension == "2D" and "figure_dist" in st.session_state:
    st.pyplot(st.session_state["figure_dist"])
elif dimension == "3D" and "figure_dist_3d" in st.session_state:
    st.pyplot(st.session_state["figure_dist_3d"])



