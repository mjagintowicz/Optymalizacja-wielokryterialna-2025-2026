import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from problem_definition import objective_F1, objective_F2, space_properties
from weights import compute_pareto_points_weights
from eps_cons import compute_pareto_epsilon
from ideal_point import compute_pareto_ideal_point
from visualizations import plot_pareto


st.title("Metody skalaryzacji")


# --- SKALARYZACJA PRZEZ FUNKCJĘ LINIOWĄ ---
st.header("Skalaryzacja przez funkcję liniową")
n = st.slider("Liczba wag (dokładność frontu Pareto)", 10, 200, 100)

if st.button("Oblicz", key="linear"):
    with st.spinner("Obliczam..."):
        pareto, cx, cy, tx, ty = compute_pareto_points_weights(n)
        fig = plot_pareto(pareto, cx, cy, tx, ty)
        st.session_state["figure_linear"] = fig

if "figure_linear" in st.session_state:
    st.pyplot(st.session_state["figure_linear"])


# -- SKALARYZCJA METODĄ EPS-OGRANICZEŃ ---
st.header("Skalaryzacja metodą ε-ograniczeń")

n = st.slider("Liczba punktów ε:", 5, 100, 20)
eps_min = st.number_input("Minimalne ε:", value=17.0)
eps_max = st.number_input("Maksymalne ε:", value=30.0)

if st.button("Oblicz", key="epsilon"):
    with st.spinner("Obliczam..."):
        pareto, cx, cy, tx, ty = compute_pareto_epsilon(n, eps_min, eps_max)
        fig = plot_pareto(pareto, cx, cy, tx, ty)
        st.session_state["figure_eps"] = fig

if "figure_eps" in st.session_state:
    st.pyplot(st.session_state["figure_eps"])


# --- SKALARYZCJA PRZEZ ODLEGŁOŚĆ OD WYBRANEGO PKT DOMINUJĄCEGO ---
st.header("Skalaryzacja przez odległość")

n = st.slider("Liczba wag:", 5, 200, 100)
w_min = st.number_input("Minimalne w:", value=0.1)
w_max = st.number_input("Maksymalne w:", value=1.0)

if st.button("Oblicz", key="distance"):
    with st.spinner("Obliczam..."):
        pareto, cx, cy, tx, ty, x_star = compute_pareto_ideal_point(n, w_min, w_max)
        fig = plot_pareto(pareto, cx, cy, tx, ty, x_star)
        st.session_state["figure_dist"] = fig

if "figure_dist" in st.session_state:
    st.pyplot(st.session_state["figure_dist"])