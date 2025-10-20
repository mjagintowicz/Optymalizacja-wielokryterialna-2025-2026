import streamlit as st
import ast
import plotly.express as px
from alg_bez_filtra import algorithm_no_filter
from pareto_naive_filt import get_P_front
from pareto_naive_ideal_pt import find_non_dominated_points
import pandas as pd
import numpy as np

# -----------------------------
# Tytuł i opis
# -----------------------------
st.title("Punkty niezdominowane")
st.write(
    "Znajdź punkty niezdominowane (front Pareto) z możliwością wyboru kryterium i benchmarkami"
)

# -----------------------------
# Wybór algorytmu
# -----------------------------
algorithm_choice = st.radio(
    "Wybierz algorytm:",
    (
        "Algorytm bez filtra (algorithm_no_filter)",
        "Algorytm z filtrem (find_non_dominated_points)",
        "Algorytm z punktem idealnym (algorithm_ideal_point)"
    )
)

# -----------------------------
# Wybór źródła danych
# -----------------------------
data_source = st.radio("Wybierz źródło danych:", ("Wpisane ręcznie", "Wygenerowane"))

X_final = []
num_dims = 2  # default, nadpisze się później

# -----------------------------
# Generator danych - dynamicznie
# -----------------------------
if data_source == "Wygenerowane":
    with st.expander("Generator danych testowych", expanded=True):
        num_points = st.number_input("Liczba punktów", min_value=1, value=20, step=1)
        num_dims = st.number_input("Liczba wymiarów/kryteriów", min_value=2, value=2, step=1)
        dist_type = st.selectbox("Rozkład danych", ["normalny", "jednolity"])
        int_only = st.checkbox("Tylko wartości całkowite", value=False)

        if dist_type == "normalny":
            mean = st.number_input("Średnia (μ)", value=5.0)
            std = st.number_input("Odchylenie standardowe (σ)", min_value=0.1, value=2.0)
        else:  # jednorodny
            min_val = st.number_input("Min", value=0.0)
            max_val = st.number_input("Max", value=10.0)

        if st.button("Generuj dane", key="generate_data"):
            if dist_type == "normalny":
                data = np.random.normal(loc=mean, scale=std, size=(num_points, num_dims))
            else:
                data = np.random.uniform(low=min_val, high=max_val, size=(num_points, num_dims))

            if int_only:
                data = np.round(data).astype(int)

            X_final = [tuple(row) for row in data]
            st.session_state.X_generated = X_final

    # Wyświetlenie wygenerowanych danych, jeśli już istnieją w sesji
    if "X_generated" in st.session_state and st.session_state.X_generated:
        X_final = st.session_state.X_generated
        st.subheader("Wygenerowane punkty")
        df = pd.DataFrame(X_final, columns=[f"Wymiar {i+1}" for i in range(num_dims)])
        st.dataframe(df.style.background_gradient(cmap="Blues", axis=None).format(precision=2))
    else:
        st.warning("Najpierw wygeneruj dane w generatorze powyżej.")

# -----------------------------
# Dane wpisane ręcznie
# -----------------------------
elif data_source == "Wpisane ręcznie":
    default_points = "[(5,5), (3,6), (4,4), (5,3), (3,3), (1,8), (3,4), (4,5), (3,10), (6,6), (4,1), (3,5)]"
    points_input = st.text_area("Podaj listę punktów (np. [(5,5),(3,6)])", default_points)

    try:
        X_manual = ast.literal_eval(points_input)
        if not isinstance(X_manual, list) or not all(isinstance(p, tuple) for p in X_manual):
            st.error("Wprowadź dane jako listę tupli")
            st.stop()
        else:
            X_final = X_manual
            num_dims = len(X_final[0])
    except Exception as e:
        st.error(f"Błąd danych: {e}")
        st.stop()

# -----------------------------
# Wybór min/max dla wymiarów
# -----------------------------
if X_final:
    st.subheader("Wybierz kierunek dla każdego wymiaru")
    directions = []
    cols = st.columns(len(X_final[0]))
    for i in range(len(X_final[0])):
        with cols[i]:
            dir_choice = st.selectbox(f"Wymiar {i+1}", ["min", "max"], index=0)
            directions.append(dir_choice)

    # -----------------------------
    # Obliczanie punktów niezdominowanych
    # -----------------------------
    if st.button("Znajdź punkty niezdominowane", key='find_non_dom_points'):
        if algorithm_choice == "Algorytm bez filtra (algorithm_no_filter)":
            P, comparisons = algorithm_no_filter(X_final.copy(), directions)
        elif algorithm_choice == "Algorytm z filtrem (find_non_dominated_points)":
            P, comparisons = get_P_front(X_final.copy(), directions)
        else:
            P, comparisons = find_non_dominated_points(X_final.copy(), directions)

        st.success(f"Znaleziono {len(P)} punktów niezdominowanych po {comparisons} porównaniach.")

        st.subheader("Punkty niezdominowane")
        print(P)
        df = pd.DataFrame(P, columns=[f"Wymiar {i+1}" for i in range(len(directions))])
        st.dataframe(df.style.background_gradient(cmap="Greens", axis=None).format(precision=2))

        # -----------------------------
        # Wizualizacja 2D/3D
        # -----------------------------
        if len(X_final[0]) == 2:
            fig = px.scatter(x=[p[0] for p in X_final], y=[p[1] for p in X_final],
                             labels={'x':'Wymiar 1','y':'Wymiar 2'}, title="Front Pareto (2D)")
            fig.add_scatter(x=[p[0] for p in P], y=[p[1] for p in P],
                            mode='markers', marker=dict(color='red', size=10, symbol='diamond'),
                            name='Niezdominowane')
            st.plotly_chart(fig, use_container_width=True)

        elif len(X_final[0]) == 3:
            fig = px.scatter_3d(x=[p[0] for p in X_final],
                                y=[p[1] for p in X_final],
                                z=[p[2] for p in X_final],
                                labels={'x':'Wymiar 1','y':'Wymiar 2','z':'Wymiar 3'},
                                title="Front Pareto (3D)")
            fig.add_scatter3d(x=[p[0] for p in P], y=[p[1] for p in P], z=[p[2] for p in P],
                              mode='markers', marker=dict(color='red', size=6, symbol='diamond'),
                              name='Niezdominowane')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Wizualizacja dostępna tylko dla danych 2D lub 3D.")