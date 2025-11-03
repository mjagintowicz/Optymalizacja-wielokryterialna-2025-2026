import streamlit as st
import ast
import plotly.express as px
import plotly.graph_objects as go
from alg_bez_filtra import algorithm_no_filter
from pareto_naive_filt import get_P_front
from pareto_naive_ideal_pt import find_non_dominated_points
from klp_pareto import klp_pareto
import pandas as pd
import numpy as np
import time
import statsmodels.api as sm

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
        "Algorytm z punktem idealnym (algorithm_ideal_point)",
        "Algorytm Kung, Lucio & Preparata"
    )
)

# -----------------------------
# Wybór źródła danych
# -----------------------------
data_source = st.radio("Wybierz źródło danych:", ("Wpisane ręcznie", "Wygenerowane"))

X_final = []
num_dims = 2

# -----------------------------
# Generator danych - dynamicznie
# -----------------------------
if data_source == "Wygenerowane":
    with st.expander("Generator danych testowych", expanded=True):
        num_points = st.number_input("Liczba punktów", min_value=1, value=20, step=1, key='np1')
        num_dims = st.number_input("Liczba wymiarów/kryteriów", min_value=2, value=2, step=1,
                                   key='nd1')
        dist_type = st.selectbox("Rozkład danych", ["normalny", "jednostajny"], key='sb_dist_type')
        int_only = st.checkbox("Tylko wartości całkowite", key='cb1', value=False)

        if dist_type == "normalny":
            mean = st.number_input("Średnia (μ)", value=5.0, key='m1')
            std = st.number_input("Odchylenie standardowe (σ)", min_value=0.1, value=2.0,
                                  key='std1')
        else:
            min_val = st.number_input("Min", value=0.0, key='mv1')
            max_val = st.number_input("Max", value=10.0, key='mx1')

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
            dir_choice = st.selectbox(f"Wymiar {i+1}", ["min", "max"], key=f'sb_dir_choice_{i}', index=0)
            directions.append(dir_choice)

    # -----------------------------
    # Obliczanie punktów niezdominowanych
    # -----------------------------
    if st.button("Znajdź punkty niezdominowane", key='find_non_dom_points'):
        if algorithm_choice == "Algorytm bez filtra (algorithm_no_filter)":
            P, comparisons_p, comparisons_c = algorithm_no_filter(X_final.copy(), directions)
        elif algorithm_choice == "Algorytm z filtrem (find_non_dominated_points)":
            P, comparisons_p, comparisons_c = get_P_front(X_final.copy(), directions)
        elif algorithm_choice == "Algorytm z filtrem (find_non_dominated_points)":
            P, comparisons_p, comparisons_c = find_non_dominated_points(X_final.copy(), directions)
        else:
            P, comparisons_p, comparisons_c = klp_pareto(X_final.copy(), directions)

        st.success(f"Znaleziono {len(P)} punktów niezdominowanych po {comparisons_p} porównaniach"
                   f" punktów i {comparisons_c} porównaniach współrzędnych.")

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


st.title("Benchmarkowanie algorytmów Pareto")

st.write("Porównaj algorytmy wyszukiwania punktów niezdominowanych "
         "dla różnych rozkładów danych i liczby wymiarów.")


# -----------------------------
# Parametry benchmarku
# -----------------------------
st.title("Benchmark algorytmów Pareto")

n_iter = st.number_input("Liczba powtórzeń benchmarku", min_value=1, value=10, step=1)


num_points = st.number_input("Liczba punktów", min_value=1, value=50, key='np2')
num_dims = st.number_input("Liczba wymiarów/kryteriów", min_value=2, value=3, key='nd2')
dist_type = st.selectbox("Rozkład danych", ["normalny", "jednostajny"], key='sb_dtype')
int_only = st.checkbox("Tylko wartości całkowite", key='cb2', value=False)

if dist_type == "normalny":
    mean = st.number_input("Średnia (μ)", value=5.0, key='m2')
    std = st.number_input("Odchylenie standardowe (σ)", min_value=0.1, value=2.0, key='std2')
else:
    min_val = st.number_input("Min", value=0.0, key='mv2')
    max_val = st.number_input("Max", value=10.0, key='mx2')

directions = ["min"] * num_dims



# -----------------------------
# Benchmark
# -----------------------------
if st.button("Uruchom benchmark"):
    algos = {
        "Bez filtra": algorithm_no_filter,
        "Z filtrem": get_P_front,
        "Punkt idealny": find_non_dominated_points,
        "KLP": klp_pareto
    }

    # -----------------------------
    # Podsumowanie konfiguracji benchmarku – wersja kompaktowa
    # -----------------------------
    st.markdown("""
    <style>
    .small-metric {
        font-size: 14px;
        text-align: center;
    }
    .small-metric strong {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("Parametry eksperymentu")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"<div class='small-metric'><strong>{n_iter}</strong><br>Liczba powtórzeń</div>",
            unsafe_allow_html=True)
    with col2:
        if dist_type == "normalny":
            st.markdown(
                f"<div class='small-metric'><strong>{dist_type.capitalize()}</strong><br>μ={mean}, σ={std}</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='small-metric'><strong>{dist_type.capitalize()}</strong><br>[{min_val}, {max_val}]</div>",
                unsafe_allow_html=True)
    with col3:
        st.markdown(
            f"<div class='small-metric'><strong>{num_dims}</strong><br>Liczba kryteriów</div>",
            unsafe_allow_html=True)
    with col4:
        st.markdown(
            f"<div class='small-metric'><strong>{num_points}</strong><br>Liczba punktów</div>",
            unsafe_allow_html=True)

    results_all = {
        name: {"nondominated": [], "comparisons_points": [], "comparisons_coords": [], "time": []}
        for name in algos.keys()}

    for i in range(n_iter):
        if dist_type == "normalny":
            data = np.random.normal(loc=mean, scale=std, size=(num_points, num_dims))
        else:
            data = np.random.uniform(low=min_val, high=max_val, size=(num_points, num_dims))
        if int_only:
            data = np.round(data).astype(int)
        X = [tuple(row) for row in data]

        for name, func in algos.items():
            start = time.time()
            P, cmp_points, cmp_coords = func(X, directions)
            end = time.time()

            results_all[name]["nondominated"].append(len(P))
            results_all[name]["comparisons_points"].append(cmp_points)
            results_all[name]["comparisons_coords"].append(cmp_coords)
            results_all[name]["time"].append(end - start)

    results_dict = {}
    for name, results in results_all.items():
        results_dict[name] = [
            f"{np.mean(results['nondominated']):.2f} ± {np.std(results['nondominated']):.2f}",
            f"{np.mean(results['comparisons_points']):.2f} ± {np.std(results['comparisons_points']):.2f}",
            f"{np.mean(results['comparisons_coords']):.2f} ± {np.std(results['comparisons_coords']):.2f}",
            f"{np.mean(results['time']):.4f} ± {np.std(results['time']):.4f}"
        ]

    summary_df = pd.DataFrame(
        results_dict,
        index=["Liczba punktów ND", "Porównania punktów", "Porównania współrzędnych", "Czas (s)"]
    )

    st.subheader("Porównanie wszystkich algorytmów")
    st.dataframe(summary_df.style.format(precision=4))


# Wykresy regresji
st.title("Regresja i przedział ufności")

n_iter = st.number_input("Liczba powtórzeń benchmarku", min_value=1, value=50, step=1)
num_points = st.number_input("Liczba punktów", min_value=2, value=100, step=1)
num_dims = st.number_input("Liczba wymiarów/kryteriów", min_value=2, value=5, step=1)

dist_type = st.selectbox("Rozkład danych", ["normalny", "jednostajny"])
int_only = st.checkbox("Tylko wartości całkowite", value=False)

if dist_type == "normalny":
    mean = st.number_input("Średnia (μ)", value=5.0)
    std = st.number_input("Odchylenie standardowe (σ)", min_value=0.1, value=2.0)
else:
    min_val = st.number_input("Min", value=0.0)
    max_val = st.number_input("Max", value=10.0)

algos = {
    "Bez filtra": algorithm_no_filter,
    "Z filtrem": get_P_front,
    "Punkt idealny": find_non_dominated_points,
    "KLP": klp_pareto
}
selected_algo_name = st.selectbox("Wybierz algorytm:", list(algos.keys()))
selected_algo = algos[selected_algo_name]

# --- Generowanie danych ---
if st.button("Generuj dane", key="btn_gen"):
    if dist_type == "normalny":
        data = np.random.normal(loc=mean, scale=std, size=(num_points, num_dims))
    else:
        data = np.random.uniform(low=min_val, high=max_val, size=(num_points, num_dims))

    if int_only:
        data = np.round(data).astype(int)

    st.session_state.X_generated = [tuple(row) for row in data]
    st.session_state.num_dims = num_dims
    st.session_state.selected_algo_name = selected_algo_name

    st.success("Dane wygenerowane — wybierz kierunki kryteriów")

# --- Wybór MIN/MAX dla każdego wymiaru ---
if "X_generated" in st.session_state:
    st.subheader("Kierunek optymalizacji dla kryteriów")

    directions = []
    cols = st.columns(st.session_state.num_dims)

    for i in range(st.session_state.num_dims):
        with cols[i]:
            d = st.selectbox(f"Kryterium {i+1}", ["min", "max"], key=f"dir_{i}")
            directions.append(d)

    st.session_state.directions = directions

# --- Benchmark + Regresja ---
if "X_generated" in st.session_state and "directions" in st.session_state:

    if st.button("Uruchom benchmark", key="btn_run"):

        X_final = st.session_state.X_generated
        num_dims = st.session_state.num_dims
        directions = st.session_state.directions
        selected_algo = algos[st.session_state.selected_algo_name]

        dims_range = list(range(num_dims, 1, -1))
        mean_results = []

        for d in dims_range:
            results = []
            for _ in range(n_iter):
                X_cut = [tuple(p[:d]) for p in X_final]
                dirs_cut = directions[:d]

                P_cut, _, _ = selected_algo(X_cut.copy(), dirs_cut)
                results.append(len(P_cut))

            mean_results.append(np.mean(results))

        dims_range.reverse()
        mean_results.reverse()

        df_nd = pd.DataFrame({"Kryteria": dims_range, "Średnia ND": mean_results})

        # Regressja
        x = np.array(df_nd["Kryteria"])
        y = np.array(df_nd["Średnia ND"])

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        pred = model.get_prediction(X)
        pred_summary = pred.summary_frame(alpha=0.05)

        # --- Wykres ---
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name="Średnia liczba punktów ND"))

        fig.add_trace(go.Scatter(
            x=x, y=pred_summary['mean'],
            mode='lines', name="Regresja"
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% przedział ufności'
        ))

        fig.update_layout(
            title=f"Wynik regresji dla algorytmu: {st.session_state.selected_algo_name}",
            xaxis_title="Liczba kryteriów",
            yaxis_title="Średnia liczba punktów niezdominowanych"
        )

        st.plotly_chart(fig, use_container_width=True)