import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import altair as alt
import qrcode
import socket


def range_report(name, val, r):
    lo, hi = r
    if lo == hi:
        in_range = (val == lo)
        if in_range:
            msg = f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] (constant) | OK (equal to data)"
        else:
            msg = (
                f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] (constant) | "
                f"OUT (diff = {abs(val-lo):.5g})"
            )
        return in_range, msg

    if val < lo:
        in_range = False
        dist = lo - val
        msg = (
            f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] | "
            f"OUT ({dist:.5g} under min)"
        )
    elif val > hi:
        in_range = False
        dist = val - hi
        msg = (
            f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] | "
            f"OUT ({dist:.5g} over max)"
        )
    else:
        in_range = True
        width = hi - lo
        pos_pct = (val - lo) / width * 100
        margin_min = val - lo
        margin_max = hi - val
        msg = (
            f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] | "
            f"OK (position: {pos_pct:.1f}% of range; margins: +{margin_min:.5g} / {margin_max:.5g})"
        )

    return in_range, msg


def get_local_ip():
    """Return local IP address (for QR code)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "localhost"
    finally:
        s.close()
    return ip


st.title("Previsione P per additivi (Streamlit)")
st.write(
    """
Replica dello script MATLAB:
- legge TabellaAdditivi.txt
- chiede E, SR, epsr
- per ogni additivo allena LSBoost, Bagging, Lineare, Polinomiale
- mostra Predizione, MAE, RMSE, R2.
"""
)

st.sidebar.header("Input utente")

E_input = st.sidebar.number_input("Valore di E", value=0.0, format="%.6f")
SR_input = st.sidebar.number_input("Valore di SR", value=0.0, format="%.6f")
epsr_input = st.sidebar.number_input("Valore di epsr", value=0.0, format="%.6f")

st.sidebar.write("---")
esegui = st.sidebar.button("Esegui analisi")

# QR code nella sidebar
with st.sidebar.expander("QR code per smartphone"):
    ip = get_local_ip()
    url = f"http://{ip}:8501"
    st.write("PC e telefono devono essere sulla stessa rete WiFi.")
    st.write("URL:", url)

    # QR piu grande e con sfondo bianco
    qr = qrcode.QRCode(
        version=1,
        box_size=8,   # aumenta per ingrandire il QR
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")

    qr_array = np.array(qr_img.convert("RGB"))
    st.image(qr_array, caption="Scansiona con la fotocamera", use_column_width=True)




@st.cache_data
def carica_dati(filename):
    """Legge il file di testo gestendo encoding tipici Windows."""
    # Prova cp1252 (tipico Windows)
    try:
        df = pd.read_csv(filename, sep=None, engine="python", encoding="cp1252")
        return df
    except UnicodeDecodeError:
        pass

    # Prova latin1
    try:
        df = pd.read_csv(filename, sep=None, engine="python", encoding="latin1")
        return df
    except UnicodeDecodeError:
        pass

    # Fallback: utf-8 ignorando caratteri che danno problemi
    df = pd.read_csv(
        filename,
        sep=None,
        engine="python",
        encoding="utf-8",
        errors="ignore",
    )
    return df


if esegui:
    filename = "TabellaAdditivi.txt"
    try:
        data = carica_dati(filename)
    except FileNotFoundError:
        st.error(f'Il file "{filename}" non e stato trovato nella cartella corrente.')
        st.stop()
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
        st.stop()

    req_vars = ["tipo", "P", "E", "SR", "epsr"]
    if not all(col in data.columns for col in req_vars):
        st.error(f"La tabella deve contenere le colonne: {', '.join(req_vars)}")
        st.write("Colonne trovate:", list(data.columns))
        st.stop()

    st.success("Dati caricati correttamente.")
    with st.expander("Anteprima dati"):
        st.dataframe(data.head())

    additivi = sorted(data["tipo"].astype(str).unique())
    risultati_records = []

    st.write("### Log analisi per additivo")

    for nome_add in additivi:
        sotto = data[data["tipo"].astype(str) == nome_add].copy()

        X = sotto[["E", "SR", "epsr"]].values
        y = sotto["P"].values
        N = X.shape[0]

        st.markdown(f"---\n#### Additivo: `{nome_add}` (N = {N})")

        range_E = (np.min(X[:, 0]), np.max(X[:, 0]))
        range_SR = (np.min(X[:, 1]), np.max(X[:, 1]))
        range_epsr = (np.min(X[:, 2]), np.max(X[:, 2]))

        inE, msgE = range_report("E", E_input, range_E)
        inSR, msgSR = range_report("SR", SR_input, range_SR)
        inEP, msgEP = range_report("epsr", epsr_input, range_epsr)

        st.write("**Valutazione range:**")
        st.text(msgE + "\n" + msgSR + "\n" + msgEP)

        in_range_all = inE and inSR and inEP

        if not in_range_all:
            st.warning(f"Input FUORI RANGE per [{nome_add}]. Additivo saltato.")
            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "(skipped)",
                    "Predizione": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": np.nan,
                    "InRange": 0,
                }
            )
            continue
        else:
            st.info(f"Input nel range per [{nome_add}]. Procedo con le previsioni...")

        if N < 3:
            st.warning(f"Troppi pochi dati (N={N}) per K-fold affidabile. Skip.")
            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "(too-few)",
                    "Predizione": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": np.nan,
                    "InRange": 1,
                }
            )
            continue

        K = min(5, max(2, N))
        cv = KFold(n_splits=K, shuffle=True, random_state=42)
        input_nuovo = np.array([[E_input, SR_input, epsr_input]])

        base_tree = DecisionTreeRegressor(max_leaf_nodes=21, random_state=42)

        def cv_predict(model, X, y, cv):
            y_pred_cv = np.full_like(y, np.nan, dtype=float)
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]
                mdl = model
                mdl.fit(X_train, y_train)
                y_pred_cv[test_idx] = mdl.predict(X_test)
            return y_pred_cv

        # LSBoost ~ GradientBoostingRegressor
        try:
            mdl_ls = GradientBoostingRegressor(
                loss="squared_error", n_estimators=100, random_state=42
            )
            y_pred_ls_cv = cv_predict(
                GradientBoostingRegressor(
                    loss="squared_error", n_estimators=100, random_state=42
                ),
                X,
                y,
                cv,
            )
            mdl_ls.fit(X, y)
            pred_ls = mdl_ls.predict(input_nuovo)[0]

            MAE_ls = np.mean(np.abs(y - y_pred_ls_cv))
            RMSE_ls = np.sqrt(np.mean((y - y_pred_ls_cv) ** 2))
            R2_ls = 1 - np.sum((y - y_pred_ls_cv) ** 2) / np.sum((y - np.mean(y)) ** 2)

            st.write("**Risultati LSBoost (GradientBoostingRegressor)**")
            st.write(
                f"Prev: {pred_ls:.2f} | MAE: {MAE_ls:.2f} | RMSE: {RMSE_ls:.2f} | R2: {R2_ls:.3f}"
            )

            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "LSBoost",
                    "Predizione": pred_ls,
                    "MAE": MAE_ls,
                    "RMSE": RMSE_ls,
                    "R2": R2_ls,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )
        except Exception as e:
            st.warning(f"LSBoost fallito per {nome_add}: {e}")
            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "LSBoost (err)",
                    "Predizione": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )

        # Bagging
        try:
            mdl_bag = BaggingRegressor(
                estimator=base_tree, n_estimators=100, random_state=42
            )
            y_pred_bag_cv = cv_predict(
                BaggingRegressor(
                    estimator=base_tree, n_estimators=100, random_state=42
                ),
                X,
                y,
                cv,
            )
            mdl_bag.fit(X, y)
            pred_bag = mdl_bag.predict(input_nuovo)[0]

            MAE_bag = np.mean(np.abs(y - y_pred_bag_cv))
            RMSE_bag = np.sqrt(np.mean((y - y_pred_bag_cv) ** 2))
            R2_bag = 1 - np.sum((y - y_pred_bag_cv) ** 2) / np.sum(
                (y - np.mean(y)) ** 2
            )

            st.write("**Risultati Bagging**")
            st.write(
                f"Prev: {pred_bag:.2f} | MAE: {MAE_bag:.2f} | RMSE: {RMSE_bag:.2f} | R2: {R2_bag:.3f}"
            )

            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "Bagging",
                    "Predizione": pred_bag,
                    "MAE": MAE_bag,
                    "RMSE": RMSE_bag,
                    "R2": R2_bag,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )
        except Exception as e:
            st.warning(f"Bagging fallito per {nome_add}: {e}")
            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "Bagging (err)",
                    "Predizione": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )

        # Lineare
        try:
            y_pred_lin_cv = np.full_like(y, np.nan, dtype=float)
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]
                mdl_lin_k = LinearRegression(fit_intercept=True)
                mdl_lin_k.fit(X_train, y_train)
                y_pred_lin_cv[test_idx] = mdl_lin_k.predict(X_test)

            mdl_lin = LinearRegression(fit_intercept=True)
            mdl_lin.fit(X, y)
            pred_lin = mdl_lin.predict(input_nuovo)[0]
            pred_lin = np.clip(pred_lin, 0, 100)

            MAE_lin = np.mean(np.abs(y - y_pred_lin_cv))
            RMSE_lin = np.sqrt(np.mean((y - y_pred_lin_cv) ** 2))
            R2_lin = 1 - np.sum((y - y_pred_lin_cv) ** 2) / np.sum(
                (y - np.mean(y)) ** 2
            )

            st.write("**Risultati Lineare**")
            st.write(
                f"Prev: {pred_lin:.2f} | MAE: {MAE_lin:.2f} | RMSE: {RMSE_lin:.2f} | R2: {R2_lin:.3f}"
            )

            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "Lineare",
                    "Predizione": pred_lin,
                    "MAE": MAE_lin,
                    "RMSE": RMSE_lin,
                    "R2": R2_lin,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )
        except Exception as e:
            st.warning(f"Lineare fallita per {nome_add}: {e}")
            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "Lineare (err)",
                    "Predizione": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )

        # Polinomiale 2 ordine
        try:
            poly_model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("linreg", LinearRegression(fit_intercept=True)),
                ]
            )

            y_pred_poly_cv = np.full_like(y, np.nan, dtype=float)
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]

                mdl_poly_k = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                        ("linreg", LinearRegression(fit_intercept=True)),
                    ]
                )
                mdl_poly_k.fit(X_train, y_train)
                y_pred_poly_cv[test_idx] = mdl_poly_k.predict(X_test)

            poly_model.fit(X, y)
            pred_poly = poly_model.predict(input_nuovo)[0]
            pred_poly = np.clip(pred_poly, 0, 100)

            MAE_poly = np.mean(np.abs(y - y_pred_poly_cv))
            RMSE_poly = np.sqrt(np.mean((y - y_pred_poly_cv) ** 2))
            R2_poly = 1 - np.sum((y - y_pred_poly_cv) ** 2) / np.sum(
                (y - np.mean(y)) ** 2
            )

            st.write("**Risultati Polinomiale (2 ordine)**")
            st.write(
                f"Prev: {pred_poly:.2f} | MAE: {MAE_poly:.2f} | RMSE: {RMSE_poly:.2f} | R2: {R2_poly:.3f}"
            )

            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "Polinomiale",
                    "Predizione": pred_poly,
                    "MAE": MAE_poly,
                    "RMSE": RMSE_poly,
                    "R2": R2_poly,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )
        except Exception as e:
            st.warning(f"Polinomiale fallito per {nome_add}: {e}")
            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Modello": "Polinomiale (err)",
                    "Predizione": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )

    if len(risultati_records) > 0:
        risultati = pd.DataFrame(risultati_records)

        col_order = [
            "Additivo",
            "Modello",
            "N",
            "K",
            "InRange",
            "Predizione",
            "MAE",
            "RMSE",
            "R2",
        ]
        risultati = risultati[col_order]

        risultati["is_nan_rmse"] = risultati["RMSE"].isna()
        risultati = risultati.sort_values(
            by=["is_nan_rmse", "RMSE"], ascending=[True, True]
        ).drop(columns=["is_nan_rmse"])

        st.write("## Riepilogo risultati (ordinati per RMSE crescente)")
        st.dataframe(risultati)

        mods_to_plot = ["LSBoost", "Bagging", "Lineare", "Polinomiale"]
        mask = (
            risultati["Modello"].isin(mods_to_plot)
            & (risultati["InRange"] == 1)
            & (~risultati["RMSE"].isna())
        )
        ris_plot = risultati.loc[mask, ["Additivo", "Modello", "RMSE"]]

        if not ris_plot.empty:
            st.write("### Confronto modelli per additivo (RMSE piu basso = migliore)")

            chart = (
                alt.Chart(ris_plot)
                .mark_bar()
                .encode(
                    x=alt.X("Additivo:N", title="Additivo"),
                    y=alt.Y("RMSE:Q", title="RMSE"),
                    color="Modello:N",
                    column=alt.Column("Modello:N", title="Modello"),
                    tooltip=["Additivo", "Modello", "RMSE"],
                )
                .properties(height=200)
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Nessun dato valido per costruire il grafico di confronto RMSE.")
