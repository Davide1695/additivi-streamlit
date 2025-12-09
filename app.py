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


# --------- mappa sigla additivo -> descrizione estesa ---------
ADDITIVO_DESCR = {
    "MW": "scarti di marmo",
    "LSW": "scarti di pietra leccese",
    "CW": "scarti ceramici",
    "OW": "scarti di legno di ulivo",
    "CBWS": "scarti di fava di cacao",
}


# --------- funzioni di supporto ---------
def range_report(name, val, r):
    """Restituisce (in_range, messaggio) per un certo intervallo [lo, hi]."""
    lo, hi = r
    if lo == hi:
        in_range = (val == lo)
        if in_range:
            msg = (
                f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] (costante) | "
                "OK (uguale al dato disponibile)"
            )
        else:
            msg = (
                f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] (costante) | "
                f"FUORI (scarto = {abs(val-lo):.5g})"
            )
        return in_range, msg

    if val < lo:
        in_range = False
        dist = lo - val
        msg = (
            f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] | "
            f"FUORI ({dist:.5g} sotto il minimo)"
        )
    elif val > hi:
        in_range = False
        dist = val - hi
        msg = (
            f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] | "
            f"FUORI ({dist:.5g} sopra il massimo)"
        )
    else:
        in_range = True
        width = hi - lo
        pos_pct = (val - lo) / width * 100
        margin_min = val - lo
        margin_max = hi - val
        msg = (
            f"* {name}: {val:.5g} | range = [{lo:.5g}, {hi:.5g}] | "
            f"OK (posizione: {pos_pct:.1f}% del range; margini: +{margin_min:.5g} / {margin_max:.5g})"
        )

    return in_range, msg


def get_local_ip():
    """Ritorna l'indirizzo IP locale (per QR code in rete locale)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "localhost"
    finally:
        s.close()
    return ip


@st.cache_data
def carica_dati(filename):
    """Legge il file tabellare gestendo encoding tipici di Windows."""
    # Prova cp1252 (Windows)
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

    # Fallback: utf-8 ignorando caratteri problematici
    df = pd.read_csv(
        filename,
        sep=None,
        engine="python",
        encoding="utf-8",
        errors="ignore",
    )
    return df


# --------- calcolo dei macro-range globali dai dati ----------
def calcola_macro_range(filename):
    try:
        df = carica_dati(filename)
        E_min, E_max = df["E"].min(), df["E"].max()
        SR_min, SR_max = df["SR"].min(), df["SR"].max()
        eps_min, eps_max = df["epsr"].min(), df["epsr"].max()
        return {
            "E": (E_min, E_max),
            "SR": (SR_min, SR_max),
            "epsr": (eps_min, eps_max),
        }
    except Exception:
        return None


MACRO_RANGES = calcola_macro_range("TabellaAdditivi.txt")


# --------- UI principale ---------
st.title("Previsione P per additivi (Streamlit)")

st.write(
    """
Questa app replica la logica dello script MATLAB:

- legge il file `TabellaAdditivi.txt`
- usa come input: E, SR, epsr
- per ogni additivo allena i modelli LSBoost, Bagging, Lineare, Polinomiale
- restituisce Predizione P, MAE, RMSE, R2.
"""
)

with st.expander("Legenda additivi (sigla -> materiale)"):
    st.markdown(
        "- **MW**: scarti di marmo\n"
        "- **LSW**: scarti di pietra leccese\n"
        "- **CW**: scarti ceramici\n"
        "- **OW**: scarti di legno di ulivo\n"
        "- **CBWS**: scarti di fava di cacao\n"
    )

with st.expander("Legenda variabili di input (E, SR, epsr)"):
    st.markdown(
        "- **E**: Modulo di Young (rigidezza del materiale), in **GPa**\n"
        "- **SR**: Tensione a rottura (resistenza a trazione), in **MPa**\n"
        "- **epsr**: Deformazione a rottura, in **percentuale (%)**\n"
    )

# --------- sidebar: input utente + macro-range + QR ----------
st.sidebar.header("Input utente")

E_input = st.sidebar.number_input(
    "E (Modulo di Young, GPa)",
    value=0.0,
    format="%.6f",
)
SR_input = st.sidebar.number_input(
    "SR (Tensione a rottura, MPa)",
    value=0.0,
    format="%.6f",
)
epsr_input = st.sidebar.number_input(
    "epsr (Deformazione a rottura, %)",
    value=0.0,
    format="%.6f",
)

# Mostra macro-range globale se disponibile
if MACRO_RANGES is not None:
    E_min, E_max = MACRO_RANGES["E"]
    SR_min, SR_max = MACRO_RANGES["SR"]
    eps_min, eps_max = MACRO_RANGES["epsr"]

    st.sidebar.markdown(
        "**Macro-range dei dati (valori consigliati per rientrare nelle previsioni):**\n"
        f"- E: `{E_min:.3g}` – `{E_max:.3g}` GPa\n"
        f"- SR: `{SR_min:.3g}` – `{SR_max:.3g}` MPa\n"
        f"- epsr: `{eps_min:.3g}` – `{eps_max:.3g}` %\n"
    )

st.sidebar.write("---")
esegui = st.sidebar.button("Esegui analisi")

# QR code nella sidebar (per uso in rete locale; per URL pubblico sostituisci url qui sotto)
with st.sidebar.expander("QR code per smartphone"):
    ip = get_local_ip()
    url = f"http://{ip}:8501"
    st.write("PC e telefono devono essere sulla stessa rete WiFi.")
    st.write("URL:", url)

    qr = qrcode.QRCode(
        version=1,
        box_size=8,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_array = np.array(qr_img.convert("RGB"))
    st.image(qr_array, caption="Scansiona con la fotocamera", use_column_width=True)


# --------- logica principale: esegui analisi ----------
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
        st.error("La tabella deve contenere le colonne: tipo, P, E, SR, epsr.")
        st.write("Colonne trovate:", list(data.columns))
        st.stop()

    st.success("Dati caricati correttamente.")
    with st.expander("Anteprima dati"):
        st.dataframe(data.head())

    additivi = sorted(data["tipo"].astype(str).unique())
    risultati_records = []

    st.write("### Analisi per singolo additivo")

    for nome_add in additivi:
        sotto = data[data["tipo"].astype(str) == nome_add].copy()

        X = sotto[["E", "SR", "epsr"]].values
        y = sotto["P"].values
        N = X.shape[0]

        descr = ADDITIVO_DESCR.get(str(nome_add), "")
        header = f"Additivo: {nome_add}"
        if descr:
            header += f" ({descr})"

        st.markdown(f"---\n#### {header}  |  N = {N}")

        # range specifici per questo additivo
        range_E = (np.min(X[:, 0]), np.max(X[:, 0]))
        range_SR = (np.min(X[:, 1]), np.max(X[:, 1]))
        range_epsr = (np.min(X[:, 2]), np.max(X[:, 2]))

        name_E = "E - Modulo di Young (GPa)"
        name_SR = "SR - Tensione a rottura (MPa)"
        name_epsr = "epsr - Deformazione a rottura (%)"


        inE, msgE = range_report(name_E, E_input, range_E)
        inSR, msgSR = range_report(name_SR, SR_input, range_SR)
        inEP, msgEP = range_report(name_epsr, epsr_input, range_epsr)

        st.write("**Valutazione rispetto ai range sperimentali dell'additivo:**")
        st.text(msgE + "\n" + msgSR + "\n" + msgEP)

        in_range_all = inE and inSR and inEP

        if not in_range_all:
            st.warning(
                f"Input FUORI RANGE per l'additivo {nome_add}. Previsioni non affidabili: additivo skippato."
            )
            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Descrizione_additivo": descr,
                    "Modello": "(skipped)",
                    "Predizione_P": np.nan,
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
            st.info(
                f"Input nel range per l'additivo {nome_add}. Si procede con le previsioni dei modelli."
            )

        if N < 3:
            st.warning(
                f"Troppi pochi dati (N={N}) per una K-fold affidabile. Additivo considerato 'too-few'."
            )
            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Descrizione_additivo": descr,
                    "Modello": "(too-few)",
                    "Predizione_P": np.nan,
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

        def cv_predict(model, X_arr, y_arr, cv_obj):
            y_pred_cv = np.full_like(y_arr, np.nan, dtype=float)
            for train_idx, test_idx in cv_obj.split(X_arr):
                X_train, X_test = X_arr[train_idx], X_arr[test_idx]
                y_train = y_arr[train_idx]
                mdl = model
                mdl.fit(X_train, y_train)
                y_pred_cv[test_idx] = mdl.predict(X_test)
            return y_pred_cv

        # ----- LSBoost (GradientBoostingRegressor) -----
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
            R2_ls = 1 - np.sum((y - y_pred_ls_cv) ** 2) / np.sum(
                (y - np.mean(y)) ** 2
            )

            st.write("**Risultati modello LSBoost (GradientBoostingRegressor):**")
            st.write(
                f"Predizione P: {pred_ls:.2f}  |  MAE: {MAE_ls:.2f}  |  "
                f"RMSE: {RMSE_ls:.2f}  |  R2: {R2_ls:.3f}"
            )

            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Descrizione_additivo": descr,
                    "Modello": "LSBoost",
                    "Predizione_P": pred_ls,
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
                    "Descrizione_additivo": descr,
                    "Modello": "LSBoost (err)",
                    "Predizione_P": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )

        # ----- Bagging -----
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

            st.write("**Risultati modello Bagging:**")
            st.write(
                f"Predizione P: {pred_bag:.2f}  |  MAE: {MAE_bag:.2f}  |  "
                f"RMSE: {RMSE_bag:.2f}  |  R2: {R2_bag:.3f}"
            )

            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Descrizione_additivo": descr,
                    "Modello": "Bagging",
                    "Predizione_P": pred_bag,
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
                    "Descrizione_additivo": descr,
                    "Modello": "Bagging (err)",
                    "Predizione_P": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )

        # ----- Lineare -----
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

            st.write("**Risultati modello Lineare:**")
            st.write(
                f"Predizione P: {pred_lin:.2f}  |  MAE: {MAE_lin:.2f}  |  "
                f"RMSE: {RMSE_lin:.2f}  |  R2: {R2_lin:.3f}"
            )

            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Descrizione_additivo": descr,
                    "Modello": "Lineare",
                    "Predizione_P": pred_lin,
                    "MAE": MAE_lin,
                    "RMSE": RMSE_lin,
                    "R2": R2_lin,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )
        except Exception as e:
            st.warning(f"Lineare fallito per {nome_add}: {e}")
            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Descrizione_additivo": descr,
                    "Modello": "Lineare (err)",
                    "Predizione_P": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )

        # ----- Polinomiale 2 ordine -----
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

            st.write("**Risultati modello Polinomiale (2 ordine):**")
            st.write(
                f"Predizione P: {pred_poly:.2f}  |  MAE: {MAE_poly:.2f}  |  "
                f"RMSE: {RMSE_poly:.2f}  |  R2: {R2_poly:.3f}"
            )

            risultati_records.append(
                {
                    "Additivo": nome_add,
                    "Descrizione_additivo": descr,
                    "Modello": "Polinomiale",
                    "Predizione_P": pred_poly,
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
                    "Descrizione_additivo": descr,
                    "Modello": "Polinomiale (err)",
                    "Predizione_P": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "N": N,
                    "K": K,
                    "InRange": 1,
                }
            )

    # --------- riepilogo complessivo + grafico ----------
    if len(risultati_records) > 0:
        risultati = pd.DataFrame(risultati_records)

        col_order = [
            "Additivo",
            "Descrizione_additivo",
            "Modello",
            "N",
            "K",
            "InRange",
            "Predizione_P",
            "MAE",
            "RMSE",
            "R2",
        ]
        risultati = risultati[col_order]

        risultati["is_nan_rmse"] = risultati["RMSE"].isna()
        risultati = risultati.sort_values(
            by=["is_nan_rmse", "RMSE"], ascending=[True, True]
        ).drop(columns=["is_nan_rmse"])

        st.write("## Riepilogo modelli per additivo (ordinati per RMSE crescente)")
        st.dataframe(risultati)

        mods_to_plot = ["LSBoost", "Bagging", "Lineare", "Polinomiale"]
        mask = (
            risultati["Modello"].isin(mods_to_plot)
            & (risultati["InRange"] == 1)
            & (~risultati["RMSE"].isna())
        )
        ris_plot = risultati.loc[
            mask, ["Additivo", "Descrizione_additivo", "Modello", "RMSE"]
        ]

        if not ris_plot.empty:
            st.write(
                "### Confronto modelli per additivo (RMSE piu basso = modello migliore)"
            )

            ris_plot["Additivo_label"] = (
                ris_plot["Additivo"] + " - " + ris_plot["Descrizione_additivo"]
            )

            chart = (
                alt.Chart(ris_plot)
                .mark_bar()
                .encode(
                    x=alt.X("Additivo_label:N", title="Additivo"),
                    y=alt.Y("RMSE:Q", title="RMSE"),
                    color="Modello:N",
                    column=alt.Column("Modello:N", title="Modello"),
                    tooltip=[
                        "Additivo",
                        "Descrizione_additivo",
                        "Modello",
                        "RMSE",
                    ],
                )
                .properties(height=250)
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Nessun dato valido per costruire il grafico di confronto RMSE.")




