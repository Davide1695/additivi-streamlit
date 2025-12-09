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
        "- **E**: modulo di Young (rigidezza del materiale), in **GPa**\n"
        "- **SR**: sigma a rottura (resistenza a trazione), in **MPa**\n"
        "- **epsr**: deformazione a rottura, in **percentuale (%)**\n"
    )

# --------- sidebar: input utente + macro-range + QR ----------
st.sidebar.header("Input utente")

E_input = st.sidebar.number_input(
    "E (modulo di Young, GPa)",
    value=0.0,
    format="%.6f",
)
SR_input = st.sidebar.number_input(
    "SR (sigma a rottura, MPa)",
    value=0.0,
    format="%.6f",
)
epsr_input = st.sidebar.number_input(
    "epsr (deformazione a rottura, %)",
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

        descr = ADDITI
