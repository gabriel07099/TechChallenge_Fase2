import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def create_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """
    Cria a variável binária 'target' a partir de 'quality'.
    target = 1 se quality >= threshold (vinho "bom"), caso contrário 0.
    """
    df = df.copy()
    df["target"] = df["quality"].apply(lambda x: 1 if x >= threshold else 0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features derivadas de conhecimento de domínio sobre
    química do vinho.
    """
    df = df.copy()
    df["acid_ratio"] = df["fixed acidity"] / (df["volatile acidity"] + 1e-6)
    df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    df["free_so2_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
    df["alcohol_density_ratio"] = df["alcohol"] / df["density"]
    df["sugar_alcohol"] = df["residual sugar"] * df["alcohol"]
    return df


def get_outlier_summary(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Retorna contagem e percentual de outliers (método IQR) por coluna."""
    summary = {}
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        summary[col] = {"n_outliers": int(n_out), "pct": round(n_out / len(df) * 100, 2)}
    return pd.DataFrame(summary).T.sort_values("pct", ascending=False)


def split_and_scale(df: pd.DataFrame, drop_cols=("Id", "quality"), test_size=0.2, random_state=42):
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = df_model.drop(columns=["target"])
    y = df_model["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
