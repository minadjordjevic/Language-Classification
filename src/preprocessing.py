import pandas as pd
from sklearn.model_selection import train_test_split


def create_dataframe(records):
    """
    Converts loaded records into a DataFrame and removes duplicate texts.
    """

    df = pd.DataFrame(records)

    df = df[["text", "language"]].copy()

    # Remove missing values
    df = df.dropna(subset=["text", "language"])

    # Remove empty texts
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Remove duplicate texts
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df


def create_train_val_test_split(
    df,
    test_size=0.10,
    validation_size=0.10,
    random_state=42
):
    """
    Creates stratified train, validation and test datasets.
    """

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["language"],
        random_state=random_state
    )

    relative_validation_size = validation_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_validation_size,
        stratify=train_val_df["language"],
        random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
