import re


FAULT_STATE_ORDER = ["healthy", "inner", "outer", "rolling"]

FAULT_STATE_DISPLAY_NAMES = {
    "rolling": "BF",
    "healthy": "NC",
    "inner": "IF",
    "outer": "OF",
}


def state_key_from_column(column_name):

    raw_name = str(column_name).strip()
    first_token = re.split(r"[-_.]", raw_name.lower(), maxsplit=1)[0]
    parts = set(re.split(r"[-_.]", raw_name.lower()))

    if first_token in {"h", "n", "normal"} or "jk" in parts:
        return "healthy"
    if first_token in {"if", "ir"} or "nq" in parts:
        return "inner"
    if first_token in {"of", "or"} or "wq" in parts:
        return "outer"
    if first_token in {"rf", "bf"} or "gz" in parts:
        return "rolling"
    return None


def _detect_state_columns(columns, source_name):
    state_to_column = {}
    unknown_columns = []
    for column in columns:
        state_key = state_key_from_column(column)
        if state_key is None:
            unknown_columns.append(str(column))
            continue
        if state_key in state_to_column:
            raise ValueError(
                f"{source_name}: duplicate columns for {FAULT_STATE_DISPLAY_NAMES[state_key]}: "
                f"{state_to_column[state_key]} and {column}"
            )
        state_to_column[state_key] = column
    return state_to_column, unknown_columns


def _is_complete_state_table(columns, state_to_column, unknown_columns):
    return not unknown_columns and len(state_to_column) == len(list(columns))


def align_modal_dataframe_columns(
    df_a, df_b, source_a="Vibration table", source_b="AE table"
):

    state_cols_a, unknown_a = _detect_state_columns(df_a.columns, source_a)
    state_cols_b, unknown_b = _detect_state_columns(df_b.columns, source_b)

    parsed_a = _is_complete_state_table(df_a.columns, state_cols_a, unknown_a)
    parsed_b = _is_complete_state_table(df_b.columns, state_cols_b, unknown_b)

    if parsed_a and parsed_b:
        states_a = set(state_cols_a)
        states_b = set(state_cols_b)
        if states_a != states_b:
            raise ValueError(
                "Vibration and AE tables contain different states. "
                f"{source_a}={sorted(states_a)}, {source_b}={sorted(states_b)}"
            )

        ordered_states = [state for state in FAULT_STATE_ORDER if state in states_a]
        df_a_aligned = df_a[[state_cols_a[state] for state in ordered_states]].copy()
        df_b_aligned = df_b[[state_cols_b[state] for state in ordered_states]].copy()
        class_names = [FAULT_STATE_DISPLAY_NAMES[state] for state in ordered_states]
        return df_a_aligned, df_b_aligned, class_names

    if len(df_a.columns) != len(df_b.columns):
        raise ValueError(
            "Modal tables contain different numbers of columns and cannot be aligned by headers. "
            f"{source_a}={len(df_a.columns)}, {source_b}={len(df_b.columns)}"
        )

    return df_a.copy(), df_b.copy(), [str(column) for column in df_a.columns]
