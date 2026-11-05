"""The shared metrics entry point. Every number in results/ and report/ comes from here."""
from sklearn.metrics import f1_score, mean_absolute_error, root_mean_squared_error


def material_scores(y_true, y_pred):
    return {"macro_f1": f1_score(y_true, y_pred, average="macro")}


def rating_scores(y_true, y_pred):
    return {"rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred)}
