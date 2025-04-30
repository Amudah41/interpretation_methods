import os

import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

PATH = os.path.join(os.path.abspath(os.getcwd()), "img")



def locstep(arg, x, y, h, dmax, varimp):
    """
    Функция выполняет локальный шаг алгоритма поиска параметров модели.

    :param arg: Вектор аргументов, который необходимо оценить.
    :param X: Матрица признаков (наблюдения).
    :param y: Вектор целевых значений.
    :param h: Параметр ширины окна (bandwidth).
    :param dmax: Максимальное количество переменных в модели.
    :param varimp: Вектор весовых коэффициентов для каждого признака.
    :return: Оценки параметров модели.
    """

    d = len(arg)  # Размерность пространства аргумента
    n = len(y)  # Количество наблюдений
    est = np.zeros(d + 1)  # Массив для хранения оценок параметров модели

    if d > 1:

        def kernel(xx):
            """
            Функция ядра для многомерного случая.
            :param xx: Входной массив.
            :return: Значения функции ядра.
            """
            return ((2 * np.pi) ** (-d / 2)) * np.exp(-np.sum(xx**2, axis=1) / 2)

        arg_matrix = np.tile(arg, (x.shape[0], 1))  # Создание матрицы из аргументов
        w = kernel(np.sqrt(varimp) * (x - arg_matrix) / h) / h**d  # Вычисление весов
        weights = w / np.sum(w)  # Нормализация весов

        model = LinearRegression().fit(x, y, sample_weight=weights)  # Построение модели
        tmp_var = model.coef_.shape[0]

        est[: model.coef_.shape[-1]] = model.coef_[0]  # Сохранение коэффициентов модели

        est[d:] = model.intercept_  # Добавление свободного члена
    else:

        def kernel(xx):
            """
            Функция ядра для одномерного случая.
            :param xx: Входной массив.
            :return: Значения функции ядра.
            """
            return np.exp(-(xx**2) / 2)

        w = kernel(np.sqrt(varimp) * (x - arg) / h) / h**d  # Вычисление весов
        weights = w / np.sum(w)  # Нормализация весов

        model = LinearRegression().fit(x, y, sample_weight=weights)  # Построение модели
        est[: model.coef_.shape[0]] = model.coef_  # Сохранение коэффициента модели
        est[d:] = model.intercept_  # Добавление свободного члена

    return est


def varimpcal(X, y):
    """
    Функция для расчета локальной и глобальной важности признаков.

    :param X: Матрица признаков (наблюдения).
    :param y: Вектор целевых значений.
    :return: Список с локальной и глобальной важностью признаков.
    """

    rf = RandomForestRegressor(n_estimators=1000, oob_score=True)
    rf.fit(X, y)

    feature_importances = rf.feature_importances_

    global_imp = np.abs(feature_importances) / np.sum(np.abs(feature_importances))
    global_imp = np.tile(global_imp, (X.shape[0], 1))

    local_imp = np.array([tree.feature_importances_ for tree in rf])
    local_imp = np.mean(np.abs(local_imp), axis=0) / np.sum(
        np.abs(local_imp), axis=1
    ).reshape(-1, 1)
    return {"localimp": local_imp, "globalimp": global_imp}


def regcluster(k, x_coef, X, y, dmax=None):
    """
    Функция создает кластеры с коэффициентами и строит регрессионные модели для каждого кластера.

    :param k: Количество кластеров.
    :param x_coef: Коэффициенты признаков без свободного члена.
    :param X: Матрица признаков.
    :param y: Вектор целевых значений.
    :param dmax: Максимальное количество переменных в модели.
    :return: Словарь с коэффициентами моделей, скорректированными значениями R-квадрата и кластерами.
    """
    if dmax is None:
        dmax = X.shape[1]

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k должно быть положительным целым числом.")

    n_samples, n_features = X.shape
    coefs = np.zeros((k, n_features + 1))
    adj_r2 = np.full((k, 1), np.nan)

    if k > 1:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(x_coef)
        clusters = kmeans.labels_

        for i in range(k):
            cluster_indices = np.where(clusters == i)[0]
            cluster_X = X[cluster_indices]
            cluster_y = y.iloc[cluster_indices]

            if n_features == 1:
                lr = LinearRegression().fit(cluster_X, cluster_y)
                coefs[i, :-1] = lr.coef_
                coefs[i, -1] = lr.intercept_
                adj_r2[i] = 1 - (1 - r2_score(cluster_y, lr.predict(cluster_X))) * (
                    n_samples - 1
                ) / (n_samples - n_features - 1)

            else:
                features_to_keep = np.std(cluster_X, axis=0) != 0
                reduced_cluster_X = cluster_X[:, features_to_keep]
                lr = LinearRegression().fit(reduced_cluster_X, cluster_y)
                non_zero_coefs = lr.coef_[0][features_to_keep]
                for j, item in enumerate(non_zero_coefs):
                    if features_to_keep[j]:
                        coefs[i, j] = non_zero_coefs[j]
                coefs[i, -1] = lr.intercept_
                adj_r2[i] = 1 - (
                    1 - r2_score(cluster_y, lr.predict(reduced_cluster_X))
                ) * (n_samples - 1) / (n_samples - np.sum(features_to_keep) - 1)

        w = np.bincount(clusters)
        adj_r2_w = np.average(adj_r2.ravel(), weights=w)

    elif k == 1:
        clusters = np.ones(n_samples, dtype=int)
        if n_features == 1:
            lr = LinearRegression().fit(X, y)
            coefs[0, :-1] = lr.coef_
            coefs[0, -1] = lr.intercept_
            adj_r2_w = 1 - (1 - r2_score(y, lr.predict(X))) * (n_samples - 1) / (
                n_samples - n_features - 1
            )
        else:
            features_to_keep = np.std(X, axis=0) != 0
            reduced_X = X[:, features_to_keep]
            lr = LinearRegression().fit(reduced_X, y)
            non_zero_coefs = lr.coef_[features_to_keep]
            coefs[0, features_to_keep] = non_zero_coefs
            coefs[0, -1] = lr.intercept_
            adj_r2_w = 1 - (1 - r2_score(y, lr.predict(reduced_X))) * (
                n_samples - 1
            ) / (n_samples - np.sum(features_to_keep) - 1)

    return {"coefs": coefs, "adjr2": adj_r2_w, "clusters": clusters}

def varimp(x, y, h, dmax):
    """
    Функция для вычисления локальных оценок важности переменных.

    :param x: Матрица признаков.
    :param y: Вектор целевых значений.
    :param h: Ширина окна.
    :param dmax: Максимальное количество переменных в модели.
    :return: Матрица локальных оценок важности переменных.
    """

    varimportance = varimpcal(x, y.to_numpy())["localimp"]
    cvarimp = np.apply_along_axis(
        lambda arg: locstep(arg, x=x, y=y, h=h, dmax=dmax, varimp=varimportance),
        axis=1,
        arr=x,
    )

    return cvarimp



def supclus(k, x, y, h, dmax):
    """
    Функция для создания суперклассов и вычисления соответствующих коэффициентов.

    :param k: Количество кластеров.
    :param x: Матрица признаков.
    :param y: Вектор целевых значений.
    :param h: Ширина окна.
    :param dmax: Максимальное количество переменных в модели.
    :return: Словарь с коэффициентами и метками кластеров.
    """

    x_coef = varimp(x=x, y=y, h=h, dmax=dmax)
    cluster_coef = regcluster(k=k, x_coef=x_coef, X=x, y=y, dmax=dmax)
    csupclus = np.zeros_like(x)
    csupclus = cluster_coef["coefs"][cluster_coef["clusters"], :-1]

    return {"coef": csupclus, "cluster": cluster_coef["clusters"]}



def plotslopes(sampsize, varnumber, x, y, xcoef, dataset_num):
    """
    Функция для построения графиков наклона.

    :param sampsize: Размер выборки.
    :param varnumber: Номер переменной.
    :param x: Матрица признаков.
    :param y: Вектор целевых значений.
    :param xcoef: Коэффициенты признаков без свободного члена.
    :return: График с точками и сегментами линий.
    """
    tmp_x = x
    x = x.values
    xcoef = xcoef[:, :-1]
    # Случайным образом выбираем sampsize индексов из диапазона от 0 до длины x
    i = np.random.choice(range(len(x)), sampsize, replace=False)

    tmp = y.iloc[i]
    column_name = tmp.columns[0]
    y_np_array = np.array(tmp[column_name])

    slopes = pd.DataFrame(
        {
            "intercept": y_np_array
            - xcoef[i, varnumber] * x[i, varnumber],  # Пересечение оси Y
            "slope": xcoef[i, varnumber],  # Наклон линии
            "x": x[i, varnumber],  # Значения x
            "y": y_np_array,  # Соответствующие значения y
            "xstart": x[i, varnumber]
            - np.std(x[:, varnumber]) / 5,  # Начальные точки x для отрезков
            "xend": x[i, varnumber]
            + np.std(x[:, varnumber]) / 5,  # Конечные точки x для отрезков
            "ystart": y_np_array
            - xcoef[i, varnumber]
            * np.std(x[:, varnumber])
            / 5,  # Начальные точки y для отрезков
            "yend": y_np_array
            + xcoef[i, varnumber]
            * np.std(x[:, varnumber])
            / 5,  # Конечные точки y для отрезков
        }
    )

    # Создаем фигуру и ось для построения графика
    fig, ax = plt.subplots(figsize=(8, 6))

    # Отображаем все точки на графике
    ax.scatter(x[:, varnumber], y, color="royalblue", alpha=0.2)

    # Подписываем оси и задаем заголовок
    ax.set_xlabel(tmp_x.columns[varnumber])
    ax.set_ylabel("f(x)")
    # ax.set_title("График локального наклона")

    # Рисуем каждый отрезок линии отдельно
    for index, row in slopes.iterrows():
        ax.plot(
            [row["xstart"], row["xend"]],
            [row["ystart"], row["yend"]],
            color="green",
            linewidth=2,
            alpha=0.4,
        )
        ax.scatter(row["x"], row["y"], color="red", s=50, alpha=0.5)

    plt.savefig(
        os.path.join(PATH, f"local_slopes_art_dataset_{dataset_num}.png"),
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    # Показываем график
    plt.show()


def ploteffect(instancen, x, y, xcoef, dataset_num, phi):
    """
    Функция для построения графика вклада признаков.

    :param instancen: Индекс экземпляра.
    :param x: Матрица признаков.
    :param y: Вектор целевых значений.
    :param xcoef: Коэффициенты признаков без свободного члена.
    :return: График вклада признаков.
    """
    # Удаляем последний столбец из xcoef, так как он содержит свободный член
    xcoef = xcoef[:, :-1]

    effect = []
    for i in range(len(xcoef)):
        tmp = []
        for k in range(len(xcoef[i])):
            tmp.append(xcoef[i][k] * x.values[i][k])
        effect.append(tmp)

    corr, _ = pearsonr(effect[instancen], phi)

    df = pd.DataFrame({"Feature": x.columns, "Effect": effect[instancen]})

    # Строим график с вкладом признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Feature",
        y="Effect",
        palette=["blue" if e >= 0 else "red" for e in df["Effect"]],
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Признаки")
    plt.ylabel("Вклад")
    plt.title(
        f"VarImp. Вклад признаков для экземпляра {instancen}, коррекляция: {corr}"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(PATH, f"corr_varimp_effect_art_dataset_{dataset_num}.png"),
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    # plt.show()
    return corr

def ploteffectclus(instancen, x, y, clustercoef, cluster, dataset_num, phi):
    """
    Функция для построения графика вклада признаков.

    :param instancen: Индекс экземпляра.
    :param x: Матрица признаков.
    :param y: Вектор целевых значений.
    :param clustercoef: Коэффициенты признаков для кластеров.
    :param cluster: Метки кластеров.
    :return: График вклада признаков.
    """
    # Вычисляем вклад признаков

    effect = clustercoef[cluster, :] * x.values

    # Формируем DataFrame с данными для построения графика
    df = pd.DataFrame({"Feature": x.columns, "Effect": effect[instancen]})

    # Apply the pearsonr()
    corr, _ = pearsonr(effect[instancen], phi)

    # Строим график с вкладом признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Feature",
        y="Effect",
        palette=["blue" if e >= 0 else "red" for e in df["Effect"]],
    )
    plt.axhline(y=0, color="gray")
    plt.xlabel("Признаки")
    plt.ylabel("Вклад")
    plt.title(
        f"SupClus. Вклад признаков для экземпляра {instancen}, коррекляция: {corr}"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(PATH, f"corr_supclass_effect_art_dataset_{dataset_num}.png"),
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    # plt.show()
    return corr


def impl(my_data, dataset_num):
    x_coef = varimp(
        x=my_data["x_test"].values,  # Признаки тестового набора
        y=my_data["y_pred"],  # Предсказанные значения
        h=0.1,  # Ширина окна
        dmax=my_data["x_test"].shape[1],  # Максимальное количество переменных
    )

    # Создаем суперкласс и получаем коэффициенты
    supcluscoef = supclus(
        k=2,  # Количество кластеров
        x=my_data["x_test"].values,  # Признаки тестового набора
        y=my_data["y_pred"],  # Предсказанные значения
        h=0.1,  # Ширина окна
        dmax=my_data["x_test"].shape[1],  # Максимальное количество переменных
    )


    # Строим графики
    plotslopes(
        sampsize=50,  # Размер выборки
        varnumber=0,  # Номер переменной
        x=my_data["x_test"],  # Признаки тестового набора
        y=my_data["y_pred"],  # Предсказанные значения
        xcoef=x_coef,  # Важность признаков
        dataset_num=dataset_num,
    )
    varimp_corr = ploteffect(
        instancen=0,  # Индекс экземпляра
        x=my_data["x_test"],  # Признаки тестового набора
        y=my_data["y_pred"],  # Предсказанные значения
        xcoef=x_coef,  # Важность признаков
        dataset_num=dataset_num,
        phi=my_data["phi"],
    )

    supclus_corr = ploteffectclus(
        instancen=0,  # Индекс экземпляра
        x=my_data["x_test"],  # Признаки тестового набора
        y=my_data["y_pred"],  # Реальные значения
        clustercoef=supcluscoef["coef"],  # Коэффициенты суперкласса
        cluster=supcluscoef["cluster"],  # Метки кластеров
        dataset_num=dataset_num,
        phi=my_data["phi"],
    )
    return {"varimp": varimp_corr, "supclus": supclus_corr}



def lime(X, y, phi, dataset_num):

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Обучение модели случайного леса
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Использование LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X.columns,
        discretize_continuous=True,
        mode="regression",
    )
    i = 0  # пример для интерпретации
    exp = explainer.explain_instance(
        X_test.iloc[i].values, rf_model.predict, num_features=20
    )

    effect = [0] * 20
    for item in exp.as_map()[0]:
        effect[item[0]] = item[-1]
    corr, _ = pearsonr(effect, phi)

    df = pd.DataFrame({"Feature": X.columns, "Effect": effect})

    # Строим график с вкладом признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Feature",
        y="Effect",
        palette=["blue" if e >= 0 else "red" for e in df["Effect"]],
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Признаки")
    plt.ylabel("Вклад")
    plt.title(f"Lime. Вклад признаков для экземпляра {0}, коррекляция: {corr}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PATH, f"corr_lime_effect_art_dataset_{dataset_num}.png"),
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    return {"lime_corr": corr}



def shap(X, y, phi, dataset_num):
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Обучение модели случайного леса
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Обучене дерева решений
    tree_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

    # Использование SHAP для дерева решений
    tree_explainer = shap.Explainer(tree_model, X_train)
    tree_shap_values = tree_explainer(X_test)

    effect = tree_shap_values[0].values

    corr, _ = pearsonr(effect, phi)

    df = pd.DataFrame({"Feature": X.columns, "Effect": effect})

    # Строим график с вкладом признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Feature",
        y="Effect",
        palette=["blue" if e >= 0 else "red" for e in df["Effect"]],
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Признаки")
    plt.ylabel("Вклад")
    plt.title(f"Shap. Вклад признаков для экземпляра {0}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PATH, f"corr_shap_effect_art_dataset_{dataset_num}.png"),
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    return {"shap_corr": corr}
