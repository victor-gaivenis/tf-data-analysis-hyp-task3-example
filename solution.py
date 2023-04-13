import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 987333364 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool: # Одна или две выборке на входе, заполняется исходя из условия
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    alpha = 0.03
    return MMD(compute_kernel="rbf", gamma=1).test(x, y)[1] < alpha # Ваш ответ, True или False
