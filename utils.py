import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
from collections import Counter
from gym import spaces
import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from datetime import datetime, timedelta
import keras
from scipy.interpolate import interp1d


class PortfolioEnv(gym.Env):
    def __init__(self, data, real_data, portfolio, initial_cash=1_000_000, transaction_cost_pct=0.0004):
        super(PortfolioEnv, self).__init__()
        self.data = data
        self.real_data = real_data
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.portfolio = portfolio
        self.reset()

        self.action_space = spaces.Box(low=-1, high=1, shape=(data.shape[1],), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(data.shape[1] * 2 + 1,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.actions_memory = [[0] * 40]
        self.portfolio_value = self.initial_cash
        self.real_portfolio_value = self.initial_cash
        self.cash = self.initial_cash
        self.real_cash = self.initial_cash
        # self.portfolio = np.zeros(self.data.shape[1])
        self.real_portfolio = np.zeros(self.real_data.shape[1])
        self.asset_memory = [self.initial_cash]
        self.real_asset_memory = [self.initial_cash]
        self.portfolio_return_memory = [0]
        self.real_portfolio_return_memory = [0]
        self.weights = np.zeros(self.data.shape[1])
        self.real_weights = np.zeros(self.real_data.shape[1])
        self.date_memory = [self.data.index[self.current_step]]
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        state = np.concatenate(([self.cash], self.portfolio, self.data.iloc[self.current_step].values))
        return state

    def _get_real_state(self):
        state = np.concatenate(([self.real_cash], self.real_portfolio, self.real_data.iloc[self.current_step].values))
        return state

    def step(self, action):
        done = self.current_step >= len(self.data) - 1
        if done:
            # df = pd.DataFrame(self.portfolio_return_memory)
            # df.columns = ['daily_return']
            # plt.plot(df.daily_return.cumsum(), 'r')
            # plt.savefig('results/cumulative_reward.png')
            # plt.close()
            #
            # plt.plot(self.portfolio_return_memory, 'r')
            # plt.savefig('results/rewards.png')
            # plt.close()
            # dates = [datetime.strptime(date_str, "%Y-%m-%d").strftime("%d-%m") for date_str in self.date_memory]
            # plt.figure(figsize=(16, 9))
            # plt.plot(dates, self.real_asset_memory, label='Стоимость настоящего портфеля')
            # plt.plot(dates, self.asset_memory, label='Стоимость виртуального портфеля', linestyle='--')
            # plt.xticks([dates[i] for i in range(0, len(dates), 4)], rotation=45)
            # plt.xlabel('Дата')
            # plt.ylabel('Стоимость портфеля(млн.руб.)')
            # plt.title('Изменения стоимости портфеля')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig('results/true_portfolio_return.png')
            # plt.close()
            #
            # plt.figure(figsize=(24, 16))
            # plt.bar(np.arange(len(self.weights)), self.weights, width=0.4, label="Предсказанные цены")
            # plt.bar(np.arange(len(self.real_weights)) + 0.4, self.real_weights, width=0.4, label="Реальные цены")
            # plt.xlabel("Акции")
            # plt.xticks(np.arange(len(self.weights)) + 0.2, self.data.columns, rotation=45)
            # plt.ylabel("Вес в портфеле")
            # plt.title("Гистограмма весов акций в портфеле")
            # plt.grid(True)
            # plt.savefig('results/portfolio_weights.png')
            # plt.close()
            # print("=================================")
            # print("begin_total_asset:{}".format(self.asset_memory[0]))
            # print("end_total_asset:{}".format(self.portfolio_value))
            # print("end_real_total_asset:{}".format(self.real_portfolio_value))
            # df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            # df_daily_return.columns = ['daily_return']
            # if df_daily_return['daily_return'].std() != 0:
            #     sharpe = (252 ** 0.5) * df_daily_return['daily_return'].mean() / \
            #              df_daily_return['daily_return'].std()
            #     print("Sharpe: ", sharpe)
            # print("=================================")
            state = self._get_state()
            return self.state, self.portfolio_return_memory[len(self.portfolio_return_memory) - 1], done, {
                'real_state': state, 'portfolio': self.portfolio}
        else:
            discrete_actions = self.discretize_actions(action)
            counts = Counter(discrete_actions)
            num = counts.get(1, 0) + counts.get(0, 0)
            if num != 0:
                share_per_one = 1 / num
            else:
                share_per_one = 0
            self.actions_memory.append(discrete_actions)
            # share_per_one = 1 / len([value for value in action if value >= 0])
            # Get current prices
            current_prices = self.data.iloc[self.current_step].values
            real_current_prices = self.real_data.iloc[self.current_step].values
            action_dict = {index: value for index, value in enumerate(discrete_actions)}
            # Сортируем словарь по значению (по убыванию)
            sorted_action_dict = dict(sorted(action_dict.items(), key=lambda item: item[1], reverse=True))
            cash_true = True
            # Обработка дискретных действий
            for i, action in sorted_action_dict.items():
                if action == 1:
                    if cash_true:
                        keys_with_zero_value = [key for key, value in action_dict.items() if
                                                value == 0 and self.portfolio[key] > 0]
                        if len(keys_with_zero_value) > 0:
                            cash_before = np.sum(
                                [self.portfolio[key] * current_prices[key] for key in keys_with_zero_value]) / (
                                                  share_per_one * len(keys_with_zero_value))
                            cash_before_real = np.sum(
                                [self.real_portfolio[key] * real_current_prices[key] for key in
                                 keys_with_zero_value]) / (
                                                       share_per_one * len(keys_with_zero_value))
                        else:
                            cash_before = self.cash
                            cash_before_real = self.real_cash
                        cash_true = False
                    if self.cash > 0:
                        # Покупаем
                        # Рассчитываем количество акций для покупки
                        buy_quantity = round(
                            (cash_before * share_per_one - self.portfolio[i] * current_prices[i]) / current_prices[i])
                        if buy_quantity < 0:
                            buy_quantity = 0
                        # Покупаем столько акций, чтобы доля была примерно одинаковая после покупки
                        buy_amount = buy_quantity * current_prices[i]
                        if buy_amount > (self.cash - self.cash * self.transaction_cost_pct):
                            buy_amount = self.cash - self.cash * self.transaction_cost_pct
                        self.cash -= buy_amount + buy_amount * self.transaction_cost_pct
                        self.portfolio[i] += buy_quantity

                        # Реальные данные для графика
                        buy_quantity_real = round(
                            (cash_before_real * share_per_one - self.real_portfolio[i] * real_current_prices[i]) /
                            real_current_prices[i])
                        if buy_quantity_real < 0:
                            buy_quantity_real = 0
                        # Покупаем столько акций, чтобы доля была примерно одинаковая после покупки
                        buy_amount = buy_quantity_real * real_current_prices[i]
                        if buy_amount > (self.real_cash - self.real_cash * self.transaction_cost_pct):
                            buy_amount = self.real_cash - self.real_cash * self.transaction_cost_pct
                        self.real_cash -= buy_amount + buy_amount * self.transaction_cost_pct
                        self.real_portfolio[i] += buy_quantity_real
                elif action == 2:  # Продаем
                    # Продаем все акции из портфеля
                    sell_quantity = self.portfolio[i]
                    sell_amount = sell_quantity * current_prices[i]
                    self.cash += sell_amount - sell_amount * self.transaction_cost_pct
                    self.portfolio[i] = 0
                    # Реальные данные для графика
                    sell_quantity_real = self.real_portfolio[i]
                    sell_amount_real = sell_quantity_real * real_current_prices[i]
                    self.real_cash += sell_amount_real - sell_amount_real * self.transaction_cost_pct
                    self.real_portfolio[i] = 0

            # Calculate portfolio value
            self.current_step += 1
            self.date_memory.append(self.data.index[self.current_step])
            current_prices = self.data.iloc[self.current_step].values
            real_current_prices = self.real_data.iloc[self.current_step].values
            portfolio_value = np.sum(self.portfolio * current_prices) + self.cash
            real_portfolio_value = np.sum(self.real_portfolio * real_current_prices) + self.real_cash
            self.portfolio_value = portfolio_value
            self.real_portfolio_value = real_portfolio_value
            self.asset_memory.append(portfolio_value)
            self.real_asset_memory.append(real_portfolio_value)

            # Calculate returns
            portfolio_return = (portfolio_value - self.asset_memory[-2]) / self.asset_memory[-2]
            real_portfolio_return = (real_portfolio_value - self.real_asset_memory[-2]) / self.real_asset_memory[-2]

            self.portfolio_return_memory.append(portfolio_return)
            self.real_portfolio_return_memory.append(real_portfolio_return)

            self.weights = self.portfolio * current_prices / portfolio_value
            self.real_weights = self.real_portfolio * real_current_prices / real_portfolio_value

            if done:
                reward = portfolio_value - self.portfolio_value
            else:
                reward = portfolio_return

            self.state = self._get_state()
            real_state = self._get_real_state()

            return self.state, reward, done, {'real_state': self.state, 'portfolio': self.portfolio,
                                              'actions': discrete_actions}

    def discretize_actions(self, new_weights):
        discrete_actions = []
        for i in range(len(new_weights)):
            if new_weights[i] >= 0.33:
                discrete_actions.append(1)  # Покупаем
            elif new_weights[i] <= -0.33:
                discrete_actions.append(2)  # Продаем
            else:
                discrete_actions.append(0)  # Держим
        return discrete_actions

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Portfolio Value: {self.asset_memory[-1]}')
        print(f'Cash: {self.cash}')
        print(f'Portfolio: {self.portfolio}')
        print(f'Real Portfolio Value: {self.real_asset_memory[-1]}')
        print(f'Real Cash: {self.real_cash}')
        print(f'Real Portfolio: {self.real_portfolio}')


# Функция для создания последовательностей для LSTM
def create_sequences(data, input_size, output_size):
    X = []
    for i in range(0, len(data) - input_size - output_size + 1, output_size):
        X.append(data[i:i + input_size])
    return np.array(X)


def interpolate_data(data):
    if len(data.shape) == 1:
        data = data.reshape(data.shape[0], 1)
    # Определение индексов пропущенных значений
    missing_indices = np.isnan(data[:, 0])

    # Создание массива индексов для известных значений
    known_indices = np.arange(len(data))[~missing_indices]

    # Создание функции интерполяции
    f = interp1d(known_indices, data[~missing_indices, 0], kind='linear', fill_value='extrapolate')

    # Интерполяция пропущенных значений
    interpolated_values = f(np.arange(len(data)))

    return interpolated_values.reshape(-1, 1)


def prepare_data_with_preprocessing(test_data, transformers):
    test_data = interpolate_data(transformers['hampel'].transform(test_data))

    test_data = interpolate_data(transformers['deseasonalizer'].transform(test_data))

    test_data = interpolate_data(transformers['transformation'].transform(test_data))

    test_data = transformers['scaler'].transform(test_data.reshape(-1, 1)).reshape(test_data.shape)
    #test_data = create_sequences(test_data, 10, 5)
    test_data = test_data.reshape(1, 10, 1)

    return test_data


def get_dataframe(data):
    all_data = {}
    # data = pd.read_csv("Prices MCXSM test.csv", index_col='index')
    # data = data.drop(['ELFV'], axis=1)
    for column in data.columns:
        ticker_data = data[column].dropna()
        #test_data, test_index = ticker_data.values, ticker_data.index
        test_data, test_index = ticker_data[-10:].values, ticker_data[-10:].index
        all_data[column] = test_data, test_index
    arr_index = []
    # input_size = 10
    # output_size = 5
    # for i in range(0, len(test_index) - input_size - output_size + 1, output_size):
    #     arr_index.append(test_index[i + input_size:i + input_size + output_size])
    # arr_index = np.array(arr_index).reshape(-1)
    model_folder = "model_2"
    models = {}
    transformers_data = {}
    for stock_symbol in os.listdir(model_folder):
        model_path = os.path.join(model_folder, stock_symbol, f"{stock_symbol}.keras")
        path = os.path.join(model_folder, stock_symbol)
        models[stock_symbol] = keras.saving.load_model(model_path)
        transformers_data[stock_symbol] = {'hampel': joblib.load(path + rf"\hampel_{stock_symbol}"),
                                           'deseasonalizer': joblib.load(path + rf"\deseasonalizer_{stock_symbol}"),
                                           'transformation': joblib.load(path + rf"\transformation_{stock_symbol}"),
                                           'scaler': joblib.load(path + rf"\scaler_{stock_symbol}")}

    # Получение предсказаний от каждой модели
    predictions = {}  # Словарь для хранения предсказаний
    true_values = {}
    for stock_symbol, model in models.items():
        if stock_symbol in ['ELFV', 'GEMC']:
            continue
        transformers = transformers_data[stock_symbol]
        test_data, test_index = all_data[stock_symbol]
        X = prepare_data_with_preprocessing(test_data, transformers)
        y_pred = model.predict(X)

        # y_test = transformers['scaler'].inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        y_pred = transformers['scaler'].inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
        # y_test = transformers['transformation'].inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        y_pred = transformers['transformation'].inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
        # y_test = transformers['deseasonalizer'].inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        y_pred = transformers['deseasonalizer'].inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

        # mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        # smape = 2 * np.mean(np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test))) * 100
        # print(stock_symbol, smape, mape)
        predictions[stock_symbol] = y_pred.reshape(-1)
        true_values[stock_symbol] = test_data
    # Объединение предсказаний в один датасет
    df = pd.DataFrame(predictions)
    # df = df.set_index(arr_index)
    df_test = pd.DataFrame(true_values)
    df_test = df_test.set_index(test_index)
    return df, df_test


def evaluate_agent(df, portfolio, cash=1000000):
    env = DummyVecEnv([lambda: PortfolioEnv(df, df, portfolio, cash)])
    agent = A2C.load('trained_a2c-2')
    obs = env.reset()
    done = False
    portfolio_values = []
    results = []
    while not done:
        result = {}
        action, _states = agent.predict(obs)
        obs, reward, done, state = env.step(action)
        if done:
            break
        real_state = state[0]['real_state']
        cash, portfolio, data = real_state[:1], real_state[1:41], real_state[41:]
        portfolio_value = np.sum(portfolio * data) + cash
        portfolio_values.append(portfolio_value)
        try:
            actions_dict = dict(zip(df.columns, state[0]['actions']))
        except:
            actions_dict = dict(zip(df.columns, [0] * 40))
        result['cash'], result['portfolio'], result['actions'] = cash, portfolio, actions_dict
        results.append(result)
    return results, portfolio_values
