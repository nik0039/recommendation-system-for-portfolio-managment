import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import xml.etree.ElementTree as ET
import requests
from datetime import date, timedelta
import apimoex
import utils
import random
from collections import defaultdict, Counter

# Темная тема
st.set_page_config(layout="wide", page_title="Portfolio Dashboard", page_icon="📈")
print("Началась работа!", st.session_state)
# Выбор дат
today = date.today()
start_of_week = today - timedelta(days=today.weekday())
end_of_week = start_of_week + timedelta(days=6)
start_date = st.sidebar.date_input("Начало периода", start_of_week)
end_date = st.sidebar.date_input("Конец периода", end_of_week)
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")
start_date_str = '2024-04-01'
end_date_str = '2024-05-01'


# Функция для показа полного графика
def show_full_chart(stock):
    st.session_state['show_full_chart'] = True
    st.session_state['full_chart_stock'] = stock


def equal_weights(df, investment=1000000):
    total_investment = investment  # 1 миллион долларов
    stock_prices = df.iloc[-1]
    # Количество акций в портфеле
    num_stocks = 40

    # Рассчитываем сумму, которую нужно инвестировать в каждую акцию
    equal_investment = total_investment / num_stocks

    # Рассчитываем количество акций для каждой компании
    stock_quantities = {stock: equal_investment // price for stock, price in stock_prices.items()}

    # Преобразуем в DataFrame для удобства работы
    df_stock_quantities = pd.DataFrame(list(stock_quantities.items()), columns=['Stock', 'Quantity'])
    return df_stock_quantities['Quantity'].values


def define_recomendation_category(data):
    stock_counts = defaultdict(list)

    for entry in data:
        for key, value in entry.items():
            if key == 'actions':
                for ticker, ticker_values in value.items():
                    stock_counts[ticker].append(ticker_values)

    # Шаг 2: Подсчитать, какое значение встречается чаще всего для каждого тикера
    most_common_values = {}

    for ticker, counts in stock_counts.items():
        count_frequency = Counter(counts)
        most_common_value, frequency = count_frequency.most_common(1)[0]
        most_common_values[ticker] = most_common_value
    return most_common_values


# Функция для загрузки данных
@st.cache_data(ttl=3600)
def load_data(start_date, end_date):
    with requests.Session() as session:
        data = apimoex.get_index_tickers(session, 'MCXSM', '2023-12-29')
        df_tickers = pd.DataFrame(data)
        df_final = pd.DataFrame()
        for ticker in df_tickers['ticker']:
            if ticker in ['ELFV', 'GEMC']:
                continue
            data = apimoex.get_board_history(session, ticker, start=start_date, end=end_date)
            df = pd.DataFrame(data)
            df.set_index('TRADEDATE', inplace=True)
            df = df.rename(columns={"CLOSE": ticker})
            df_final = pd.concat([df_final, df[ticker]], axis=1)

        df_final.index = pd.to_datetime(df_final.index)
        security_info = []
        security_data = apimoex.get_board_securities(session)
        for ticker in df_tickers['ticker']:
            if security_data:
                result = next((item for item in security_data if item['SECID'] == ticker), None)
                security_info.append({
                    'Ticker': ticker,
                    'Name': result['SHORTNAME'],
                    # 'Sector': security_data[0]['SECTOR']
                })

        df_info = pd.DataFrame(security_info)
        url = f'https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/MESMTR.xml?from={start_date_str}&till={end_date_str}'
        response = requests.get(url)

        # Проверка успешности запроса
        if response.status_code == 200:
            # Парсинг XML
            root = ET.fromstring(response.content)

            # Получение всех элементов 'row'
            rows = root.findall('.//row')

            # Извлечение значений из поля 'CLOSE'
            index_arr = [float(row.attrib['CLOSE']) for row in rows]
        else:
            print(f"Ошибка при выполнении запроса: {response.status_code}")
            index_arr = []
        return df_final, df_info, index_arr


# Кнопка для загрузки данных
if st.sidebar.button('Загрузить данные') or len(st.session_state) == 0:
    df_final, df_info, MCXSM_index = load_data(start_date_str, end_date_str)
    tickers = df_final.columns
    st.session_state['tickers'] = tickers
    predictions, prices = utils.get_dataframe(df_final)
    stocks_data = {}
    for ticker in tickers:
        stocks_data[ticker] = {'prices': prices[ticker].values, 'predicted': predictions[ticker].values,
                               'name': df_info[df_info.Ticker == ticker].Name.values[0],
                               'date_index': prices[ticker].index}
    st.session_state['stocks_data'] = stocks_data
    st.session_state['df_predicted'] = predictions
    st.session_state['df_final'] = df_final
    st.session_state['MCXSM_index'] = MCXSM_index

print(st.session_state)

if 'show_full_chart' not in st.session_state:
    st.session_state['show_full_chart'] = False
    st.session_state['full_chart_stock'] = None

if 'stocks_data' in st.session_state:
    tickers = st.session_state['tickers']
    stocks_data = st.session_state['stocks_data']
    print("Данные сохранились!")
# Пример данных
# data = {
#     'Date': pd.date_range(start='2023-01-01', periods=12, freq='ME'),
#     'Stock Price': list(range(100, len(MCXSM_index), 50)),
#     'Индекс': MCXSM_index
# }
data = {
    'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'Stock Price': [100, 200, 300, 250, 400, 500, 450, 550, 600, 700, 750, 800],
    'Index': [80, 150, 230, 180, 300, 450, 360, 440, 470, 520, 560, 610]
}
# Данные для круговых диаграмм
# portfolio_data = {
#     'Stock': ['TWTR', 'AAPL', 'XIACF', 'TSLA', 'GOOGL', 'AMZN'],
#     'Investment': [36600, 145180, 26600, 536600, 30000, 80000]
# }

sector_data = {
    'Stock': ['OGKB', 'MVID', 'MTLR', 'LSRG', 'GLTR', 'RNFT', 'SELG', 'CBOM', 'APTK', 'AFKS'],
    'Investment': [36600, 145180, 26600, 536600, 30000, 80000, 25000, 4100, 53000, 33050]
}

# Создание DataFrame
df = pd.DataFrame(data)
df['Portfolio Return'] = df['Stock Price'].pct_change().fillna(0)
df['Index Return'] = df['Index'].pct_change().fillna(0)
# Накопленная доходность
df['Cumulative Portfolio Return'] = (1 + df['Portfolio Return']).cumprod() - 1
df['Cumulative Index Return'] = (1 + df['Index Return']).cumprod() - 1
# Преобразование доходности в проценты
df['Cumulative Portfolio Return Percent'] = df['Cumulative Portfolio Return'] * 100
df['Cumulative Index Return Percent'] = df['Cumulative Index Return'] * 100
# portfolio_df = pd.DataFrame(portfolio_data)
sector_df = pd.DataFrame(sector_data)

# Основной график
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative Portfolio Return Percent'], mode='lines+markers', name='Портфель',
                         line=dict(color='orange'), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative Index Return Percent'], mode='lines+markers', name='Индекс MCXSM',
                         line=dict(dash='dash', color='white')))

fig.update_layout(
    title='График накопленной доходности портфеля и индекса',
    xaxis=dict(title='Дата'),
    yaxis=dict(title='Накопленная доходность (%)'),
    yaxis_tickformat='%{n}',  # Форматирование оси Y в процентах
    legend=dict(x=0.01, y=0.99)
)

# Streamlit layout
st.title("Дашборд портфеля")

# Кнопка "Получить рекомендацию"
if st.sidebar.button("Получить рекомендацию"):
    # random.shuffle(stock_cards)
    portfolio_test = equal_weights(st.session_state['df_final'])
    #portfolio_test = equal_weights(df_final)
    agent_results, predictions_value = utils.evaluate_agent(st.session_state['df_predicted'], portfolio_test, 0)
    #agent_results, predictions_value = utils.evaluate_agent(predictions, portfolio_test, 0)
    category_dict = define_recomendation_category(agent_results)
    recommended_category = {
        "Покупать": [key for key, value in category_dict.items() if value == 1],
        "Продавать": [key for key, value in category_dict.items() if value == 2],
        "Держать": [key for key, value in category_dict.items() if value == 0]
    }
    recommended_stocks = {}
    # Выводим карточки в каждой категории
    for category, stocks in recommended_category.items():
        st.subheader(category)
        for stock in stocks:
            price_data = stocks_data[stock]['prices']
            predicted_data = stocks_data[stock]['predicted']
            name = stocks_data[stock]['name']
            recommended_stocks[stock] = category
            # Полный график
            full_fig = go.Figure()
            full_fig.add_trace(go.Scatter(
                x=list(range(len(price_data))),
                y=price_data,
                mode='lines+markers',
                line=dict(color='green' if price_data[-1] >= price_data[0] else 'red'),
                name='Текущий'
            ))
            full_fig.add_trace(go.Scatter(
                x=list(range(len(price_data), len(price_data) + len(predicted_data))),
                y=predicted_data,
                mode='lines+markers',
                line=dict(color='blue'),
                name='Прогноз'
            ))
            full_fig.update_layout(
                title=f'{name} прогнозируемый график',
                xaxis=dict(title='Month'),
                yaxis=dict(title='Price'),
                template='plotly_dark'
            )
            st.plotly_chart(full_fig, use_container_width=True)
    st.session_state['recommended_stocks'] = recommended_stocks
# Вкладка "Мой портфель"
with st.sidebar.expander("Мой портфель", expanded=False):
    portfolio_input = {}
    for stock in tickers:
        portfolio_input[stock] = st.number_input(f'{stock} акций', min_value=0, value=0)
        st.write(f'Текущая цена: ₽ {stocks_data[stock]["prices"][-1]}')
    st.session_state['portfolio'] = portfolio_input

# Обновление данных круговой диаграммы на основе ввода пользователя
new_portfolio_df = pd.DataFrame(list(portfolio_input.items()), columns=['Stock', 'Investment'])

# Круговая диаграмма
pie_fig = px.pie(new_portfolio_df, values='Investment', names='Stock', title='Распределение активов', hole=0.3,
                 template='plotly_dark')
pie_fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1, 0, 0, 0])

# Круговая диаграмма по секторам
sector_pie_fig = px.pie(sector_df, values='Investment', names='Stock', title='Портфель алгоритма', hole=0.3,
                        template='plotly_dark')
sector_pie_fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1, 0, 0, 0])

# Размещение элементов
col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Баланс")
    st.markdown("""
        <div style="background-color: #333; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="color: white; font-size: 30px;">₽ 8342,098</h1>
            <p style="color: white; font-size: 18px;">Инвестировано</p>
            <p style="color: white; font-size: 22px;">₽ 5360,600 </p>
            <p style="color: white; font-size: 18px;">Доходность</p>
            <p style="color: green; font-size: 22px;">₽ 2860,603 (+53%)</p>
            <p style="color: white; font-size: 18px;">Кэш</p>
            <p style="color: white; font-size: 22px;">₽ 120,895 </p>
        </div>
    """, unsafe_allow_html=True)

col3, col4 = st.columns([1, 1])

with col3:
    st.plotly_chart(pie_fig, use_container_width=True)

with col4:
    st.plotly_chart(sector_pie_fig, use_container_width=True)

# Карточки акций с мини-графиками
st.subheader("Акции")
stock_cards = list(stocks_data.keys())
num_cards = len(stock_cards)
cols = st.columns(3)

for i in range(num_cards):
    with cols[i % 3]:
        stock = stock_cards[i]
        price_data = stocks_data[stock]['prices']
        predicted_data = stocks_data[stock]['predicted']
        name = stocks_data[stock]['name']

        # Процентное изменение цены
        change_percent = ((price_data[-1] - price_data[-2]) / price_data[-2]) * 100
        color = 'green' if change_percent >= 0 else 'red'

        # Создание мини-графика
        mini_fig = go.Figure()
        mini_fig.add_trace(go.Scatter(
            x=list(range(len(price_data))),
            y=price_data,
            mode='lines',
            line=dict(color=color),
            fill='tozeroy',
            fillcolor=f'rgba(0, 255, 0, 0.2)' if color == 'green' else f'rgba(255, 0, 0, 0.2)'
        ))
        mini_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=150,
            xaxis=dict(visible=False),
            yaxis=dict(range=[min(price_data), max(price_data)]),
            showlegend=False
        )

        # Отображение карточки
        if 'recommended_stocks' in st.session_state:
            recommended_stocks = st.session_state['recommended_stocks']
            st.write(f"**{stock} Рекомендация: {recommended_stocks[stock]}**")
        else:
            st.write(f"**{stock}**")
        st.plotly_chart(mini_fig, use_container_width=True)
        st.write(f"**{name}  {change_percent:.1f}%**")
        # st.write(f"**{change_percent:.1f}%**")
        st.button(f'Показать полный график {stock}', key=f'button_{stock}', on_click=show_full_chart, args=(stock,))

# Отображение полного графика при необходимости
if st.session_state['show_full_chart']:
    stock = st.session_state['full_chart_stock']
    price_data = stocks_data[stock]['prices']
    predicted_data = stocks_data[stock]['predicted']
    print(predicted_data)
    name = stocks_data[stock]['name']
    date_index = stocks_data[stock]['date_index']
    full_fig = go.Figure()
    full_fig.add_trace(go.Scatter(
        x=list(range(1, len(price_data) + 1)),
        y=price_data,
        mode='lines+markers',
        line=dict(color='green' if price_data[-1] >= price_data[0] else 'red'),
        name='Актуальный'
    ))
    full_fig.add_trace(go.Scatter(
        x=list(range(len(price_data) + 1, len(price_data) + len(predicted_data) + 1)),
        y=predicted_data,
        mode='lines+markers',
        line=dict(color='blue'),
        name='Прогноз'
    ))
    full_fig.update_layout(
        title=f'{name} прогнозируемый график',
        xaxis=dict(title='День',
                   tickvals=list(range(1, len(price_data) + len(predicted_data) + 1)),
                   ticktext=[str(i) for i in range(1, len(price_data) + len(predicted_data) + 1)]
                   ),
        yaxis=dict(title='Цена'),
        template='plotly_dark'
    )
    st.plotly_chart(full_fig, use_container_width=True)

# Обработчик для обновления данных при изменении диапазона дат
if start_date != st.session_state.get("start_date", start_of_week) or end_date != st.session_state.get("end_date",
                                                                                                       end_of_week):
    st.session_state["start_date"] = start_date
    st.session_state["end_date"] = end_date
    st.experimental_rerun()
