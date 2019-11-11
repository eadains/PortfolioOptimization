import aiohttp
import asyncio
import pandas as pd
from config import INTRINIO_API


async def get(session, url, params=None):
    """
    Does a single HTTP fetch with error handling. Returns JSON decoded response.
    :param session: aiohttp session
    :param url: URL to fetch
    :param params: additional URL parameters
    :return: JSON decoded HTTP response, or aiohttp ClientResponseError
    """
    retries = 0
    while True:
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                return await resp.json()
        # Retry up to 5 times, then raise exception
        except aiohttp.ClientResponseError as ex:
            if retries < 5:
                retries += 1
                await asyncio.sleep(0.10)
                continue
            else:
                raise ex


async def pages_get(session, url, output_field, **kwargs):
    """
    Handles pagination in Intrinio API. **kwargs passes any extra URL endpoints
    :param session: aiohttp session
    :param url: URL to fetch
    :param output_field: JSON output field to keep. Ex: in the prices endpoint, 'stock_prices' keeps the pricing data
    :return: list
    """
    data = []
    params = kwargs

    while True:
        # First loop params['next_page'] doesnt exist. Each loop after params gets updated with next page
        json = await get(session, url, params)
        if json['next_page']:
            # Appends data on page and updates params so next page is fetched
            data.extend(json[output_field])
            params.update({'next_page': json['next_page']})
            continue
        else:
            data.extend(json[output_field])
            return data


async def prices_download(tickers, start_dates):
    """
    Downloads from Intrinio security prices endpoint. Designed to be used by database.update_prices()
    :param tickers: list-like of strings
    :param start_dates: list-like of ISO formatted date strings to be passed to 'start_date' URL parameter.
    :return: pandas dataframe
    """
    headers = {'Authorization': f'Bearer {INTRINIO_API}'}

    async with aiohttp.ClientSession(headers=headers) as session:
        # Form task for each ticker and corresponding start_date. Passes no parameter if start_date == None
        tasks = []
        for ticker, date in zip(tickers, start_dates):
            url = f'https://api-v2.intrinio.com/securities/{ticker}/prices'
            if date:
                # psycopg2 returns dates as datetime objects, so use string representation
                tasks.append(pages_get(session, url, 'stock_prices', start_date=str(date)))
            else:
                tasks.append(pages_get(session, url, 'stock_prices'))

        data = await asyncio.gather(*tasks)

        frames = [pd.DataFrame(x) for x in data]
        # Add ticker column to each dataframe
        for frame, ticker in zip(frames, tickers):
            frame['ticker'] = ticker
        frame = pd.concat(frames)
        return frame


async def adjustments_download(tickers, start_dates):
    """
    Downloads from Intrinio security price adjustments endpoint. Designed to be used by database.update_adjustments()
    :param tickers: list-like of strings
    :param start_dates: list-like of ISO formatted date strings to be passed to 'start_date' URL parameter.
    :return: pandas dataframe
    """
    headers = {'Authorization': f'Bearer {INTRINIO_API}'}

    async with aiohttp.ClientSession(headers=headers) as session:
        # Form task for each ticker and corresponding start_date. Passes no parameter if start_date == None
        tasks = []
        for ticker, date in zip(tickers, start_dates):
            url = f'https://api-v2.intrinio.com/securities/{ticker}/prices/adjustments'
            if date:
                # psycopg2 returns dates as datetime objects, so use string representation
                tasks.append(pages_get(session, url, 'stock_price_adjustments', start_date=str(date)))
            else:
                tasks.append(pages_get(session, url, 'stock_price_adjustments'))

        data = await asyncio.gather(*tasks)

        frames = [pd.DataFrame(x) for x in data]
        # Add ticker column to each dataframe
        for frame, ticker in zip(frames, tickers):
            frame['ticker'] = ticker
        frame = pd.concat(frames)
        return frame


async def direct_download(tickers):
    """
    Download from Intrinio security prices endpoint for each ticker. Returns only adjusted columns.
    :param tickers: list-like of strings
    :return: dataframe
    """
    urls = [f'https://api-v2.intrinio.com/securities/{ticker}/prices' for ticker in tickers]
    headers = {'Authorization': f'Bearer {INTRINIO_API}'}

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [pages_get(session, url, 'stock_prices') for url in urls]
        data = await asyncio.gather(*tasks)

        frames = [pd.DataFrame(x) for x in data]
        # Add ticker column to each dataframe
        for frame, ticker in zip(frames, tickers):
            frame['ticker'] = ticker
        frame = pd.concat(frames)
        # Format date column and set multiindex
        frame['date'] = pd.to_datetime(frame['date'])
        frame.set_index(['date', 'ticker'], inplace=True)
        frame.sort_index(inplace=True)
        return frame[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]


async def distinct_download(tickers):
    """
    Download from Intrinio security prices endpoint for each ticker. Creates dict instead of concatenated dataframe.
    :param tickers: list-like of strings
    :return: dict {ticker: dataframe, ...}
    """
    urls = [f'https://api-v2.intrinio.com/securities/{ticker}/prices' for ticker in tickers]
    headers = {'Authorization': f'Bearer {INTRINIO_API}'}

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [pages_get(session, url, 'stock_prices') for url in urls]
        data = await asyncio.gather(*tasks)

        frames = [pd.DataFrame(x) for x in data]
        # Add ticker column to each dataframe
        for frame in frames:
            frame['date'] = pd.to_datetime(frame['date'])
            frame.set_index('date', inplace=True)
            frame.sort_index(inplace=True)
            frame.drop(columns=['open', 'high', 'low', 'close', 'volume', 'frequency', 'intraperiod'], inplace=True)

        return dict(zip(tickers, frames))
