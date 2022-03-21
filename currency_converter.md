# Crypto Currency (aka "Bitcoin currency" Converter
This program converts the value of various crypto currencies into its value in US dollars. The crypto currency converter program is divided into 4 parts:  
1. we show the basic algorithm that converts the amount of US dollars as given by the user as input, into the number of bitcoins. 
1. we generalize our bitcoin currency converter model by providing realtime bitcoin exchange rates. This is done by linking our API on the front-end to the back-end database server of coindesk.com - a leading news site that specializes in bitcoin and digital currencies. 
1. we convert our bitcoin price from a string format to a decimal format. We use a decimal format because of its precision in performing calculations.
1. Lastly, we build a crypto currency price checker that checks the prices of various bitcoin ticker symbols and converts them to US$.

### Part 1: Basic algorithm that converts US$ (in whole number) to the value in bitcoin currency.

```python
dollars = int(input('How many dollars to convert?: '))
price = 19266.70

print(f'You can purchase {dollars/price} bitcoin')
```  

> Part 1: Simulated conversion from US$ (in whole numbers only) to Bitcoin

*How many dollars to convert?: 20000*  
*You can purchase 1.0380604877846231 bitcoin*

### Part 2: Generalize the Bitcoin conversion model via realtime bitcoin exchange rates.
>We import the requests module which allows us to send HTTP requests to the Coindesk exchange platform using Python. The HTTP request returns a Response Object with all the response data (Bitcoin price index 'BPI' and current price ('rate_float') in US dollars 'USD').

```python
import requests

dollars = int(input('How many dollars to convert?: '))
response = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')

price = response.json().get('bpi').get('USD').get('rate_float')

print(f'You can purchase {dollars/price} bitcoin')
```
> Part 2: Simulated conversion from US$ (in whole numbers only) to Bitcoin

*How many dollars to convert?: 20000*  
*You can purchase 0.48564171967160724 bitcoin*

### Part 3: Change bitcoin price format from string to decimal.
>We import the Decimal module from the decimal library in python. We recommend the use of decimal format for banking transactions since they tend to be more precise.

```python
import requests
from decimal import Decimal

response = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
price = Decimal(response.json().get('bpi').get('USD').get('rate').replace(',', ''))

print(price)
```
> Part 3: Output of HTTP bitcoin price request from our API to the Coindesk exchange platform.

*41254.0350*

### Part 4: We build a Crypto currency price checker API to check the price of various bitcoin ticker symbols. HTTP price requests are sent to coinmarketcap.com. 

>Coinmarketcap.com provides an authentication code each time a request is made. 
- Crypto currency ticker symbols are also provided by Coinmarket.com. 
- If the user enters an invalid crypto ticker symbol, the API program produces an error message and requests the user to re-enter a valid crypto ticker symbol.

```python
import requests

def get_coin_id(symbol):
    response = requests.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/map?CMC_PRO_API_KEY=22198400-e4e2-4c9d-9dad-72c7529ab31f&symbol=' + symbol)
    return response.json().get('data')[0].get('id')

def get_coin_price(coin_id):
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    key = '22198400-e4e2-4c9d-9dad-72c7529ab31f'
    response = requests.get(f'{url}?CMC_PRO_API_KEY={key}&id={coin_id}')
    return response.json().get('data').get(str(coin_id)).get('quote').get('USD').get('price')

try:
    coin = input('Enter a crypto symbol: ')
    coin_id = get_coin_id(coin)
    price = get_coin_price(coin_id)

    print(f'One {coin} is worth ${round(price, 2)}')
except Exception as e:
    print("Error: enter a valid symbol.")
```
> Part 4: Output of Crypto currency price checker. The API prompts the user to enter a crypto ticker symbol. The API returns the cryptocurrency value of the ticker symbol.

*Enter a crypto symbol: BTC*  
*One BTC is worth $41168.4*

>The user enters an invalid ticker symbol. The API outputs an error message and requests the user to re-enter a valid symbol.

*Enter a crypto symbol: LDKSL*  
*Error: enter a valid symbol.*
