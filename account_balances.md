# Account Balances
>This object oriented program simulates account balance transactions via two bank accounts - a checking and a savings account. 
### Step 1: create a parent class "Account" with 3 state variables.
```python
class Account:
    def __init__(self, name, balance, min_balance):
        self.name = name
        self.balance = balance
        self.min_balance = min_balance
```
### Step 2: create class methods
```python
    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if self.balance - amount >= self.min_balance:
            self.balance -= amount
        else:
            print("Sorry, insufficient funds!")

    def statement(self):
        print("Account Balance: ${}".format(self.balance))
```
### Step 3: create a constructor for the checking account; we pass a parameter of an overdraft balance of $1,000 to the parent account.
```python
class Checking(Account):
    def __init__(self, name, balance):
        super().__init__(name, balance, min_balance = -1000)

    def __str__(self):
        return "{}'s Checking Account: Balance ${}".format(self.name, self.balance)
```
> Simulated Transactions: Checking account
```python
M = Checking("Mario", 500)
M.deposit(300)
M.statement()
```
*Account Balance: $800*
```python
M.withdraw(1000)
M.statement()
```
*Account Balance: $-200*
```python
M.withdraw(800)
M.statement()
```
*Account Balance: $-1000*
```python
M.withdraw(1)
```
*Sorry, insufficient funds!*
```python
print(M)
```
*Mario's Checking Account: Balance $-1000*
```python
M = Checking("Mario", 500)
print(M)
```
*Mario's Checking Account: Balance $500*
### Step 4: create a constructor for the Savings account; we pass a parameter of a minimum balance of $0 to the parent account.
```python
class Savings(Account):
    def __init__(self, name, balance):
        super().__init__(name, balance, min_balance = 0)

    def __str__(self):
        return "{}'s Savings Account: Balance ${}".format(self.name, self.balance)
```
> Simulated Transactions: Savings account
```python
T = Savings("Tom", 300)
print(T)
```
*Tom's Savings Account: Balance $300*
```python
T.withdraw(300)
T.statement()
```
*Account Balance: $0*
```python
T.withdraw(1)
```
*Sorry, insufficient funds!*

