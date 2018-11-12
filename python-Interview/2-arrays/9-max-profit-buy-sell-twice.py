'''
calculate maximum profit if we can make at most two buys and sells
the second buy MUSt happen after the first sell
we calculate:
1- max profit if we sell on day i
2- max profit if we buy on day i
'''
def max_profit_two_times(a):
    min_price_so_far = a[0]
    max_total_profit = 0
    # profit if we sell on day i:
    first_trade_profit = [0] * len(a)
    for i in range(0, len(a)):
        min_price_so_far = min (a[i], min_price_so_far)
        max_total_profit = max(max_total_profit, a[i] - min_price_so_far)
        first_trade_profit[i] = max_total_profit

    print(first_trade_profit)

    # profit if we buy on day i
    max_price_sofar = 0
    for i, price in reversed(list(enumerate(a[1:], 1))):
        print(i, price)
        max_price_sofar = max(max_price_sofar, price)
        max_total_profit = max(max_total_profit, max_price_sofar - price + first_trade_profit[i-1])

    return max_total_profit

print(max_profit_two_times([12,11,13,9,12,8,14,13,15]))


