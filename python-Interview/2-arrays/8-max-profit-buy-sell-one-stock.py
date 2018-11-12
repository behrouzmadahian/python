'''
given an array of stock prices for time 0-t, find the maximum profit that can be made
'''
def max_profit_buy_sell_once(a):
    min_price_so_far = a[0]
    max_profit = 0
    for i in range(1, len(a)):
        max_profit = max(max_profit, a[i] - min_price_so_far)
        min_price_so_far = min(min_price_so_far, a[i])
    return max_profit

print(max_profit_buy_sell_once([310, 315, 275, 295, 260, 270, 290, 230, 255, 250]))