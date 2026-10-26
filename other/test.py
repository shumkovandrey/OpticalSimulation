t = int(input())

if 6 <= t <= 8:
    print("breakfast")
elif t == 9 or t == 10:
    print("lunch")
elif 11 <= t <= 13:
    print("next lunch")
elif t == 14 or t == 15:
    print("afternoon lunch")
elif t == 16:
    print("afternoon snack")
elif t == 17 or t == 18:
    print("tea")
elif 19 <= t <= 22:
    print("dinner")
else:
    print("bedtime")
