import random

handheld_objects = [
    "coffee_cup",
    "paint_brush",
    "cologne_bottle",
    "ketchup_bottle",
    "hanger",
    "flowers",
    "shorts",
    "rock",
    "carrot",
    "notebook",
    "pepper",
    "apple",
    "bottle_opener",
    "clippers",
    "bicycle_chain",
    "teddy_bear"
]

n = 8

for i in range(48):
    group = set()
    while len(group) < n:
        group.add(random.choice(handheld_objects))
    print(group)