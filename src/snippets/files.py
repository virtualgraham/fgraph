import random 

files_a = [
    "333_apple_bear_carrot_chain_clippers_cup_notebook_opener.mp4",
    "360_brush_chain_cup_hanger_ketchup_opener_pepper_rock.mp4",
    "337_apple_carrot_clippers_cologne_cup_hanger_notebook_pepper.mp4",
    "335_chain_cologne_cup_hanger_ketchup_notebook_opener_shorts.mp4",
    "344_brush_chain_clippers_cologne_cup_flowers_hanger_pepper.mp4"]

files_b = [
    "339_bear_chain_clippers_flowers_hanger_notebook_opener_pepper.mp4",
    "349_apple_bear_chain_clippers_cup_flowers_notebook_shorts.mp4",
    "352_apple_bear_carrot_clippers_cup_flowers_ketchup_notebook.mp4",
    "340_apple_bear_brush_chain_cologne_cup_ketchup_notebook.mp4",
    "364_apple_bear_brush_chain_clippers_flowers_pepper_shorts.mp4",
    "330_brush_cologne_flowers_hanger_notebook_pepper_rock_shorts.mp4",
    "370_apple_brush_flowers_hanger_ketchup_opener_rock_shorts.mp4",
    "354_bear_brush_clippers_cologne_hanger_ketchup_notebook_shorts.mp4",
    "368_carrot_chain_clippers_cologne_cup_hanger_notebook_pepper.mp4",
    "332_apple_chain_cologne_ketchup_opener_pepper_rock_shorts.mp4",
    "325_bear_brush_carrot_clippers_cologne_ketchup_pepper_shorts.mp4",
    "366_apple_brush_chain_cologne_flowers_notebook_opener_pepper.mp4",
    "334_carrot_chain_flowers_hanger_ketchup_notebook_pepper_shorts.mp4",
    "350_apple_bear_cologne_hanger_notebook_pepper_rock_shorts.mp4",
    "353_bear_brush_carrot_chain_cologne_cup_pepper_shorts.mp4",
    "355_apple_bear_brush_carrot_chain_clippers_hanger_rock.mp4",
    "357_brush_carrot_cup_hanger_ketchup_notebook_opener_shorts.mp4",
    "367_bear_brush_clippers_cologne_ketchup_notebook_opener_shorts.mp4",
    "326_apple_bear_cologne_hanger_ketchup_notebook_rock_shorts.mp4",
    "348_apple_brush_carrot_chain_clippers_cologne_cup_ketchup.mp4",
    "338_apple_brush_carrot_clippers_cup_hanger_notebook_shorts.mp4",
    "358_bear_brush_clippers_cologne_cup_flowers_hanger_notebook.mp4",
    "346_apple_brush_chain_clippers_cologne_cup_flowers_shorts.mp4",
    "345_apple_carrot_chain_clippers_cologne_hanger_opener_rock.mp4",
    "327_bear_carrot_chain_cup_hanger_ketchup_pepper_shorts.mp4",
    "336_apple_carrot_clippers_cologne_hanger_notebook_pepper_rock.mp4",
    "347_apple_carrot_clippers_cologne_flowers_hanger_notebook_opener.mp4",
    "356_apple_bear_chain_clippers_cologne_flowers_hanger_pepper.mp4",
    "331_apple_chain_cologne_flowers_ketchup_opener_pepper_rock.mp4",
    "343_apple_bear_carrot_hanger_ketchup_pepper_rock_shorts.mp4",
    "359_apple_bear_carrot_flowers_ketchup_opener_rock_shorts.mp4",
    "369_apple_bear_carrot_chain_cup_flowers_ketchup_opener.mp4",
    "328_bear_brush_chain_cologne_cup_opener_pepper_shorts.mp4",
    "351_apple_bear_cup_flowers_notebook_pepper_rock_shorts.mp4",
    "341_apple_bear_brush_clippers_cologne_flowers_hanger_shorts.mp4",
    "362_bear_carrot_chain_clippers_cologne_ketchup_notebook_shorts.mp4",
    "365_bear_brush_carrot_chain_cup_flowers_ketchup_shorts.mp4",
    "363_apple_bear_chain_clippers_cologne_cup_flowers_ketchup.mp4",
    "361_apple_bear_brush_cologne_flowers_notebook_rock_shorts.mp4",
    "329_carrot_chain_clippers_cologne_hanger_ketchup_notebook_pepper.mp4",
    "342_brush_chain_clippers_hanger_notebook_opener_rock_shorts.mp4"]

i = 0

while True:
    i+=1
    if i % 100000 == 0:
        print(i)

    objects = {
        "apple":0,
        "bear":0,
        "brush":0,
        "carrot":0,
        "chain":0,
        "clippers":0,
        "cologne":0,
        "cup":0,
        "flowers":0,
        "hanger":0,
        "ketchup":0,
        "notebook":0,
        "opener":0,
        "pepper":0,
        "rock":0,
        "shorts":0,
    }

    canditate = files_a + random.sample(files_b, 7)

    for c in canditate:
        for o in objects.keys():
            if o in c:
                objects[o] += 1

    if sum([v >= 5 and v < 8 for v in objects.values()]) == 16 and sum([v == 6 for v in objects.values()]) >= 14:
        print(sum([v == 6 for v in objects.values()]), canditate, objects)
        continue

