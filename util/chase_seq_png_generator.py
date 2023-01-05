"""CHASE sequence PNG generator."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# PC font path
font_path = "C:/Windows/Fonts/"
courier = "courbd.ttf"

# Mac font path
# font_path = 'Library/Fonts/'
# courier = 'cour.dfont'

fnt_a_40 = ImageFont.truetype(font_path + "Arial.ttf", 40)
fnt_a_100 = ImageFont.truetype(font_path + "Arial.ttf", 100)
fnt_c_40 = ImageFont.truetype(font_path + courier, 40)


def write_png_sequence(chase_seq):
    """Write PNG sequence from Chase log file."""
    for row in range(chase_seq.shape[0]):
        arena = chase_seq.iloc[row, 5:406].values.reshape(20, 20)
        e = str(chase_seq.iloc[row, 0])
        s = str(chase_seq.iloc[row, 1])
        a = str(chase_seq.iloc[row, 2])
        r = str(chase_seq.iloc[row, 3])
        arena = np.array2string(arena)

        arena = " " + arena.replace("0", " ").replace("1", "X").replace(
            "2", "X"
        ).replace("3", "R").replace("4", "A").replace("[", "").replace("]", "")

        img = Image.new("RGB", (1920, 1080), color="white")
        d = ImageDraw.Draw(img)

        d.text((24, 10), "CHASE", font=fnt_a_100, fill=(0, 0, 0))
        d.text(
            (32, 108),
            "A toy-text reinforcement learning environment",
            font=fnt_a_40,
            fill=(0, 0, 0),
        )
        d.text((10, 170), arena, font=fnt_c_40, fill=(0, 0, 0))
        d.text(
            (32, 940),
            "Episode: "
            + e
            + "        Step: "
            + s
            + "        Action: "
            + a
            + "        Reward: "
            + r,
            font=fnt_a_40,
            fill=(0, 0, 0),
        )

        if row % 10 == 0:
            print("Completed frame:", row)

        img.save("frames/test" + str(row).zfill(4) + ".png")


# ------
try:
    chase_seq = pd.read_csv("Chase - exp_23e_train - 20190714 - 1239.csv")
except FileNotFoundError:
    print("Can not open log file.")

# chase_seq = chase_seq[chase_seq['Episode'] >= 290000]
# chase_seq.tail()
# write_png_sequence(chase_seq)

# SOME DESCRIPTIVE STATS...
chase_seq["Reward"].replace("None", "0", inplace=True)
chase_seq["Reward"] = chase_seq["Reward"].astype("int64")
# chase_seq['Reward'].dtype

# Total episodes.
chase_ep = chase_seq["Episode"].value_counts().count()
print("Total episodes:", chase_ep)

# Least number of steps.
chase_min_steps = chase_seq.groupby("Episode")["Step"].max().min()
print("Min steps:", chase_min_steps)

# Mean number of steps.
chase_mean_steps = chase_seq.groupby("Episode")["Step"].max().mean()
print("Mean steps:", chase_mean_steps)

# Most number of steps.
chase_max_steps = chase_seq.groupby("Episode")["Step"].max().max()
print("Max steps:", chase_max_steps)

# Mean of reward.
chase_mean_reward = chase_seq["Reward"].sum() / chase_ep
print("Mean reward:", round(chase_mean_reward, 4), " (Random agent: -0.4257)")

# Breakdown of scores
reward_breakdown = chase_seq.groupby("Episode")["Reward"].sum().value_counts()
reward_breakdown.sort_index(inplace=True)
print("Breakdown of reward\n", reward_breakdown)

# Games won
try:
    pct_games_won = (reward_breakdown[5] / chase_ep) * 100
except ValueError:
    pct_games_won = 0

print("Games won : ", round(pct_games_won, 2), "% (Random agent: 0.08%)")


max_series = chase_seq[chase_seq["Step"] == 1001]
max_episodes = max_series["Episode"].values
max_seq = chase_seq[
    (chase_seq["Episode"].isin(max_episodes)) & (chase_seq["Step"] < 100)
]
write_png_sequence(max_seq)
