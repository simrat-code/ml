import random

from tqdm import tqdm
from time import sleep

def do_something():
    num = random.randint(1, 3)
    sleep(num)

for i in tqdm(range(50), desc="Processing"):
    do_something()