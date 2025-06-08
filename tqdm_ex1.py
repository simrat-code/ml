import random

from tqdm import tqdm
from time import sleep

def do_something():
    num = random.randint(1, 3)
    sleep(num)

def run_tqdm(count=10):
    for i in tqdm(range(count), desc="Processing"):
        do_something()


if __name__ == "__main__":
    run_tqdm()