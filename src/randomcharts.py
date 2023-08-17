import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import string
import nltk
from nltk.corpus import words
nltk.download('words')

random.seed=1

# Function to get random English words
def random_english_words(n):
    word_list = words.words()
    return random.sample(word_list, n)

# Number of categories
num_categories = 10

def gen_random_categorical_data(num_categories:int=-1):
    '''
    Generate random categorical data
    '''
    if num_categories <= 0:
        num_categories = random.randint(2, 20)
    # Create random data
    data = {'Category': random_english_words(num_categories),
            'Value': np.random.randint(10, 100, size=num_categories)}

    df = pd.DataFrame(data)
    return df


def gen_random_horiz_barchart(img_outfile:str=None,
    dpi=300,
    show=True):
    # Randomly generate data
    # data = {'Category': [f'Category {i+1}' for i in range(10)],
    #        'Value': np.random.randint(10, 100, size=10)}
    data = gen_random_categorical_data()
    df = pd.DataFrame(data)

    # Random color generator
    def random_color():
        return f'#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}'

    # Random font size generator
    def random_font_size():
        return random.uniform(8, 20)

    # Random string generator
    def random_string(length):
        return ''.join(random.choices(string.ascii_letters, k=length))

    # Set plot style and background color
    sns.set(style="whitegrid")
    plt.rcParams["axes.facecolor"] = random_color()

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Value', y='Category', data=df, palette="husl", ax=ax)

    # Set random foreground color
    ax.set_facecolor(random_color())

    # Set random font properties for axis labels, title, and tick labels
    for axis in ['x', 'y']:
        ax.tick_params(axis=axis, labelsize=random_font_size(), colors=random_color())
        ax.set_xlabel(ax.get_xlabel(), fontsize=random_font_size(), color=random_color())
        ax.set_ylabel(ax.get_ylabel(), fontsize=random_font_size(), color=random_color())

    # Set random title and axis labels
    ax.set_title(random_string(20), fontsize=random_font_size(), color=random_color())
    ax.set_xlabel(random_string(10), fontsize=random_font_size(), color=random_color())
    ax.set_ylabel(random_string(10), fontsize=random_font_size(), color=random_color())

    # Show plot
    plt.tight_layout()
    if show:
        plt.show()
    if img_outfile is not None:
        print(f"saving to {img_outfile}")
        plt.savefig(img_outfile, dpi=dpi, bbox_inches='tight')

def run_random_horiz_barchart(
    outfilebase:str,
    n=10,dpi=150,show=False):
    '''
    Randomly generated chart
    '''
    for i in range(n):
        outfile = outfilebase + '_' + str(i) + '.jpg'
        gen_random_horiz_barchart(img_outfile=outfile,
            dpi=dpi,
            show=show)        


if __name__ == '__main__':
    gen_random_horiz_barchart(show=True)
    # production run of generating 1000 images:
    # run_random_horiz_barchart(n=1000, outfilebase = 'tmp_horizbarcharts/horizbar')