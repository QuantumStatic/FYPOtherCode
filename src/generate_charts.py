import matplotlib.pyplot as plt
from myfunctions import execute_this, clear_output_screen
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import csv
import nltk
from nltk.util import ngrams
import pathlib
import seaborn as sns
import numpy as np
import random
import statistics

# nltk.download('punkt')


FYP_FOLDER_PATH = pathlib.Path("/Users/utkarsh/Desktop/Utkarsh/College/Year 4/FYP")
BIG_DS_PATH = FYP_FOLDER_PATH / "Dataset"
CHART_PATH = FYP_FOLDER_PATH / 'Charts'



def create_word_cloud() -> None:
    """
    This code reads in a CSV file containing text data, creates a word cloud from the n-grams (contiguous sequences of n words) 
    in the text, and saves the resulting image to a file. The CountVectorizer from the sklearn library is used to generate the 
    n-grams, and the WordCloud library is used to create the word cloud. The resulting plot is saved to a file in the CHART_PATH 
    directory with a filename based on the minimum and maximum n-gram lengths. The code also includes an optional line to display 
    the plot in a window, which can be uncommented if desired.
    """


    # Initializing an empty list to store all data
    all_data:list[str] = []

    # Opening the csv file and reading the data into the list
    with open(f"{BIG_DS_PATH}/big_ds_cleaned.csv", "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            all_data.append(row[0])
    
    # Joining all the data into a single string
    all_data = " ".join(all_data)
    
    # Seting the minimum and maximum n-gram range for the CountVectorizer
    # For example, if min_n = 2 and max_n = 5, the CountVectorizer creates n-grams of 2, 3, 4, and 5 words
    # The higher the n-gram range, the more accurate the word cloud, but it takes longer to generate
    min_n = 2
    max_n = 5
    
    # Creates a CountVectorizer object with the specified n-gram range and fits it to the data
    vectorizer = CountVectorizer(ngram_range=(min_n, max_n))
    _ = vectorizer.fit_transform([all_data])
    
    # Gets the list of feature names (i.e., the n-grams)
    features = vectorizer.get_feature_names_out()
    
    # Joins the feature names into a single string
    n_grams = ' '.join(features)
    
    # Creates a WordCloud object with the specified dimensions, background color, and font size, and generates the word cloud
    wordcloud = WordCloud(width=1600, height=1600, background_color="white", min_font_size=8).generate(n_grams)
    
    # Creates a plot to display the word cloud and saves it as a PNG file
    plt.figure(figsize=(10, 10), facecolor=None)
    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig(f"{CHART_PATH}/word_clouds/{min_n}_{max_n}.png", dpi=640)
    plt.show()


def popular_words():
    """
    This code reads data from a CSV file, counts the occurrences of n-grams (up to trigrams), and creates bar charts to 
    visualize the top 10 n-grams for each value of n. It also prints the top 10 n-grams for each value of n to the console.
    """

    # creates an empty list to store all data
    all_data:list[str] = []
    
    # reads data from a csv file and appends each line to the all_data list
    with open(f"{BIG_DS_PATH}/big_ds_cleaned.csv", "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            all_data.append(row[0])

    # creates empty dictionaries to store n-gram counts for each value of n
    gram_dicts: list[dict[str, int]] = []
    
    # iterates over n-grams from 1 to 3 and counts their occurrences
    for x in range(1, 4):
        gram_dicts.append({})
        for line in all_data:
            for gram in ngrams(nltk.word_tokenize(line), x):
                try:
                    gram_dicts[-1][gram] += 1
                except KeyError:
                    gram_dicts[-1][gram] = 1
        
        # sorts the n-gram dictionary by count in descending order
        gram_dicts[-1] = {k: v for k, v in sorted(gram_dicts[-1].items(), key=lambda item: item[1], reverse=True)}
    
    # removes certain n-grams that are not useful for analysis
    try:
        del gram_dicts[0][('says',)]
        del gram_dicts[0][('New',)]
        del gram_dicts[2][('5', 'things', 'know')]
        del gram_dicts[2][('things', 'know', 'stock')]
    except KeyError:
        pass
    
    # creates bar charts to visualize the top 10 n-grams for each value of n
    for idx, gram_dict in enumerate(gram_dicts):
        sns.set_style('whitegrid')
        if idx == 0:
            plt.figure(figsize=(10, 10))
            ax = sns.barplot(y=list(gram_dict.values())[:10], x=list(map(lambda x: x[0], tuple(gram_dict.keys())[:10])))
        elif idx > 0:
            plt.figure(figsize=(10, 10 + (idx*2)))
            ax = sns.barplot(y=list(gram_dict.values())[:10], x=list(map(' '.join, tuple(gram_dict.keys())[:10])))
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel(f'{idx+1}-grams')
        ax.set_ylabel('Occurences')
        plt.savefig(f"{CHART_PATH}/bar_charts/{idx+1}_grams.png", dpi=640)

    # prints the top 10 n-grams for each value of n to the console
    for gram_dict in gram_dicts:
        for x, item in enumerate(gram_dict.items()):
            if x == 10:
                input("Press any key to continue...")
                clear_output_screen()
                break
            print(item)



def lr_decay_chart():
    """
    This function generates a plot to visualize the learning rate decay over time using an exponential decay function. 
    It uses the NumPy and Seaborn libraries to create and plot the data. 
    The resulting plot is saved to a file in the specified directory.
    """

    # Creates an array of values for the x-axis
    x = np.linspace(0, 232, 232*4)
    # Calculates the y-values using the exponential decay formula
    y = (0.99**x) * 1e-3

    # Creates a line plot using Seaborn
    ax = sns.lineplot(x=x, y=y)

    plt.title("Learning Rate Decay")
    plt.xlabel("Scheduler Steps")
    plt.ylabel("Learning Rate")

    # Saves the plot to a file
    plt.savefig(f"{CHART_PATH}/line_charts/lr_decay.png", dpi=640)
    plt.show()


def cut_off_analytics():
    """
    Creates an empty dictionary to hold the number of datapoints with a certain length
    The keys of the dictionary are the length of the datapoints divided by 5 (rounded down)
    The values are the number of datapoints with that length
    """

    length_nums: dict[int, int] = {}

    # Initializes a variable to keep track of the total number of datapoints
    tot = 0

    # Opens the CSV file containing the data and reads it line by line
    with open(f"{BIG_DS_PATH}/big_ds_cleaned.csv", 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            # Tries to increment the count of datapoints with a certain length
            try:
                length_nums[len(line[0]) // 5] += 1
            # If the length is not yet in the dictionary, adds it with a count of 1
            except KeyError:
                length_nums[len(line[0]) // 5] = 1
            finally:
                tot += 1

    # Calculates the total number of datapoints with length less than or equal to 180
    # (which corresponds to a length of 36 after dividing by 5)
    # This will be used to calculate the percentage of data covered by datapoints with length less than or equal to 180
    min_tot = 0
    for x in range(36):
        try:
            min_tot += length_nums[x]
        except KeyError:
            pass
        
    # Prints out the percentage of data covered by datapoints with length less than or equal to 180
    # (i.e., datapoints with length less than or equal to 36 after dividing by 5)
    print(f"data covered: {(min_tot / tot) * 100:.3f} by {36 * 5}")

    # Sets the number of bars to display in the bar chart of the distribution of datapoint lengths
    bars = 25

    # Sorts the length_nums dictionary by key (i.e., length)
    length_nums = dict(sorted(length_nums.items()))

    # Creates a bar chart of the distribution of datapoint lengths
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        x=list(length_nums.keys())[:bars+2],
        y=list(length_nums.values())[:bars+2]
    )

    # Sets the x-ticks with a range of values and corresponding labels for each value
    plt.xticks(
        ticks=[i for i in range(0, bars+2)],
        labels=[f"{i*5}-{(i+1)*5}" for i in range(0, bars+2)],
        rotation=45
    )

    # Labels the plot and saves it
    plt.xlabel("Length of Datapoint (in characters)")
    plt.ylabel("Number of Datapoints")
    plt.title("Distribution of Datapoint Lengths")
    plt.savefig(f"{CHART_PATH}/bar_charts/cumulative_3.png", dpi=640)

    plt.show()


    # Normalizes the values in length_nums dictionary by the total number of datapoints
    length_nums[0] /= tot

    # Loops through the keys in length_nums dictionary
    for x in range(1, len(length_nums.keys())):
        try:
            # Normalizes the value by the total number of datapoints
            length_nums[x] /= tot
            
            # Adds the previous value to the current value to get the cumulative distribution
            length_nums[x] += length_nums[x-1]
            
            # Removes the key from dictionary if the cumulative distribution is same as previous key's and breaks the loop
            if length_nums[x] == length_nums[x-1]:
                del length_nums[x]
                break
        except KeyError:
            # If key is not in the dictionary, sets it equal to the previous key's value
            length_nums[x] = length_nums[x-1]

    # Sorts the dictionary by keys
    length_nums = dict(sorted(length_nums.items()))

    # Sets the number of bars to display in the plot
    bars = 42

    # Creates a bar plot with the first 'bars' keys and values in length_nums dictionary
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(tuple(length_nums.keys())[:bars], tuple(length_nums.values())[:bars])

    # Highlights the threshold bar
    ax.bar(36, length_nums[36], color='red')

    # Draws a horizontal line at the level of the highlighted datapoint with red dashed line
    plt.axhline(y=length_nums[36], color='red', linestyle='--', alpha=0.65)

    # Adds a text label to the highlighted datapoint length
    plt.text(0, length_nums[36], f"{length_nums[36]:.3f}", color='red', va='bottom', ha="left")

    # Sets the labels
    ax.set_xticks(ticks=[i for i in range(bars)])
    ax.set_xticklabels(labels=[f"{i*5}-{5*(i+1)}" for i in range(bars)], rotation=90)
    ax.set_yticks([i/10 for i in range(11)])

    ax.grid(axis='y', alpha=0.65)

    ax.set_xlabel("Length of text (in words)")
    ax.set_ylabel("Percentage of Datapoints")
    ax.set_title("Cumulative Distribution of Datapoint Lengths")

    # Sets the padding for the plot
    plt.tight_layout(pad=1)

    # Saves the plot to a file with a given filename and dpi
    plt.savefig(f"{CHART_PATH}/bar_charts/cut_off_2.png", dpi=640)

    plt.show()



def generate_lr_classifier_chart():
    """
    Define a function called "generate_lr_classifier_chart" that generates a chart showing the loss vs learning rate across epochs
    """

    # Initializes an empty dictionary called "lr_data" that stores learning rate values and their corresponding losses
    lr_data: dict[float, list[float]] = {}
    
    # Iterates over all CSV files in a specific directory that match the naming pattern "lr_*.csv"
    for lr_csv_file in FYP_FOLDER_PATH.joinpath("final report/results_classfier").glob("lr_*.csv"):
        # Extracts a processed name from the file name by removing the extension and converting to a float value
        processed_name = round(float('.'.join(lr_csv_file.name.split('.')[:-1])[3:]), 7)
        # Creates a new key in "lr_data" for the processed name, with an empty list as its value
        lr_data[processed_name] = []
        
        # Opens the CSV file for reading and iterates over its rows
        with open(lr_csv_file, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                # Appends the first value in each row (i.e., the loss value) to the corresponding list in "lr_data"
                lr_data[processed_name].append(float(row[0]))

    # Sorts the items in "lr_data" by their keys (i.e., the learning rate values)
    lr_data = dict(sorted(lr_data.items()))
    # Creates a list of x-values (i.e., epochs) for the chart, with values evenly spaced between 0 and 2 (the total number of epochs was 26)
    x = [(2/26)*(i+1) for i in range(len(lr_data[tuple(lr_data.keys())[0]]))]

    # Iterates over the items in "lr_data" and plots a line for each one on the chart
    for idx, (file_name, lr_vals) in enumerate(lr_data.items()):
        # Uses Seaborn's "lineplot" function to plot the values in "lr_vals" on the y-axis and the values in "x" on the x-axis
        # Labels each line with the corresponding file name from the "lr_*.csv" file that it came from, and uses a pastel color palette
        sns.lineplot(x=x, y=lr_vals, label=file_name, color=sns.color_palette("pastel")[idx])

    # Adds the labels to the cart
    plt.title("Loss vs Learning Rate across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross Entropy)")
    plt.legend()

    # Saves the chart
    plt.savefig(CHART_PATH.joinpath("line_charts/lr.png"), dpi=640)
    plt.show()



def generate_wd_classifier_chart():
    """
    This function generates a line chart comparing the loss and weight decay across epochs
    """

    # Initializes a dictionary to store weight decay data with the type hint for values being a list of floats
    wd_data: dict[float, list[float]] = {}
    
    # Loops through each csv file starting with "wd_" in the results_classifier folder of the FYP directory
    for wd_csv_file in FYP_FOLDER_PATH.joinpath("final report/results_classfier").glob("wd_*.csv"):
        # Processes the file name to get the weight decay value and adds a key to the dictionary with an empty list as the value
        processed_name = round(float('.'.join(wd_csv_file.name.split('.')[:-1])[3:]), 7)
        wd_data[processed_name] = []

        # Reads the csv file and appends each row value (converted to float) to the corresponding list in the dictionary
        with open(wd_csv_file, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                wd_data[processed_name].append(float(row[0]))

    # Sorts the dictionary by keys (which are weight decay values) in ascending order
    wd_data = dict(sorted(wd_data.items()))

    # Generates x values based on the number of data points (epochs) in the first entry of the dictionary values
    x = [(2/26)*(i+1) for i in range(len(wd_data[tuple(wd_data.keys())[0]]))]

    # Loops through each item in the dictionary and plot a line for the corresponding weight decay value
    for idx, (file_name, wd_vals) in enumerate(wd_data.items()):
        sns.lineplot(x=x, y=wd_vals, label=file_name, color=sns.color_palette("pastel")[idx])

    # Adds the labels
    plt.title('Loss vs Weight Decay across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Cross Entropy)')

    plt.legend()

    # Saves the chart
    plt.savefig(CHART_PATH.joinpath("line_charts/wd.png"), dpi=640)

    plt.show()



def generate_confusion_matrix_test():
    """
    This function generates a confusion matrix for the test set.
    """

    # Defines the confusion matrix
    matrix = [[8929, 1710], [1571, 16509]]
    
    # Creates a heatmap of the confusion matrix (commented out)
    # sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["biased", "Non-biased"], yticklabels=["biased", "Non-biased"])
    # plt.savefig(CHART_PATH.joinpath("Diagrams/con_matrix.png"), dpi=640)
    
    # Creates the predicted and true labels for the classification report and ROC curve
    preds = [0 for _ in range(8929)]
    labels = [0 for _ in range(8929)]
    preds.extend([1 for _ in range(16509)])
    labels.extend([1 for _ in range(16509)])
    preds.extend([0 for _ in range(1571)])
    labels.extend([1 for _ in range(1571)])
    preds.extend([1 for _ in range(1710)])
    labels.extend([0 for _ in range(1710)])

    # Generates and prints the classification report
    print(classification_report(labels, preds, target_names=["biased", "Non-biased"]))
    
    # Calculates the ROC AUC score
    roc_score = roc_auc_score(labels, preds)
    print(roc_score)
    
    # Calculates the false positive rate, true positive rate, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, preds)
    
    # Sets the style and creates the ROC curve
    sns.set_style('whitegrid')
    plt.plot(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_score)
    plt.plot([0, 1], [0, 1], linestyle="dashdot")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    
    # Saves the ROC curve
    plt.savefig(CHART_PATH.joinpath("Diagrams/roc_curve.png"), dpi=640)



def create_paraphraser_L_chart():
    """
    This function creates a line chart comparing the loss across epochs for the paraphraser model.
    """
    
    # Reads values from a CSV file
    values = []
    with open(FYP_FOLDER_PATH.joinpath("vals.csv"), 'r', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            values.append(float(row[0]))
    
    # Prints the length of values list
    print(len(values))
    
    # Sets seaborn style
    sns.set(style="darkgrid")
    
    # Creates a line plot with x values as epoch numbers and y values as loss values
    sns.lineplot(x=[i/930 for i in range(len(values))], y=values)

    # Adds x and y axis labels and a title to the plot
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross Entropy)")
    plt.title("Loss vs Epochs for Paraphraser")
    
    # Saves the plot
    plt.savefig(CHART_PATH.joinpath("line_charts/paraphraser_L.png"), dpi=640)
    
    plt.show()


@execute_this(stack_trace=False)
def main():
    cut_off_analytics()