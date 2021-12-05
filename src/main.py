from gensim import downloader
import csv
import os

synonyms_file = "../synonyms/synonyms.csv"
deliverables_path = "../deliverables/"

# Task 1
google_model_name = "word2vec-google-news-300"

# Task 2.1
twitter_model_name_50 = "glove-twitter-50"
wiki_model_name = "glove-wiki-gigaword-50"

# Task 2.2
twitter_model_name_25 = "glove-twitter-25"
twitter_model_name_100 = "glove-twitter-100"

detail_results = []

counter_correct = 0
counter_wrong = 0
counter_guess = 0


# CSV Format
# Row content
# 0 - question-word
# 1 - correct answer-word
# 2 to 5 - possible choices
def load_synonyms():
    with open(synonyms_file, 'r') as csv_file:
        # creating a csv reader object
        csv_reader = csv.reader(csv_file)

        # skip header
        next(csv_reader)
        synonyms = list(csv_reader)

    return synonyms


def evaluate_synonyms(model, synonyms):
    global counter_guess, counter_wrong, counter_correct
    for row in synonyms:
        # Fetch question word and correct answer
        question_word = row[0]
        correct_answer = row[1]

        # Make a guess
        guess_word = ''
        if question_word in model.key_to_index:
            guess_similarities = []
            for i in range(2, 6):
                try:
                    guess_similarities.append((row[i], model.similarity(question_word, row[i])))
                except:
                    pass
            if len(guess_similarities) > 0:
                guess = max(guess_similarities, key=lambda x: x[1])
                guess_word = guess[0]

        # Determine result state
        result = ""
        if guess_word == correct_answer:
            result = "correct"
            counter_correct += 1
        elif guess_word == "":
            result = "guess"
            counter_guess += 1
        else:
            result = "wrong"
            counter_wrong += 1

        # Append to details list
        detail_results.append([question_word, correct_answer, guess_word, result])


def evaluate_synonyms_with_model(model_name, synonyms_list, first=False):
    global detail_results, counter_correct, counter_wrong, counter_guess

    # Load model into memory
    print(F"Loading {model_name}...\n")
    model = downloader.load(model_name)

    # Evaluate guesses for each synonym question
    print('Evaluating synonyms...\n')
    evaluate_synonyms(model, synonyms_list)

    # Print counter results
    print(F'Correct\t\t{counter_correct}')
    print(F'Wrong\t\t{counter_wrong}')
    print(F'Guess\t\t{counter_guess}\n')

    # Create deliverables folder if it does not exist
    if not os.path.exists(deliverables_path):
        os.makedirs(deliverables_path)

    # Write CSV details output file
    with open(deliverables_path + model_name + "-details.csv", 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        csv_writer.writerows(detail_results)

    # Write CSV analysis output csv_file
    with open(deliverables_path + "analysis.csv", 'w' if first else 'a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')

        without_guess = 80 - counter_guess
        accuracy = counter_correct / without_guess
        csv_writer.writerow([model_name, len(model.index_to_key), counter_correct, without_guess, accuracy])

    detail_results = []
    counter_correct = 0
    counter_wrong = 0
    counter_guess = 0


def main():
    # Load synonyms
    synonyms = load_synonyms()

    # Task 1
    evaluate_synonyms_with_model(google_model_name, synonyms, True)

    # Task 2.1
    evaluate_synonyms_with_model(twitter_model_name_50, synonyms)
    evaluate_synonyms_with_model(wiki_model_name, synonyms)

    # Task 2.2
    evaluate_synonyms_with_model(twitter_model_name_25, synonyms)
    evaluate_synonyms_with_model(twitter_model_name_100, synonyms)


if __name__ == "__main__":
    main()
