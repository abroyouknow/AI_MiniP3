from gensim import similarities, downloader
from gensim.models import KeyedVectors
import csv

model_base_dir = "../Models"
synonyms_file = "../synonyms/synonyms.csv"
deliverables_path = "./deliverables/"
model_name = "word2vec-google-news-300"

detail_results = []


# CSV Format
# Row content
# 0 - quesiton-word
# 1 - correct answer-word
# 2 to 5 - possible choices
def load_synonyms():
    with open(synonyms_file, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        #skip header
        next(csvreader)
        synonyms = list(csvreader)

    return synonyms

def evaluate_synonyms(model, synonyms):



def main():
    # load synonyms
    synonyms = load_synonyms()

    # Open output file
    #details_file = csv.writer(deliverables_path + model_name + "-details.csv", delimiter=',')

    #load model into memory
    print("loading model...")
    model = downloader.load(model_name)

    # Extract model to word vectors
    cosine_value = model.similarity("apple", "orange")
    print(cosine_value)


if __name__ == "__main__":
    main()
