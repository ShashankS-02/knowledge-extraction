import pickle
import uuid
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import tensorflow_hub as hub
from docx import Document
import numpy as np
import pandas as pd
import glob


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)

def knn_model_train(df_encoded_output):
    kneigh_model = KNeighborsClassifier(n_neighbors=7)
    x_train = np.array(df_encoded_output['para_embedding'].tolist())
    y_train = np.array(df_encoded_output['k_means_labels'].tolist())
    print(x_train.shape)
    print(y_train.shape)

    kneigh_model.fit(x_train, y_train)

    with open("knn_trained_model.sav", "wb") as saved_model_file:
        pickle.dump(kneigh_model, saved_model_file)


def knn_model_test(df_encoded_output):
    with open("knn_trained_model.sav", 'rb') as saved_model_file:
        loaded_kneigh_model = pickle.load(saved_model_file)
        result = loaded_kneigh_model.predict(np.array(df_encoded_output['para_embedding'].tolist()))
        df_encoded_output["kneigh_output"] = result

    return df_encoded_output

def paragraph_extractor(filename):
    word_doc = Document(filename)
    paragraphs = []
    for para in word_doc.paragraphs:
        if (para.text.strip()):
            paragraphs.append(para.text.strip())
    return paragraphs


def encode(paragraphs, filename):
    embeddings = model(paragraphs)

    paragraph_encodings = []
    for i, message_embedding in enumerate(np.array(embeddings)):
        para_obj = {}
        para_obj["id"] = uuid.uuid4()
        para_obj["para_text"] = paragraphs[i]
        para_obj["para_embedding"] = message_embedding
        para_obj["filename"] = filename

        paragraph_encodings.append(para_obj)

    return paragraph_encodings


def cluster_paragraphs(df_encoded_output, no_of_clusters):
    kmeans = KMeans(n_clusters=no_of_clusters, random_state=0, n_init="auto")
    kmeans.fit(np.array(df_encoded_output["para_embedding"].tolist()))

    df_encoded_output["k_means_labels"] = kmeans.labels_

    return df_encoded_output


if __name__ == '__main__':
    file_encoding_list = []
    list_of_files_to_process = glob.glob('/Users/shashankshandilya/PycharmProjects/Information-Extraction/train_split/*.docx')
    list_of_files_to_process.extend(glob.glob('/Users/shashankshandilya/PycharmProjects/Information-Extraction/test_split/*.docx'))



    df_encoded_output = pd.DataFrame()
    for filename in list_of_files_to_process:
        paragraphs = paragraph_extractor(filename=filename)
        paragraph_encoding = encode(paragraphs, filename=filename)
        df_encoded_output = pd.concat([df_encoded_output, pd.DataFrame(paragraph_encoding)])

    # files_to_train = ['/Users/shashankshandilya/PycharmProjects/Information-Extraction/train_split/*']
    # df_train = df_encoded_output[["filename"]].isin(files_to_train)
    #
    # files_to_test = ['/Users/shashankshandilya/PycharmProjects/Information-Extraction/test_split/*']
    # df_test = df_encoded_output[["filename"]].isin(files_to_test)

    df_clustered_output = cluster_paragraphs(df_encoded_output, 7)
    df_clustered_train = df_clustered_output[df_clustered_output['filename'].isin(glob.glob('/Users/shashankshandilya/PycharmProjects/Information-Extraction/train_split/*.docx'))]
    knn_model_train(df_clustered_train)

    df_clustered_test = df_clustered_output[df_clustered_output['filename'].isin(glob.glob(
        '/Users/shashankshandilya/PycharmProjects/Information-Extraction/test_split/*.docx'))]
    df_knn_output = knn_model_test(df_clustered_test)

    df_knn_output.to_csv("/Users/shashankshandilya/PycharmProjects/Information-Extraction/knn_input/first_clustering.csv")