import uuid
from sklearn.cluster import KMeans
import tensorflow_hub as hub
from docx import Document
import numpy as np
import pandas as pd
import re



module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)


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


# TODO - Add clustering code for 2nd level function
def second_level_clustering(df_encoded_output, key, no_of_clusters):
    df_secondary_clusters = cluster_paragraphs(df_encoded_output[df_encoded_output['k_means_labels'] == key][['id', 'para_text', 'para_embedding', 'k_means_labels']], no_of_clusters)
    df_hierarchical_cluster_output = df_encoded_output.merge(df_secondary_clusters[["id", "k_means_labels"]], left_on="id", right_on="id", how="left", suffixes=("_primary", "_secondary"))
    return df_hierarchical_cluster_output


def create_vendor_purchaser_dataframe(vendor_detail_list, purchaser_detail_list,df_hierchical_clustered_output):
    df_vendor_list = pd.DataFrame(vendor_detail_list)
    df_purchaser_list = pd.DataFrame(purchaser_detail_list)
    df_merged_vendor_details = pd.merge(df_hierchical_clustered_output, df_vendor_list, on=["id","para_text"], how="left")
    df_merged_details = pd.merge(df_merged_vendor_details, df_purchaser_list, on=["id","para_text"], how="left")
    # print(df_merged_details[["k_means_labels_primary", "p_pan"]])
    # print(df_merged_details[["k_means_labels_primary", "para_text", "p_pan"]].to_string())
    return df_merged_details



def get_pan(entry):
    para_text = entry["para_text"]
    pan_pattern = r'[A-Z][A-Z][A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9][A-Z]'
    person_pan = re.findall(pan_pattern,para_text)
    # if len(person_pan) > 0:
    #     print(person_pan[0])
    return person_pan


def get_aadhar(entry):
    para_text = entry["para_text"]
    aadhar_pattern = r'[0-9]{4}\s[0-9]{4}\s[0-9]{4}'
    person_aadhar = re.findall(aadhar_pattern, para_text)
    # if len(person_aadhar) > 0:
    #     print(person_aadhar[0] + "\n\n")
    return person_aadhar


def get_name(entry):
    para_text = entry["para_text"]
    name_pattern = r'(.+?(?=D\/o))|(.+?(?=S\/o))|(.+?(?=W\/o))|(.+?(?=C\/o))'
    regex_results = re.findall(name_pattern, para_text)

    person_name = ""
    if len(regex_results) > 0:
        for name in regex_results[0]:
            if len(name.strip()) > 1:
                person_name = re.sub(r'Sri. |Sri.|SRI\.|SRI\. |Smt. |Smt.|SMT.|SMT. ','',name)
                person_name = re.sub(r'\(.*?\)', '', person_name)
                person_name = person_name.strip().replace(',','')
                person_name = person_name.split("aged")[0].strip()

        print(person_name)
    return person_name
def get_vendor_purchaser_details(df_hierchical_clustered_output):
    df_vendor_purchaser_details = df_hierchical_clustered_output[df_hierchical_clustered_output['k_means_labels_primary'] == 4][["id", "filename", "para_text"]]

    vendor_detail_list = []
    for index, row in df_vendor_purchaser_details.iterrows():
        if "VENDOR" in row["para_text"]:
            vendor_detail = {}
            vendor_detail["id"] = row["id"]
            vendor_detail["para_text"] = row["para_text"]
            vendor_detail["v_name"] = get_name(row)
            if len(vendor_detail["v_name"]) > 0:
                vendor_detail["v_pan"] = get_pan(row)
                vendor_detail["v_aadhar"] = get_aadhar(row)

            vendor_detail_list.append(vendor_detail)

    purchaser_detail_list = []
    for index, row in df_vendor_purchaser_details.iterrows():
        if "PURCHASER" in row["para_text"]:
            purchaser_detail = {}
            purchaser_detail["id"] = row["id"]
            purchaser_detail["para_text"] = row["para_text"]
            purchaser_detail["p_name"] = get_name(row)
            if len(purchaser_detail["p_name"]) > 0:
                purchaser_detail["p_pan"] = get_pan(row)
                purchaser_detail["p_aadhar"] = get_aadhar(row)

            purchaser_detail_list.append(purchaser_detail)

    # print("-------------------- VENDOR DETAILS -----------------------")
    # print(vendor_detail_list)
    # print("-------------------- PURCHASER DETAILS -----------------------")
    # print(purchaser_detail_list)
    df_final_merged_details = create_vendor_purchaser_dataframe(vendor_detail_list, purchaser_detail_list,df_hierchical_clustered_output)
    return df_final_merged_details

if __name__ == '__main__':
    file_encoding_list = []
    list_of_files_to_process = ['/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/GOLDEN SPRINGS SITE No.03.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/GOLDEN SPRINGS SITE No. 17.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/GOLDEN SPRINGS SITE NO. 35.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/GOLDEN SPRINGS SITE NO 09.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs site no- 54.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/Site no.  48  (golden springs).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/Site no.  49 (golden springs).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/Site no. 40  (golden springs).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/Site no. 42 swarnambhana.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/Site no. 52 (golden springs).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 06 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 11 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 13 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 16 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 21 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 22 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 25 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 26 (GOLDEN SPRINGS)(JOINT PURCHASERS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 38 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 39 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 42 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 49 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 57 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 58 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 59 (GOLDEN SPRINGS).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 28 (GOLDEN SPRINGS) (NAVEEN.A.S).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 29 (GOLDEN SPRINGS) (PALLAVI.S).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 30 (GOLDEN SPRINGS) (CHARANA.A.V).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/ST NO. 40 GOLDEN SPRINGS (SALE DEED).docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -MEGHANA.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No- 07.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No- 12.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-5.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-08.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-14.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-15.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-19.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-24.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-33.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-41.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-43.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-45.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-51.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL -Site No-53.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL SITE NO- 36.docx',
                                '/Users/shashankshandilya/PycharmProjects/Information-Extraction/Final copy/golden springs FINAL.docx']



    df_encoded_output = pd.DataFrame()
    for filename in list_of_files_to_process:
        paragraphs = paragraph_extractor( filename=filename )
        paragraph_encoding = encode(paragraphs, filename=filename)
        df_encoded_output = pd.concat([df_encoded_output, pd.DataFrame(paragraph_encoding)])

    df_clustered_output = cluster_paragraphs(df_encoded_output, 7)
    df_hierchical_clustered_output = second_level_clustering(df_clustered_output, 6, 4)
    df_final_details = get_vendor_purchaser_details(df_hierchical_clustered_output)
    # print(df_final_details)

    # df_clustered_output[["filename", "para_text", "k_means_labels"]].to_csv("/Users/shashankshandilya/PycharmProjects/Information-Extraction/Output_vector_files/first_level_clusters_21_02.csv")
    # df_hierchical_clustered_output[["filename", "para_text", "k_means_labels_primary", "k_means_labels_secondary"]].to_csv("/Users/shashankshandilya/PycharmProjects/Information-Extraction/Output_vector_files/hierachical_cluster_output_21_02.csv")
    df_final_details[["filename", "para_text","k_means_labels_primary", "k_means_labels_secondary","p_name","p_pan","p_aadhar","v_name","v_pan","v_aadhar"]].to_csv("/Users/shashankshandilya/PycharmProjects/Information-Extraction/Output_vector_files/test.csv")